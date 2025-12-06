"""
Training Script for 6D Pose Estimation

Train PoseEstimator model on LineMOD dataset with:
- AdamW optimizer with CosineAnnealingLR scheduler
- Gradient accumulation for larger effective batch size
- Mixed precision training (FP16)
- Validation with ADD metric
- Checkpoint saving and Wandb logging
"""

import argparse
import os
import sys
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from models.pose_estimator import PoseEstimator
from dataset.custom_dataset import create_pose_dataloaders
from utils.losses import PoseLoss
from utils.metrics import load_all_models, load_models_info, compute_add_batch
from utils.transforms import quaternion_to_rotation_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Train 6D Pose Estimation Model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=str(Config.DATA_ROOT),
                       help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=Config.POSE_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default=Config.POSE_BACKBONE,
                       help='Backbone architecture')
    parser.add_argument('--dropout', type=float, default=Config.POSE_DROPOUT,
                       help='Dropout probability')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use ImageNet pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                       help='Freeze backbone (only train head) - MUCH faster for quick tests')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=Config.POSE_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.POSE_LR,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=Config.POSE_WEIGHT_DECAY,
                       help='Weight decay')
    parser.add_argument('--gradient_accum_steps', type=int, default=Config.GRADIENT_ACCUM_STEPS,
                       help='Gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true', default=Config.USE_AMP,
                       help='Use automatic mixed precision')
    
    # Loss parameters
    parser.add_argument('--lambda_trans', type=float, default=Config.LAMBDA_TRANS,
                       help='Translation loss weight')
    parser.add_argument('--lambda_rot', type=float, default=Config.LAMBDA_ROT,
                       help='Rotation loss weight')
    
    # Evaluation parameters
    parser.add_argument('--val_interval', type=int, default=5,
                       help='Validation interval (epochs)')
    parser.add_argument('--add_threshold', type=float, default=Config.ADD_THRESHOLD,
                       help='ADD metric threshold (fraction of diameter)')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default=str(Config.CHECKPOINT_DIR),
                       help='Checkpoint directory')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Checkpoint save interval (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', default=Config.USE_WANDB,
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default=Config.WANDB_PROJECT,
                       help='Wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for logging')
    
    # Device
    parser.add_argument('--device', type=str, default=Config.DEVICE,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, 
                gradient_accum_steps, use_amp, epoch):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_trans_loss = 0.0
    running_rot_loss = 0.0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        rgb_crop = batch['rgb_crop'].to(device)
        gt_quaternion = batch['quaternion'].to(device)
        gt_translation = batch['translation'].to(device)
        
        # Forward pass with mixed precision
        # Note: MPS doesn't support autocast, so we use regular precision on MPS
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type=device_type):
                pred_quaternion, pred_translation = model(rgb_crop)
                losses = criterion(pred_quaternion, pred_translation, 
                                 gt_quaternion, gt_translation)
                loss = losses['total'] / gradient_accum_steps
        else:
            pred_quaternion, pred_translation = model(rgb_crop)
            losses = criterion(pred_quaternion, pred_translation, 
                             gt_quaternion, gt_translation)
            loss = losses['total'] / gradient_accum_steps
        
        # Backward pass
        if use_amp and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accum_steps == 0:
            if use_amp and device.type == 'cuda':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Update statistics
        running_loss += losses['total'].item()
        running_trans_loss += losses['trans'].item()
        running_rot_loss += losses['rot'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'trans': f"{losses['trans'].item():.4f}",
            'rot': f"{losses['rot'].item():.2f}"
        })
    
    # Average losses
    avg_loss = running_loss / len(train_loader)
    avg_trans_loss = running_trans_loss / len(train_loader)
    avg_rot_loss = running_rot_loss / len(train_loader)
    
    return {
        'loss': avg_loss,
        'trans_loss': avg_trans_loss,
        'rot_loss': avg_rot_loss
    }


def validate(model, val_loader, criterion, models_dict, models_info, device, 
            add_threshold, symmetric_objects):
    """Validate model."""
    model.eval()
    
    running_loss = 0.0
    running_trans_loss = 0.0
    running_rot_loss = 0.0
    
    all_add_values = []
    all_is_correct = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch in pbar:
            # Move data to device
            rgb_crop = batch['rgb_crop'].to(device)
            gt_quaternion = batch['quaternion'].to(device)
            gt_translation = batch['translation'].to(device)
            obj_ids = batch['obj_id'].cpu().numpy().tolist()
            
            # Forward pass
            pred_quaternion, pred_translation = model(rgb_crop)
            
            # Compute loss
            losses = criterion(pred_quaternion, pred_translation, 
                             gt_quaternion, gt_translation)
            
            running_loss += losses['total'].item()
            running_trans_loss += losses['trans'].item()
            running_rot_loss += losses['rot'].item()
            
            # Convert quaternions to rotation matrices for ADD computation
            batch_size = pred_quaternion.shape[0]
            pred_R_batch = []
            gt_R_batch = []
            
            for i in range(batch_size):
                pred_R = quaternion_to_rotation_matrix(pred_quaternion[i])
                gt_R = quaternion_to_rotation_matrix(gt_quaternion[i])
                pred_R_batch.append(pred_R.cpu().numpy())
                gt_R_batch.append(gt_R.cpu().numpy())
            
            pred_R_batch = np.array(pred_R_batch)
            gt_R_batch = np.array(gt_R_batch)
            
            # Compute ADD metric
            add_results = compute_add_batch(
                pred_R_batch,
                pred_translation.cpu().numpy(),
                gt_R_batch,
                gt_translation.cpu().numpy(),
                obj_ids,
                models_dict,
                models_info,
                symmetric_objects=symmetric_objects,
                threshold=add_threshold
            )
            
            all_add_values.extend(add_results['add_values'])
            all_is_correct.extend(add_results['is_correct'])
    
    # Average metrics
    avg_loss = running_loss / len(val_loader)
    avg_trans_loss = running_trans_loss / len(val_loader)
    avg_rot_loss = running_rot_loss / len(val_loader)
    mean_add = np.mean(all_add_values)
    accuracy = np.mean(all_is_correct) * 100
    
    return {
        'loss': avg_loss,
        'trans_loss': avg_trans_loss,
        'rot_loss': avg_rot_loss,
        'mean_add': mean_add,
        'accuracy': accuracy
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Initialize Wandb if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"pose_estimation_{time.strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Create dataloaders
    print(f"\n📦 Loading dataset from: {args.data_dir}")
    train_loader, val_loader = create_pose_dataloaders(
        dataset_root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_margin=Config.POSE_CROP_MARGIN,
        output_size=Config.POSE_IMAGE_SIZE
    )
    
    # Load 3D models for ADD metric
    print(f"\n📐 Loading 3D models for ADD metric...")
    models_dict = load_all_models(Config.MODELS_PATH)
    models_info = load_models_info(Config.MODELS_INFO_PATH)
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = PoseEstimator(pretrained=args.pretrained, dropout=args.dropout, freeze_backbone=args.freeze_backbone)
    model = model.to(device)
    
    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    if args.freeze_backbone:
        print(f"   ⚡ Backbone frozen - training only head (~3-4x faster!)")
    
    # Create loss function
    criterion = PoseLoss(lambda_trans=args.lambda_trans, lambda_rot=args.lambda_rot)
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler (only for CUDA)
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None
    
    # Disable AMP on MPS (not supported)
    if args.use_amp and device.type == 'mps':
        print("⚠️  Mixed precision (AMP) not supported on MPS, using FP32")
        args.use_amp = False
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_add = float('inf')
    
    if args.resume:
        print(f"\n📂 Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_add = checkpoint['metrics'].get('mean_add', float('inf'))
        print(f"✅ Resumed from epoch {start_epoch}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"   Effective batch size: {args.batch_size * args.gradient_accum_steps}")
    print(f"   Mixed precision: {args.use_amp}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            args.gradient_accum_steps, args.use_amp, epoch + 1
        )
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 Epoch {epoch + 1}/{args.epochs} - {epoch_time:.1f}s")
        print(f"   Train Loss: {train_metrics['loss']:.4f}")
        print(f"   Trans Loss: {train_metrics['trans_loss']:.4f}")
        print(f"   Rot Loss: {train_metrics['rot_loss']:.2f}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(
                model, val_loader, criterion, models_dict, models_info,
                device, args.add_threshold, Config.SYMMETRIC_OBJECTS
            )
            
            print(f"\n   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val ADD: {val_metrics['mean_add']:.2f} mm")
            print(f"   Val Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['mean_add'] < best_add:
                best_add = val_metrics['mean_add']
                best_path = checkpoint_dir / 'best_model.pth'
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_path)
                print(f"   ⭐ New best ADD: {best_add:.2f} mm")
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/trans_loss': train_metrics['trans_loss'],
                    'train/rot_loss': train_metrics['rot_loss'],
                    'val/loss': val_metrics['loss'],
                    'val/mean_add': val_metrics['mean_add'],
                    'val/accuracy': val_metrics['accuracy'],
                    'lr': scheduler.get_last_lr()[0]
                })
        else:
            # Log train metrics only
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/trans_loss': train_metrics['trans_loss'],
                    'train/rot_loss': train_metrics['rot_loss'],
                    'lr': scheduler.get_last_lr()[0]
                })
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, checkpoint_path)
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pth'
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, train_metrics, final_path)
    
    print(f"\n✅ Training completed!")
    print(f"   Best ADD: {best_add:.2f} mm")
    print(f"   Models saved in: {checkpoint_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
