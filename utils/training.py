"""
Training utilities for 6D Pose Estimation models (ResNet, etc.)
Centralizes the training loop for both rotation-only and rotation+translation models.
"""

import torch
from tqdm import tqdm
import yaml
import os
import pandas as pd



# --- Utility: save args and results header ---
def _init_training_dirs_and_files(checkpoint_dir, training_config, csv_header):
    os.makedirs(checkpoint_dir, exist_ok=True)
    weights_dir = os.path.join(checkpoint_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    results_file = os.path.join(checkpoint_dir, 'results.csv')
    if training_config is not None:
        args_path = os.path.join(checkpoint_dir, 'args.yaml')
        with open(args_path, 'w') as f:
            yaml.dump(training_config, f)
    with open(results_file, 'w') as f:
        f.write(csv_header + '\n')
    return weights_dir, results_file

# --- Baseline: only rotation ---
def train_pose_baseline(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    accumulation_steps=1,
    checkpoint_dir=None,
    training_config=None,
    save_best=True,
    save_last=True,
    verbose=True
):
    """
    Training loop for baseline (rotation only).
    """
    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    best_epoch = 0
    if checkpoint_dir is not None:
        weights_dir, results_file = _init_training_dirs_and_files(checkpoint_dir, training_config, 'epoch,train_loss,val_loss')
    else:
        results_file = None
        weights_dir = None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['rgb_crop'].to(device)
            gt_quaternion = batch['quaternion'].to(device)
            pred_quaternion = model(images)
            loss_dict = criterion(pred_quaternion, gt_quaternion)
            loss = loss_dict['total'] / accumulation_steps
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
        avg_loss = epoch_loss / len(train_loader)
        # Validation step (if val_loader is provided)
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['rgb_crop'].to(device)
                    gt_quaternion = batch['quaternion'].to(device)
                    pred_quaternion = model(images)
                    loss_dict = criterion(pred_quaternion, gt_quaternion)
                    val_loss += loss_dict['total'].item()
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
        else:
            avg_val_loss = float('nan')
            history['val_loss'].append(avg_val_loss)
        history['train_loss'].append(avg_loss)
        if results_file:
            with open(results_file, 'a') as f:
                f.write(f"{epoch+1},{avg_loss:.6f},{avg_val_loss:.6f}\n")
        if scheduler is not None:
            scheduler.step(avg_loss)
        if verbose:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}")
        # Save last if requested
        if save_last and weights_dir:
            last_path = os.path.join(weights_dir, 'last.pt')
            torch.save(model.state_dict(), last_path)
        # Save best if requested
        if save_best and avg_loss < best_loss and weights_dir:
            best_loss = avg_loss
            best_epoch = epoch
            best_path = os.path.join(weights_dir, 'best.pt')
            torch.save(model.state_dict(), best_path)
            if verbose:
                print(f"💾 Best model salvato in: {best_path}")
    return history, best_loss, best_epoch

# --- Full pose: rotation + translation ---
def train_pose_full(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    accumulation_steps=1,
    use_amp=False,
    checkpoint_dir=None,
    training_config=None,
    save_best=True,
    save_last=True,
    warmup_epochs=0,
    lr_backbone=None,
    lr_head=None,
    verbose=True
):
    """
    Training loop for full pose (rotation + translation).
    """
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    history = {'train_loss': [], 'val_loss': [], 'train_trans': [], 'train_rot': [], 'lr_backbone': [], 'lr_head': []}
    best_loss = float('inf')
    best_epoch = 0
    if checkpoint_dir is not None:
        weights_dir, results_file = _init_training_dirs_and_files(
            checkpoint_dir, training_config,
            'epoch,train_loss,val_loss,train_trans_loss,train_rot_loss,lr_backbone,lr_head')
    else:
        results_file = None
        weights_dir = None
    for epoch in range(epochs):
        # Warmup LR
        if warmup_epochs and epoch < warmup_epochs and lr_backbone and lr_head:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                if 'name' in param_group and param_group['name'] == 'backbone':
                    param_group['lr'] = lr_backbone * warmup_factor
                else:
                    param_group['lr'] = lr_head * warmup_factor
            if verbose:
                print(f"🔥 Warmup epoch {epoch+1}/{warmup_epochs} - LR scale: {warmup_factor:.2f}")
        model.train()
        epoch_loss = 0
        epoch_trans = 0
        epoch_rot = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['rgb_crop'].to(device)
            gt_quaternion = batch['quaternion'].to(device)
            gt_translation = batch.get('translation', None)
            if gt_translation is not None:
                gt_translation = gt_translation.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        pred_quaternion, pred_translation = outputs
                    else:
                        pred_quaternion = outputs
                        pred_translation = None
                    loss_dict = criterion(pred_quaternion, pred_translation, gt_quaternion, gt_translation)
                    loss = loss_dict['total'] / accumulation_steps
            else:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    pred_quaternion, pred_translation = outputs
                else:
                    pred_quaternion = outputs
                    pred_translation = None
                loss_dict = criterion(pred_quaternion, pred_translation, gt_quaternion, gt_translation)
                loss = loss_dict['total'] / accumulation_steps
            # Backward
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # Step
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            # Metrics
            epoch_loss += loss.item() * accumulation_steps
            epoch_trans += loss_dict.get('trans', 0.0)
            epoch_rot += loss_dict.get('rot', 0.0)
            current_lr_backbone = optimizer.param_groups[0]['lr']
            current_lr_head = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'trans': f"{loss_dict.get('trans', 0.0):.4f}",
                'rot': f"{loss_dict.get('rot', 0.0):.2f}",
                'lr_bb': f"{current_lr_backbone:.2e}",
                'lr_hd': f"{current_lr_head:.2e}"
            })
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_trans = epoch_trans / num_batches
        avg_rot = epoch_rot / num_batches
        # Validation step (if val_loader is provided)
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['rgb_crop'].to(device)
                    gt_quaternion = batch['quaternion'].to(device)
                    gt_translation = batch.get('translation', None)
                    if gt_translation is not None:
                        gt_translation = gt_translation.to(device)
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        pred_quaternion, pred_translation = outputs
                    else:
                        pred_quaternion = outputs
                        pred_translation = None
                    loss_dict = criterion(pred_quaternion, pred_translation, gt_quaternion, gt_translation)
                    val_loss += loss_dict['total'].item()
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
        else:
            avg_val_loss = float('nan')
            history['val_loss'].append(avg_val_loss)
        history['train_loss'].append(avg_loss)
        history['train_trans'].append(avg_trans)
        history['train_rot'].append(avg_rot)
        history['lr_backbone'].append(current_lr_backbone)
        history['lr_head'].append(current_lr_head)
        if results_file:
            with open(results_file, 'a') as f:
                f.write(f"{epoch+1},{avg_loss:.6f},{avg_val_loss:.6f},{avg_trans:.6f},{avg_rot:.6f},{current_lr_backbone:.8e},{current_lr_head:.8e}\n")
        if scheduler is not None:
            scheduler.step(avg_loss)
        if verbose:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}")
        # Save last if requested
        if save_last and weights_dir:
            last_path = os.path.join(weights_dir, 'last.pt')
            torch.save(model.state_dict(), last_path)
        # Save best if requested
        if save_best and avg_loss < best_loss and weights_dir:
            best_loss = avg_loss
            best_epoch = epoch
            best_path = os.path.join(weights_dir, 'best.pt')
            torch.save(model.state_dict(), best_path)
            if verbose:
                print(f"💾 Best model salvato in: {best_path}")
        # Early stopping (optional, patience=10)
        if epoch - best_epoch >= 10 and epoch >= (warmup_epochs or 0):
            if verbose:
                print(f"\n⚠️  Early stopping! Loss did not improve for {epoch - best_epoch} epochs")
                print(f"   Best loss: {best_loss:.4f} @ epoch {best_epoch + 1}")
            break
    return history, best_loss, best_epoch
