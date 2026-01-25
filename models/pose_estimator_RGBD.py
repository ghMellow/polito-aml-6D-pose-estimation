"""
RGB-D Fusion Model for 6D Pose Estimation

End-to-end RGB-D fusion architecture for 6D object pose 
estimation trained on the LineMOD dataset. The model combines:
- RGB features (ResNet-18 backbone)
- Depth features (custom DepthEncoder CNN)
- Meta features (bounding box + camera intrinsics + custom MLP)

Architecture:
    RGB Branch:   ResNet-18 (ImageNet pretrained) → 512-dim
    Depth Branch: DepthEncoder (custom CNN)       → 512-dim
    Meta Branch:  MetaEncoder (MLP)               → 32-dim
    Fusion:       Concatenation                   → 1056-dim
    Regressor:    PoseRegressor (MLP)             → 7-dim

Output: 7-dim pose vector [qw, qx, qy, qz, tx, ty, tz]
- Quaternion (4D): rotation as unit quaternion (normalized)
- Translation (3D): object position in camera frame (meters)

Author: Alessandro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Optional, Dict, Union
from pathlib import Path


# external custom models
from models.depth_encoder import DepthEncoder
from models.meta_encoder import MetaEncoder
from models.pose_regressor import PoseRegressor


class RGBDFusionModel(nn.Module):
    """
    Complete RGB-D Fusion Model for 6D Pose Estimation.
    
    This model combines three branches:
    1. RGB Branch: ResNet-18 pretrained on ImageNet → 512-dim features
    2. Depth Branch: Custom DepthEncoder CNN → 512-dim features
    3. Meta Branch: MLP for bbox + intrinsics → 32-dim features
    
    Features are fused via late concatenation: [f_rgb, f_depth, f_meta] → 1056-dim
    A pose regressor MLP then predicts the 7-dim pose (quaternion + translation).
    
    Example:
        >>> model = RGBDFusionModel()
        >>> model.load_weights('checkpoints/pose/fusion_rgbd_512/best.pt')
        >>> model.eval()
        >>> 
        >>> pose = model(rgb_tensor, depth_tensor, meta_tensor)
        >>> quaternion = pose[:, :4]   # [qw, qx, qy, qz]
        >>> translation = pose[:, 4:]  # [tx, ty, tz] in meters
    """
    
    def __init__(
        self,
        rgb_output_dim: int = 512,
        depth_output_dim: int = 512,
        meta_output_dim: int = 32,
        meta_input_dim: int = 10,
        pose_dropout: float = 0.3,
        meta_dropout: float = 0.1,
        pretrained_rgb: bool = True
    ):
        """
        Initial configuration of the RGB-D Fusion Model.
        
        Args:
            rgb_output_dim: Output dimension of RGB encoder (default: 512)
            depth_output_dim: Output dimension of depth encoder (default: 512)
            meta_output_dim: Output dimension of meta encoder (default: 32)
            meta_input_dim: Input dimension of meta encoder (default: 10)
            pose_dropout: Dropout for pose regressor (default: 0.3)
            meta_dropout: Dropout for meta encoder (default: 0.1)
            pretrained_rgb: Use ImageNet pretrained weights for RGB encoder (default: True)
        """
        super().__init__()
        
        self.rgb_output_dim = rgb_output_dim
        self.depth_output_dim = depth_output_dim
        self.meta_output_dim = meta_output_dim
        self.meta_input_dim = meta_input_dim
        
        self.fused_dim = rgb_output_dim + depth_output_dim + meta_output_dim
        
        # =====================================================================
        # RGB Encoder: ResNet-18
        # =====================================================================
        if pretrained_rgb:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # Remove final FC layer to get 512-dim feature vector
        # children(): conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # =====================================================================
        # Depth Encoder: Custom CNN (from models/depth_encoder.py)
        # =====================================================================
        # Processes single-channel depth images → 512-dim features
        self.depth_encoder = DepthEncoder(output_dim=depth_output_dim)
        
        # =====================================================================
        # Meta Encoder: MLP (from models/meta_encoder.py)
        # =====================================================================
        # Encodes bbox location + camera intrinsics to a 32-dim tensor
        # Input should be generated with build_crop_meta()
        # Input: 10-dim [uc, vc, wn, hn, area_n, ar, fx_n, fy_n, cx_n, cy_n]
        self.meta_encoder = MetaEncoder(
            input_dim=meta_input_dim,
            output_dim=meta_output_dim,
            dropout=meta_dropout
        )
        
        # =====================================================================
        # Pose Regressor: MLP (from models/pose_regressor.py)
        # =====================================================================
        # Input: 1056-dim fused features
        # Output: 7-dim pose [qw, qx, qy, qz, tx, ty, tz]
        self.pose_regressor = PoseRegressor(
            input_dim=self.fused_dim,
            dropout=pose_dropout
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        meta: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the RGB-D Fusion model.
        
        Args:
            rgb: RGB image tensor (B, 3, 224, 224)
                 Expected to be normalized with ImageNet mean/std
            depth: Depth image tensor (B, 1, 224, 224)
                   Expected to be normalized to [0, 1] (depth_mm / 2000.0)
            meta: Metadata tensor (B, 10)
                  Contains normalized bbox and camera intrinsics
        
        Returns:
            pose: Predicted pose tensor (B, 7)
                  Format: [qw, qx, qy, qz, tx, ty, tz]
                  - Quaternion is normalized to unit norm
                  - Translation is in meters
        """
        # Extract RGB features: (B, 512)
        # ResNet output is (B, 512, 1, 1), squeeze spatial dims
        f_rgb = self.rgb_encoder(rgb)                # (B, 512, 1, 1)
        f_rgb = f_rgb.squeeze(-1).squeeze(-1)        # (B, 512)
        
        # Extract depth features: (B, 512)
        f_depth = self.depth_encoder(depth)          # (B, 512)
        
        # Extract meta features: (B, 32)
        f_meta = self.meta_encoder(meta)             # (B, 32)
        
        # Late fusion: concatenate all features
        f_fused = torch.cat([f_rgb, f_depth, f_meta], dim=1)  # (B, 1056)
        
        # Predict pose: (B, 7)
        pose = self.pose_regressor(f_fused)          # (B, 7)
        
        return pose
    
    def extract_features(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        meta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features (useful for analysis/visualization).
        
        Args:
            rgb: RGB image tensor (B, 3, H, W)
            depth: Depth image tensor (B, 1, H, W)
            meta: Metadata tensor (B, 10)
            
        Returns:
            Dictionary with feature tensors:
                - 'rgb_features': (B, 512)
                - 'depth_features': (B, 512)
                - 'meta_features': (B, 32)
                - 'fused_features': (B, 1056)
        """
        f_rgb = self.rgb_encoder(rgb).squeeze(-1).squeeze(-1)
        f_depth = self.depth_encoder(depth)
        f_meta = self.meta_encoder(meta)
        f_fused = torch.cat([f_rgb, f_depth, f_meta], dim=1)
        
        return {
            'rgb_features': f_rgb,
            'depth_features': f_depth,
            'meta_features': f_meta,
            'fused_features': f_fused
        }
    
    def load_weights(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        strict: bool = True
    ) -> Dict:
        """
        Load model weights from a single checkpoint file.
        
        Checkpoint must provide following values:
        {
            'rgb_encoder': state_dict,
            'depth_encoder': state_dict,
            'meta_encoder': state_dict,
            'pose_regressor': state_dict,
            ...
        }
        
        Args:
            checkpoint_path: .pt checkpoint file path
            device: Device to load weights to (default: auto-detect)
            strict: Whether to strictly enforce state_dict keys match (default: True)
            
        Returns:
            The loaded checkpoint dictionary (for accessing metadata like epoch, val_loss)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.rgb_encoder.load_state_dict(checkpoint['rgb_encoder'], strict=strict)
        self.depth_encoder.load_state_dict(checkpoint['depth_encoder'], strict=strict)
        self.meta_encoder.load_state_dict(checkpoint['meta_encoder'], strict=strict)
        self.pose_regressor.load_state_dict(checkpoint['pose_regressor'], strict=strict)
        
        print(f"Weights successfully loaded from: {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch'] + 1}")
        if 'val_loss' in checkpoint:
            print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint
    
    def save_weights(
        self,
        save_path: Union[str, Path],
        epoch: int = 0,
        val_loss: float = 0.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **extra_info
    ) -> None:
        """
        Save model weights to a checkpoint file.
        
        Args:
            save_path: Path to save the .pt checkpoint file
            epoch: Current epoch number (for logging)
            val_loss: Validation loss (for logging)
            optimizer: Optional optimizer state to save
            **extra_info: Additional metadata to save in checkpoint
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'rgb_encoder': self.rgb_encoder.state_dict(),
            'depth_encoder': self.depth_encoder.state_dict(),
            'meta_encoder': self.meta_encoder.state_dict(),
            'pose_regressor': self.pose_regressor.state_dict(),
            'val_loss': val_loss
        }
        
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        # Add any extra info
        checkpoint.update(extra_info)
        
        torch.save(checkpoint, save_path)
        print(f"Weights succcessfully saved to: {save_path}")
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.
        
        Returns:
            Dictionary with parameter counts for each module and total
        """
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'rgb_encoder': count(self.rgb_encoder),
            'depth_encoder': count(self.depth_encoder),
            'meta_encoder': count(self.meta_encoder),
            'pose_regressor': count(self.pose_regressor),
            'total': count(self),
            'trainable': count_trainable(self)
        }


# =============================================================================
# Utility Functions
# =============================================================================

def build_crop_meta(
    bbox_xywh: torch.Tensor,
    cam_K: torch.Tensor,
    img_h: int,
    img_w: int
) -> torch.Tensor:
    """
    Build scalar metadata for translation disambiguation.
    
    This function creates a 10-dimensional metadata vector from:
    - bbox location and size (normalized via image dimensions)
    - Camera intrinsics (normalized via image dimensions)
    
    Args:
        bbox_xywh: bbox tensor [x, y, w, h] in pixels (4,)
        cam_K: Camera intrinsic matrix (3, 3)
        img_h: Image height in pixels
        img_w: Image width in pixels
        
    Returns:
        Metadata tensor (10,):
        - uc, vc: normalized bbox center (0-1)
        - wn, hn: normalized bbox width/height (0-1)
        - area_n: normalized bbox area (0-1)
        - ar: aspect ratio (w/h)
        - fx_n, fy_n: normalized focal lengths
        - cx_n, cy_n: normalized principal point
    """
    x, y, w, h = bbox_xywh.float()
    W = float(img_w)
    H = float(img_h)
    
    # Compute normalized bbox center (in [0, 1])
    uc = (x + 0.5 * w) / (W + 1e-6)
    vc = (y + 0.5 * h) / (H + 1e-6)
    
    # Compute normalized bbox size and derived features
    wn = w / (W + 1e-6)
    hn = h / (H + 1e-6)
    area_n = (w * h) / ((W * H) + 1e-6)
    ar = w / (h + 1e-6)  # aspect ratio
    
    # Extract and normalize camera intrinsics
    fx = cam_K[0, 0].float()
    fy = cam_K[1, 1].float()
    cx = cam_K[0, 2].float()
    cy = cam_K[1, 2].float()
    
    fx_n = fx / (W + 1e-6)
    fy_n = fy / (H + 1e-6)
    cx_n = cx / (W + 1e-6)
    cy_n = cy / (H + 1e-6)
    
    # Stack into single tensor
    meta = torch.stack([uc, vc, wn, hn, area_n, ar, fx_n, fy_n, cx_n, cy_n], dim=0)
     
    return meta


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RGB-D Fusion Model for 6D Pose Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test model with random weights
    python rgbd_fusion_model.py
    
    # Load trained weights
    python rgbd_fusion_model.py --checkpoint checkpoints/pose/fusion_rgbd_512/best.pt
    
    # Specify device
    python rgbd_fusion_model.py --checkpoint best.pt --device cuda
        """
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, default=None,
        help='Path to checkpoint file (.pt)'
    )
    parser.add_argument(
        '--device', '-d', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use (default: auto)'
    )
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("RGB-D Fusion Model for 6D Pose Estimation")
    print("=" * 60)
    print(f"\nDevice: {device}")
    
    # Initialize model
    # Use pretrained_rgb=False if loading checkpoint (weights will be replaced anyway)
    model = RGBDFusionModel(pretrained_rgb=(args.checkpoint is None))
    model = model.to(device)
    
    # Load weights if checkpoint provided
    if args.checkpoint:
        checkpoint = model.load_weights(args.checkpoint, device=device)
    else:
        print("⚠️  No checkpoint provided, using random/pretrained weights")
    
    model.eval()
    
    # Print model summary
    params = model.count_parameters()
    print(f"\n📊 Model Parameters:")
    print(f"   RGB Encoder:    {params['rgb_encoder']:>10,}")
    print(f"   Depth Encoder:  {params['depth_encoder']:>10,}")
    print(f"   Meta Encoder:   {params['meta_encoder']:>10,}")
    print(f"   Pose Regressor: {params['pose_regressor']:>10,}")
    print(f"   {'─' * 25}")
    print(f"   Total:          {params['total']:>10,}")
    print(f"   Trainable:      {params['trainable']:>10,}")
    
    print(f"\n📐 Feature Dimensions:")
    print(f"   RGB:    {model.rgb_output_dim}")
    print(f"   Depth:  {model.depth_output_dim}")
    print(f"   Meta:   {model.meta_output_dim}")
    print(f"   Fused:  {model.fused_dim}")
    
    # Test forward pass
    print("\n" + "-" * 60)
    print("🧪 Testing forward pass with dummy data...")
    
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    depth = torch.randn(batch_size, 1, 224, 224).to(device)
    meta = torch.randn(batch_size, 10).to(device)
    
    with torch.no_grad():
        pose = model(rgb, depth, meta)
    
    print(f"\n📥 Input shapes:")
    print(f"   RGB:   {tuple(rgb.shape)}")
    print(f"   Depth: {tuple(depth.shape)}")
    print(f"   Meta:  {tuple(meta.shape)}")
    
    print(f"\n📤 Output shape: {tuple(pose.shape)}")
    
    # Verify quaternion normalization
    quat_norms = torch.norm(pose[:, :4], dim=1)
    print(f"\n🔍 Quaternion norms (should be ~1.0): {quat_norms.cpu().numpy()}")
    
    print(f"\n📋 Sample output (first sample):")
    print(f"   Quaternion [qw, qx, qy, qz]: {pose[0, :4].cpu().numpy()}")
    print(f"   Translation [tx, ty, tz]:    {pose[0, 4:].cpu().numpy()}")
    
    print("\n✅ Model test complete!")