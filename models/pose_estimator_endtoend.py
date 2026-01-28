"""
6D Pose Estimation Model

This module implements the PoseEstimator model using ResNet-50 as backbone
for predicting 6D object pose (rotation as quaternion + translation).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple, Optional

from config import Config


class PoseEstimator(nn.Module):
    """
    6D Pose Estimator using ResNet-50 backbone.
    
    Predicts:
        - Rotation as quaternion (4D): [qw, qx, qy, qz]
        - Translation vector (3D): [tx, ty, tz]
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights (default: from Config.POSE_PRETRAINED)
        dropout: Dropout probability (default: from Config.POSE_DROPOUT)
        freeze_backbone: If True, freeze backbone and only train head (default: from Config.POSE_FREEZE_BACKBONE)
    """
    
    def __init__(self, pretrained: bool = None, dropout: float = None, freeze_backbone: bool = None):
        super(PoseEstimator, self).__init__()
        
        # Use Config defaults if not specified
        pretrained = pretrained if pretrained is not None else Config.POSE_PRETRAINED
        dropout = dropout if dropout is not None else Config.POSE_DROPOUT
        freeze_backbone = freeze_backbone if freeze_backbone is not None else Config.POSE_FREEZE_BACKBONE
        
        # Load ResNet-50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove final FC layer
        # ResNet-50 outputs 2048-dim features before FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optionally freeze backbone for faster training (only train head)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature dimension from ResNet-50
        self.feature_dim = 2048
        
        # Regression head for 6D pose
        # Output: 4 (quaternion) + 3 (translation) = 7 values
        self.pose_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 7)  # 4 (quat) + 3 (trans)
        )
        
        print(f"   PoseEstimator initialized")
        print(f"   Backbone: {Config.POSE_BACKBONE} (pretrained={pretrained}, frozen={freeze_backbone})")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Output: 7 values (4 quaternion + 3 translation)")
        print(f"   Dropout: {dropout}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W) where H=W=Config.POSE_IMAGE_SIZE (default: 224)
            
        Returns:
            quaternion: Normalized quaternion (B, 4)
            translation: Translation vector (B, 3)
        """
        # Extract features with backbone
        features = self.backbone(x)  # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        
        # Predict pose
        pose = self.pose_head(features)  # (B, 7)
        
        # Split into quaternion and translation
        quaternion = pose[:, :4]  # (B, 4)
        translation = pose[:, 4:]  # (B, 3)
        
        # Normalize quaternion to unit length
        quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
        
        return quaternion, translation
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict pose for input images.
        
        Args:
            x: Input images (B, 3, H, W) where H=W=Config.POSE_IMAGE_SIZE (default: 224)
            
        Returns:
            Dictionary with 'quaternion' and 'translation' keys
        """
        self.eval()
        with torch.no_grad():
            quaternion, translation = self.forward(x)
        
        return {
            'quaternion': quaternion,
            'translation': translation
        }
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params
        }


def create_pose_estimator(pretrained: bool = None, dropout: float = None, 
                         freeze_backbone: bool = None, device: Optional[str] = None) -> PoseEstimator:
    """
    Create and initialize PoseEstimator model.
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights (default: from Config.POSE_PRETRAINED)
        dropout: Dropout probability (default: from Config.POSE_DROPOUT)
        freeze_backbone: If True, freeze backbone and only train head (default: from Config.POSE_FREEZE_BACKBONE)
        device: Device to move model to (default: from Config.DEVICE)
        
    Returns:
        PoseEstimator model
    """
    # Use Config defaults if not specified
    if device is None:
        device = Config.DEVICE
    
    model = PoseEstimator(pretrained=pretrained, dropout=dropout, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # Print model info
    params_info = model.get_num_parameters()
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {params_info['total']:,}")
    print(f"   Trainable: {params_info['trainable']:,}")
    
    return model
