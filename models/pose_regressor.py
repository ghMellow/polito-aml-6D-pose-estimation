"""
Pose Regressor MLP for 6D Pose Estimation

MLP head that takes fused RGB-D features and predicts 6D pose:
- Quaternion (4D): [qw, qx, qy, qz] normalized to unit norm
- Translation (3D): [tx, ty, tz] in meters

Usage:
    regressor = PoseRegressor(input_dim=2304)
    pose = regressor(f_fused)  # (B, 2304) → (B, 7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRegressor(nn.Module):
    """
    MLP regressor for 6D pose estimation from fused features.
    
    Architecture:
        Linear(input_dim → 1024) → BN → ReLU → Dropout
        Linear(1024 → 512) → BN → ReLU → Dropout
        Linear(512 → 256) → BN → ReLU → Dropout
        Linear(256 → 7)  # 4 quaternion + 3 translation
    
    Args:
        input_dim: Input feature dimension (default: 2304 = 2048 RGB + 256 Depth)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_dim: int = 2304, dropout: float = 0.1):
        super(PoseRegressor, self).__init__()
        
        self.input_dim = input_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            
            nn.Linear(256, 7)  # 4 quaternion + 3 translation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Fused features (B, input_dim)
            
        Returns:
            pose: (B, 7) with [qw, qx, qy, qz, tx, ty, tz]
                  Quaternion is normalized to unit norm.
        """
        out = self.mlp(x)  # (B, 7)
        
        # Normalize quaternion to unit norm
        quat = out[:, :4]
        quat = F.normalize(quat, p=2, dim=1)
        trans = out[:, 4:]
        
        return torch.cat([quat, trans], dim=1)
