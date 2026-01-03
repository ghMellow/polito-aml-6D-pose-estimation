"""
Depth Encoder for 6D Pose Estimation

Lightweight CNN encoder for processing depth images.
Inspired by DenseFusion (Wang et al., 2019) depth processing branch,
but simplified to work with 2D depth crops instead of point clouds.

Usage:
    encoder = DepthEncoder(output_dim=256)
    depth_features = encoder(depth_tensor)  # (B, 1, 224, 224) → (B, 256)
"""

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """
    Lightweight CNN encoder for depth images.
    
    Takes single-channel depth input (normalized to [0, 1]) and extracts 
    spatial features suitable for fusion with RGB features.
    
    Architecture:
        Conv2D(1→32) → Conv2D(32→64) → Conv2D(64→128) → Conv2D(128→256) → Conv2D(256→out)
        Each block: Conv + BatchNorm + ReLU + MaxPool
        Finally: AdaptiveAvgPool2d → output_dim feature vector
    
    Args:
        output_dim: Output feature dimension (default: 256)
    """
    
    def __init__(self, output_dim: int = 256):
        super(DepthEncoder, self).__init__()
        
        self.output_dim = output_dim
        
        self.features = nn.Sequential(
            # Block 1: 1 → 32 channels
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 128 → 256 channels
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 256 → output_dim channels
            nn.Conv2d(256, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling → (B, output_dim, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Depth input (B, 1, H, W) normalized to [0, 1]
            
        Returns:
            features: (B, output_dim) depth features
        """
        x = self.features(x)          # (B, output_dim, 1, 1)
        x = x.view(x.size(0), -1)     # (B, output_dim)
        return x
