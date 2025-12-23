"""
Baseline 6D Pose Estimation Model (Pinhole + ResNet Rotation)

This module implements the BASELINE model as required by the professor:
    YOLO → Bounding Boxes → Translation (Pinhole Camera Model) → ResNet → 6D Pose

Key differences from end-to-end model:
    - NO translation head (translation computed geometrically with Pinhole)
    - Only rotation head (quaternion prediction)
    - Lighter and faster to train
    - Deterministic translation (no learning required)
    - All teams have same baseline for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional

from config import Config


class PoseEstimatorBaseline(nn.Module):
    """
    Baseline 6D Pose Estimator using ResNet-50 for rotation only.
    
    Baseline approach:
        - Translation: Computed with Pinhole Camera Model (NO neural network)
        - Rotation: Predicted by ResNet-50 → Quaternion
    
    Predicts:
        - Rotation as quaternion (4D): [qw, qx, qy, qz]
        - Translation is computed separately using utils.pinhole.compute_translation_pinhole()
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights (default: from Config.POSE_PRETRAINED)
        dropout: Dropout probability (default: from Config.POSE_DROPOUT)
        freeze_backbone: If True, freeze backbone and only train head (default: from Config.POSE_FREEZE_BACKBONE)
    """
    
    def __init__(self, pretrained: bool = None, dropout: float = None, freeze_backbone: bool = None):
        super(PoseEstimatorBaseline, self).__init__()

        # Use Config defaults if not specified
        pretrained = pretrained if pretrained is not None else Config.POSE_PRETRAINED
        dropout = dropout if dropout is not None else Config.POSE_DROPOUT
        freeze_backbone = freeze_backbone if freeze_backbone is not None else Config.POSE_FREEZE_BACKBONE

        # Handle torchvision weights API (deprecates 'pretrained')
        from torchvision.models import ResNet50_Weights
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        resnet = models.resnet50(weights=weights)

        # Remove final FC layer
        # ResNet-50 outputs 2048-dim features before FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Optionally freeze backbone for faster training (only train head)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Feature dimension from ResNet-50
        self.feature_dim = 2048

        # BASELINE: Solo rotation head (quaternion)
        # NO translation head (calcolato con Pinhole Camera Model)
        self.quaternion_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 4)  # Solo 4 valori per quaternion
        )

        print(f"PoseEstimatorBaseline initialized (BASELINE MODEL)")
        print(f"   Backbone: {Config.POSE_BACKBONE} (weights={weights}, frozen={freeze_backbone})")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   Output: 4 values (quaternion only)")
        print(f"   Translation: Computed with Pinhole Camera Model (NOT learned)")
        print(f"   Dropout: {dropout}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - predice solo rotation (quaternion).
        
        Args:
            x: Input images (B, 3, H, W) where H=W=Config.POSE_IMAGE_SIZE (default: 224)
            
        Returns:
            quaternion: Normalized quaternion (B, 4) in format [qw, qx, qy, qz]
        
        Note:
            Translation è calcolata separatamente con Pinhole Camera Model:
            - X = Z * (u - cx) / fx
            - Y = Z * (v - cy) / fy
            - Z = median(depth_map[bbox])
        """
        # Extract features with backbone
        features = self.backbone(x)  # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        
        # Predict only rotation (quaternion)
        quaternion = self.quaternion_head(features)  # (B, 4)
        
        # CRITICAL: Normalize quaternion to unit length (norm = 1)
        # Questo è essenziale per avere quaternion validi
        quaternion = F.normalize(quaternion, p=2, dim=1)
        
        return quaternion
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict rotation for input images.
        
        Args:
            x: Input images (B, 3, H, W) where H=W=Config.POSE_IMAGE_SIZE (default: 224)
            
        Returns:
            Dictionary with 'quaternion' key
            
        Note:
            Per ottenere la 6D pose completa, usa:
            1. quaternion = model.predict(cropped_image)['quaternion']
            2. translation = compute_translation_pinhole(bbox, depth_path, intrinsics)
            3. pose_6d = (quaternion, translation)
        """
        self.eval()
        with torch.no_grad():
            quaternion = self.forward(x)
        
        return {'quaternion': quaternion}
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params
        }


def create_pose_estimator_baseline(
    pretrained: bool = None,
    dropout: float = None, 
    freeze_backbone: bool = None,
    device: Optional[str] = None
) -> PoseEstimatorBaseline:
    """
    Create and initialize PoseEstimatorBaseline model.
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights (default: from Config.POSE_PRETRAINED)
        dropout: Dropout probability (default: from Config.POSE_DROPOUT)
        freeze_backbone: If True, freeze backbone and only train head (default: from Config.POSE_FREEZE_BACKBONE)
        device: Device to move model to (default: from Config.DEVICE)
        
    Returns:
        PoseEstimatorBaseline model
        
    Example:
        >>> model = create_pose_estimator_baseline(pretrained=True, freeze_backbone=False)
        >>> model.eval()
        >>> 
        >>> # Prediction workflow:
        >>> # 1. YOLO detection
        >>> detections = yolo_detector.detect(image)
        >>> bbox = detections[0]['bbox']
        >>> 
        >>> # 2. Crop and preprocess
        >>> cropped = crop_image_from_bbox(image, bbox)
        >>> cropped_tensor = transforms(cropped).unsqueeze(0).to(device)
        >>> 
        >>> # 3. Predict rotation with ResNet
        >>> quaternion = model(cropped_tensor)
        >>> 
        >>> # 4. Compute translation with Pinhole
        >>> translation = compute_translation_pinhole(bbox, depth_path, intrinsics)
        >>> 
        >>> # 5. Combine to get 6D pose
        >>> pose_6d = (quaternion.cpu().numpy()[0], translation)
    """
    # Use Config defaults if not specified
    if device is None:
        device = Config.DEVICE
    
    model = PoseEstimatorBaseline(
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    # Print model info
    params_info = model.get_num_parameters()
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {params_info['total']:,}")
    print(f"   Trainable: {params_info['trainable']:,}")
    print(f"\n💡 Baseline Model Ready:")
    print(f"   - Rotation: Learned by ResNet (quaternion)")
    print(f"   - Translation: Computed with Pinhole Camera Model")
    print(f"   - Conforme alle specifiche del professore ✅")
    
    return model