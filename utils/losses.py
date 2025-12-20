"""
Loss Functions for 6D Pose Estimation

This module implements loss functions for training pose estimation models:
- Translation loss (L1 smooth loss)
- Rotation loss (geodesic distance on quaternions)
- Combined pose loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class PoseLoss(nn.Module):
    """
    Combined loss for 6D pose estimation.
    
    Combines:
        - Translation loss: Smooth L1 loss
        - Rotation loss: Geodesic distance on quaternions
    
    Args:
        lambda_trans: Weight for translation loss (default: from Config.LAMBDA_TRANS)
        lambda_rot: Weight for rotation loss (default: from Config.LAMBDA_ROT)
    """
    
    def __init__(self, lambda_trans: float = None, lambda_rot: float = None):
        super(PoseLoss, self).__init__()
        
        # Use Config defaults if not specified
        self.lambda_trans = lambda_trans if lambda_trans is not None else Config.LAMBDA_TRANS
        self.lambda_rot = lambda_rot if lambda_rot is not None else Config.LAMBDA_ROT
        
        print(f"✅ PoseLoss initialized")
        print(f"   λ_trans: {lambda_trans}")
        print(f"   λ_rot: {lambda_rot}")
    
    def translation_loss(self, pred_t: torch.Tensor, gt_t: torch.Tensor) -> torch.Tensor:
        """
        Compute translation loss using Smooth L1.
        
        Args:
            pred_t: Predicted translation (B, 3)
            gt_t: Ground truth translation (B, 3)
            
        Returns:
            Translation loss (scalar)
        """
        return F.smooth_l1_loss(pred_t, gt_t)
    
    def rotation_loss(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation loss using geodesic distance on quaternions.
        
        The geodesic distance is:
            d = arccos(|q_pred · q_gt|)
        
        where the dot product gives the cosine of the angle between quaternions.
        
        Args:
            pred_q: Predicted quaternion (B, 4), assumed normalized
            gt_q: Ground truth quaternion (B, 4), assumed normalized
            
        Returns:
            Rotation loss (scalar)
        """
        # Compute dot product (cosine of angle)
        # Take absolute value to handle q and -q representing same rotation
        dot_product = torch.abs(torch.sum(pred_q * gt_q, dim=1))
        
        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Geodesic distance
        angle = torch.acos(dot_product)
        
        # Return mean angle
        return torch.mean(angle)
    
    def forward(
        self,
        pred_q: torch.Tensor,
        pred_t: torch.Tensor,
        gt_q: torch.Tensor,
        gt_t: torch.Tensor
    ) -> dict:
        """
        Compute combined pose loss.
        
        Args:
            pred_q: Predicted quaternion (B, 4)
            pred_t: Predicted translation (B, 3)
            gt_q: Ground truth quaternion (B, 4)
            gt_t: Ground truth translation (B, 3)
            
        Returns:
            Dictionary with 'total', 'trans', 'rot' losses
        """
        # Ensure tensors are contiguous to avoid view/stride issues
        pred_t = pred_t.contiguous()
        gt_t = gt_t.contiguous()
        pred_q = pred_q.contiguous()
        gt_q = gt_q.contiguous()
        
        # Compute individual losses
        loss_trans = self.translation_loss(pred_t, gt_t)  # Now works in meters
        
        # ✅ Scale translation loss to maintain comparable magnitude with millimeters
        # Dataset now provides translations in meters, so we scale loss by 1000
        # to keep it in the same numerical range as before for stable training
        loss_trans = loss_trans * 1000
        
        loss_rot = self.rotation_loss(pred_q, gt_q)
        
        # Combined loss
        loss_total = self.lambda_trans * loss_trans + self.lambda_rot * loss_rot
        
        return {
            'total': loss_total,
            'trans': loss_trans,
            'rot': loss_rot
        }


class PoseLossBaseline(nn.Module):
    """
    Baseline loss for 6D pose estimation (rotation only).
    
    Baseline model:
        - Translation: Computed with Pinhole Camera Model (NO learning)
        - Rotation: Learned by ResNet → Quaternion loss
    
    This loss is used ONLY for rotation training.
    Translation is computed geometrically and not part of the loss.
    
    Args:
        None (only rotation loss, no weighting needed)
    """
    
    def __init__(self):
        super(PoseLossBaseline, self).__init__()
        
        print(f"✅ PoseLossBaseline initialized (BASELINE MODEL)")
        print(f"   Loss: Only rotation (quaternion geodesic distance)")
        print(f"   Translation: Computed with Pinhole (not part of loss)")
    
    def rotation_loss(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation loss using geodesic distance on quaternions.
        
        The geodesic distance is:
            d = arccos(|q_pred · q_gt|)
        
        where the dot product gives the cosine of the angle between quaternions.
        
        Args:
            pred_q: Predicted quaternion (B, 4), assumed normalized
            gt_q: Ground truth quaternion (B, 4), assumed normalized
            
        Returns:
            Rotation loss (scalar)
        """
        # Compute dot product (cosine of angle)
        # Take absolute value to handle q and -q representing same rotation
        dot_product = torch.abs(torch.sum(pred_q * gt_q, dim=1))
        
        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Geodesic distance
        angle = torch.acos(dot_product)
        
        # Return mean angle
        return torch.mean(angle)
    
    def forward(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> dict:
        """
        Compute rotation loss (baseline).
        
        Args:
            pred_q: Predicted quaternion (B, 4)
            gt_q: Ground truth quaternion (B, 4)
            
        Returns:
            Dictionary with 'total' and 'rot' losses
            
        Note:
            'total' == 'rot' since there's no translation loss in baseline.
            Translation is computed with Pinhole Camera Model separately.
        """
        # Ensure tensors are contiguous
        pred_q = pred_q.contiguous()
        gt_q = gt_q.contiguous()
        
        # Compute rotation loss
        loss_rot = self.rotation_loss(pred_q, gt_q)
        
        return {
            'total': loss_rot,
            'rot': loss_rot
        }


class QuaternionL2Loss(nn.Module):
    """
    Alternative rotation loss using L2 distance on quaternions.
    
    Simpler but less geometrically meaningful than geodesic distance.
    """
    
    def __init__(self):
        super(QuaternionL2Loss, self).__init__()
    
    def forward(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 loss on quaternions.
        
        Args:
            pred_q: Predicted quaternion (B, 4)
            gt_q: Ground truth quaternion (B, 4)
            
        Returns:
            L2 loss (scalar)
        """
        # Handle q and -q ambiguity
        loss_pos = F.mse_loss(pred_q, gt_q)
        loss_neg = F.mse_loss(pred_q, -gt_q)
        
        # Take minimum
        return torch.min(loss_pos, loss_neg)


if __name__ == '__main__':
    # Test loss functions
    print("Testing Loss Functions...\n")
    
    # ==================== Test PoseLoss (End-to-End) ====================
    print("=" * 60)
    print("1️⃣  Testing PoseLoss (End-to-End Model)")
    print("=" * 60)
    
    # Create loss (uses Config defaults)
    criterion = PoseLoss()
    
    # Generate random predictions and ground truth
    batch_size = 4
    
    pred_q = torch.randn(batch_size, 4)
    pred_q = pred_q / torch.norm(pred_q, dim=1, keepdim=True)  # Normalize
    
    gt_q = torch.randn(batch_size, 4)
    gt_q = gt_q / torch.norm(gt_q, dim=1, keepdim=True)  # Normalize
    
    pred_t = torch.randn(batch_size, 3) * 100  # Translation in mm
    gt_t = torch.randn(batch_size, 3) * 100
    
    # Compute loss
    losses = criterion(pred_q, pred_t, gt_q, gt_t)
    
    print(f"\n📊 Loss Values (Random Predictions):")
    print(f"   Total: {losses['total'].item():.4f}")
    print(f"   Translation: {losses['trans'].item():.4f}")
    print(f"   Rotation: {losses['rot'].item():.4f} (radians)")
    print(f"   Rotation (degrees): {torch.rad2deg(losses['rot']).item():.2f}°")
    
    # Test with identical predictions (should be ~0)
    print(f"\n🧪 Testing with identical predictions:")
    losses_zero = criterion(gt_q, gt_t, gt_q, gt_t)
    print(f"   Total: {losses_zero['total'].item():.6f}")
    print(f"   Translation: {losses_zero['trans'].item():.6f}")
    print(f"   Rotation: {losses_zero['rot'].item():.6f}")
    
    # ==================== Test PoseLossBaseline (Baseline) ====================
    print(f"\n" + "=" * 60)
    print("2️⃣  Testing PoseLossBaseline (Baseline Model)")
    print("=" * 60)
    
    # Create baseline loss
    criterion_baseline = PoseLossBaseline()
    
    # Compute loss (only rotation)
    losses_baseline = criterion_baseline(pred_q, gt_q)
    
    print(f"\n📊 Loss Values (Random Predictions):")
    print(f"   Total: {losses_baseline['total'].item():.4f}")
    print(f"   Rotation: {losses_baseline['rot'].item():.4f} (radians)")
    print(f"   Rotation (degrees): {torch.rad2deg(losses_baseline['rot']).item():.2f}°")
    print(f"   Note: No translation loss (computed with Pinhole)")
    
    # Test with identical predictions
    print(f"\n🧪 Testing with identical predictions:")
    losses_baseline_zero = criterion_baseline(gt_q, gt_q)
    print(f"   Total: {losses_baseline_zero['total'].item():.6f}")
    print(f"   Rotation: {losses_baseline_zero['rot'].item():.6f}")
    
    # ==================== Comparison ====================
    print(f"\n" + "=" * 60)
    print("3️⃣  Comparison: PoseLoss vs PoseLossBaseline")
    print("=" * 60)
    print(f"\nPoseLoss (End-to-End):")
    print(f"   - Learns: Translation + Rotation")
    print(f"   - Loss components: λ_trans * L_trans + λ_rot * L_rot")
    print(f"   - Training: More complex (2 outputs)")
    print(f"   - Input required: RGB crop only")
    print(f"\nPoseLossBaseline:")
    print(f"   - Learns: Rotation only")
    print(f"   - Loss components: L_rot (quaternion geodesic)")
    print(f"   - Training: Simpler (1 output)")
    print(f"   - Input required: RGB crop + Pinhole for translation")
    print(f"\n✅ Both losses tested successfully!")
    print("=" * 60)
