import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class PoseLoss(nn.Module):
    def __init__(self, lambda_trans=None, lambda_rot=None):
        super().__init__()
        self.lambda_trans = lambda_trans if lambda_trans is not None else Config.LAMBDA_TRANS
        self.lambda_rot   = lambda_rot   if lambda_rot   is not None else Config.LAMBDA_ROT

    def translation_loss(self, pred_t, gt_t):
        # Work in mm so SmoothL1 beta has mm meaning
        pred_mm = pred_t * 1000.0
        gt_mm   = gt_t   * 1000.0
        return F.smooth_l1_loss(pred_mm, gt_mm, beta=1.0)  # 1mm transition

    def rotation_loss(self, pred_q, gt_q):
        # Normalize BOTH (robust against small drift / dataset issues)
        pred_q = F.normalize(pred_q, p=2, dim=1)
        gt_q   = F.normalize(gt_q,   p=2, dim=1)

        dot = torch.abs(torch.sum(pred_q * gt_q, dim=1))
        dot = torch.clamp(dot, 0.0, 1.0)

        angle = 2.0 * torch.acos(dot)  # true SO(3) angle in radians
        return angle.mean()

    def forward(self, pred_q, pred_t, gt_q, gt_t):
        loss_trans = self.translation_loss(pred_t, gt_t)
        loss_rot   = self.rotation_loss(pred_q, gt_q)
        loss_total = self.lambda_trans * loss_trans + self.lambda_rot * loss_rot
        return {"total": loss_total, "trans": loss_trans, "rot": loss_rot}


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