import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


def quaternion_geodesic_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """Mean geodesic distance on unit quaternions."""
    pred_q = F.normalize(pred_q, p=2, dim=1)
    gt_q = F.normalize(gt_q, p=2, dim=1)

    dot = torch.abs(torch.sum(pred_q * gt_q, dim=1))
    dot = torch.clamp(dot, 0.0, 1.0)

    angle = 2.0 * torch.acos(dot)  # SO(3) angle in radians
    return angle.mean()


class PoseLoss(nn.Module):
    def __init__(self, lambda_trans: float | None = None, lambda_rot: float | None = None) -> None:
        super().__init__()
        self.lambda_trans = lambda_trans if lambda_trans is not None else Config.LAMBDA_TRANS
        self.lambda_rot = lambda_rot if lambda_rot is not None else Config.LAMBDA_ROT

    def translation_loss(self, pred_t: torch.Tensor, gt_t: torch.Tensor) -> torch.Tensor:
        pred_mm = pred_t * 1000.0
        gt_mm = gt_t * 1000.0
        return F.smooth_l1_loss(pred_mm, gt_mm, beta=1.0)  # beta=1.0 → 1 mm

    def rotation_loss(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
        return quaternion_geodesic_loss(pred_q, gt_q)

    def forward(
        self, pred_q: torch.Tensor, pred_t: torch.Tensor, gt_q: torch.Tensor, gt_t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        loss_trans = self.translation_loss(pred_t, gt_t)
        loss_rot = self.rotation_loss(pred_q, gt_q)
        loss_total = self.lambda_trans * loss_trans + self.lambda_rot * loss_rot
        return {"total": loss_total, "trans": loss_trans, "rot": loss_rot}


class PoseLossBaseline(nn.Module):
    """Rotation-only loss for the baseline pose estimator."""

    def __init__(self) -> None:
        super().__init__()

    def rotation_loss(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
        return quaternion_geodesic_loss(pred_q, gt_q)

    def forward(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_q = pred_q.contiguous()
        gt_q = gt_q.contiguous()
        loss_rot = self.rotation_loss(pred_q, gt_q)
        return {"total": loss_rot, "rot": loss_rot}