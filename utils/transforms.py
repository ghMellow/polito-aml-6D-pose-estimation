"""
Transforms and utilities for 6D pose estimation.

Provides rotation conversions (matrix/quaternion), image preprocessing, and data augmentation.
"""

import torch
import numpy as np
import torchvision.transforms as T
from typing import Union

from config import Config


def rotation_matrix_to_quaternion(R: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz].
    
    Args:
        R: Rotation matrix (3, 3).
        
    Returns:
        Quaternion (4,) with scalar part first, normalized.
    """
    is_torch = isinstance(R, torch.Tensor)
    
    if is_torch:
        R = R.cpu().numpy()
    
    assert R.shape == (3, 3), f"Expected 3x3 matrix, got {R.shape}"
    
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    q = np.array([qw, qx, qy, qz])
    q = q / np.linalg.norm(q)
    
    return torch.from_numpy(q).float() if is_torch else q


def quaternion_to_rotation_matrix_batch(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of quaternions to rotation matrices (vectorized, ~10x faster than loops).
    
    Args:
        quaternions: Shape [B, 4], each row is [qw, qx, qy, qz].
    
    Returns:
        Shape [B, 3, 3] rotation matrices.
    """
    batch_size = quaternions.shape[0]
    device = quaternions.device
    
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    
    qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    R = torch.zeros((batch_size, 3, 3), device=device, dtype=quaternions.dtype)
    
    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
    
    return R



def get_pose_transforms(train: bool = True, color_jitter: bool = None) -> T.Compose:
    """
    Get image transforms pipeline for pose estimation.
    
    Args:
        train: Include augmentation if True.
        color_jitter: If None, uses Config.POSE_COLOR_JITTER.
        
    Returns:
        Composed transforms.
    """
    if color_jitter is None:
        color_jitter = Config.POSE_COLOR_JITTER
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if train:
        transforms_list = []
        if color_jitter:
            transforms_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        transforms_list.extend([T.RandomHorizontalFlip(p=0.5), T.ToTensor(), normalize])
        return T.Compose(transforms_list)
    else:
        return T.Compose([T.ToTensor(), normalize])
