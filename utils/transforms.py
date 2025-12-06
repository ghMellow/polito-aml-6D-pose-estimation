"""
Transforms and Utilities for 6D Pose Estimation

This module provides transformation functions for:
- Rotation representations (matrix <-> quaternion)
- Image preprocessing (crop, resize, normalize)
- Data augmentation for pose estimation
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Tuple, Union


def rotation_matrix_to_quaternion(R: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: Rotation matrix (3, 3) as numpy array or torch tensor
        
    Returns:
        Quaternion [qw, qx, qy, qz] (4,) where qw is the scalar part
        
    Reference:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    is_torch = isinstance(R, torch.Tensor)
    
    if is_torch:
        R = R.cpu().numpy()
    
    # Ensure 3x3 matrix
    assert R.shape == (3, 3), f"Expected 3x3 rotation matrix, got {R.shape}"
    
    # Compute quaternion components
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
    
    # Normalize
    q = q / np.linalg.norm(q)
    
    # Convert back to torch if needed
    if is_torch:
        q = torch.from_numpy(q).float()
    
    return q


def quaternion_to_rotation_matrix(q: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [qw, qx, qy, qz] (4,) where qw is the scalar part
        
    Returns:
        Rotation matrix (3, 3)
        
    Reference:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    """
    is_torch = isinstance(q, torch.Tensor)
    
    if is_torch:
        device = q.device
        q = q.cpu().numpy()
    
    # Ensure normalized
    q = q / np.linalg.norm(q)
    
    qw, qx, qy, qz = q
    
    # Compute rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])
    
    # Convert back to torch if needed
    if is_torch:
        R = torch.from_numpy(R).float().to(device)
    
    return R


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize quaternion to unit length.
    
    Args:
        q: Quaternion tensor (..., 4)
        
    Returns:
        Normalized quaternion (..., 4)
    """
    return q / torch.norm(q, dim=-1, keepdim=True)


def crop_image_from_bbox(
    image: Union[np.ndarray, Image.Image],
    bbox: Union[np.ndarray, list],
    margin: float = 0.1,
    output_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Crop image using bounding box with margin and resize to output size.
    
    Args:
        image: Input image (H, W, 3) as numpy array or PIL Image
        bbox: Bounding box [x, y, width, height]
        margin: Margin percentage to add around bbox (default: 0.1 = 10%)
        output_size: Output image size (height, width)
        
    Returns:
        Cropped and resized PIL Image
    """
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    # Extract bbox coordinates
    x, y, w, h = bbox
    
    # Add margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, int(x - margin_w))
    y1 = max(0, int(y - margin_h))
    x2 = min(image.width, int(x + w + margin_w))
    y2 = min(image.height, int(y + h + margin_h))
    
    # Crop
    cropped = image.crop((x1, y1, x2, y2))
    
    # Resize to output size
    cropped = cropped.resize(output_size, Image.BILINEAR)
    
    return cropped


def get_pose_transforms(train: bool = True) -> T.Compose:
    """
    Get image transforms for pose estimation.
    
    Args:
        train: Whether to include training augmentations
        
    Returns:
        Composed transforms
    """
    # ImageNet normalization
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        # Training transforms with augmentation
        transforms = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize
        ])
    else:
        # Validation/test transforms (no augmentation)
        transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])
    
    return transforms


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (3, H, W) or (B, 3, H, W)
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    # Handle batch dimension
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


def project_3d_points(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray
) -> np.ndarray:
    """
    Project 3D points to 2D image plane using camera parameters.
    
    Args:
        points_3d: 3D points (N, 3)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        K: Camera intrinsic matrix (3, 3)
        
    Returns:
        2D projected points (N, 2)
    """
    # Transform points to camera frame
    points_cam = (R @ points_3d.T).T + t  # (N, 3)
    
    # Project to image plane
    points_2d_homo = (K @ points_cam.T).T  # (N, 3)
    
    # Normalize by depth
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    return points_2d


def get_3d_bbox_corners(size_x: float, size_y: float, size_z: float, 
                       min_x: float, min_y: float, min_z: float) -> np.ndarray:
    """
    Get 3D bounding box corner points.
    
    Args:
        size_x, size_y, size_z: Size of the bounding box
        min_x, min_y, min_z: Minimum coordinates
        
    Returns:
        8 corner points (8, 3)
    """
    corners = np.array([
        [min_x, min_y, min_z],
        [min_x + size_x, min_y, min_z],
        [min_x + size_x, min_y + size_y, min_z],
        [min_x, min_y + size_y, min_z],
        [min_x, min_y, min_z + size_z],
        [min_x + size_x, min_y, min_z + size_z],
        [min_x + size_x, min_y + size_y, min_z + size_z],
        [min_x, min_y + size_y, min_z + size_z]
    ])
    return corners
