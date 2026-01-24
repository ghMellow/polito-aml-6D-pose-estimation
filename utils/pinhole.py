"""
Pinhole Camera Model for 3D translation computation from 2D bounding boxes
and depth maps using camera intrinsics.

Formula:
    X = Z * (u - cx) / fx
    Y = Z * (v - cy) / fy
    Z = median(depth_map[bbox])
"""

import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Union


def load_camera_intrinsics(gt_yml_path: Union[str, Path]) -> Dict[str, float]:
    """
    Load camera intrinsics from LineMOD info.yml file.
    
    Camera intrinsics are stored in info.yml (not gt.yml) in the same directory
    as the gt file, with cam_K as a flattened 3x3 matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    """
    gt_yml_path = Path(gt_yml_path)
    info_yml_path = gt_yml_path.parent / 'info.yml'
    
    if not info_yml_path.exists():
        raise FileNotFoundError(f"Info file not found: {info_yml_path}")
    
    with open(info_yml_path, 'r') as f:
        info_data = yaml.safe_load(f)
    
    first_img_id = list(info_data.keys())[0]
    first_info = info_data[first_img_id]
    
    if 'cam_K' not in first_info:
        raise KeyError(f"'cam_K' not found in {info_yml_path}")
    
    cam_K = first_info['cam_K']
    
    return {
        'fx': cam_K[0],
        'fy': cam_K[4],
        'cx': cam_K[2],
        'cy': cam_K[5]
    }


def compute_translation_pinhole(
    bbox: Union[list, tuple, np.ndarray],
    depth_path: Union[str, Path],
    camera_intrinsics: Dict[str, float],
    depth_scale: float = 1.0,
    use_median: bool = True
) -> np.ndarray:
    """
    Compute 3D translation using Pinhole Camera Model.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        depth_path: Path to depth image (PNG 16-bit)
        camera_intrinsics: Dict with keys {fx, fy, cx, cy}
        depth_scale: Scale factor for depth values (default: 1.0 for mm)
        use_median: Use median (True) or mean (False) for depth aggregation
    
    Returns:
        Translation array [tx, ty, tz] in depth units (typically millimeters)
    
    Raises:
        ValueError: If bbox is invalid or no valid depth values found
        FileNotFoundError: If depth image does not exist
    """
    bbox = np.array(bbox, dtype=np.float32)
    if bbox.shape[0] != 4:
        raise ValueError(f"Bbox must have 4 elements, got: {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    
    depth_path = Path(depth_path)
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    
    depth_map = np.array(Image.open(depth_path))
    
    x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
    x2_int, y2_int = min(depth_map.shape[1], int(x2)), min(depth_map.shape[0], int(y2))
    
    if x1_int >= x2_int or y1_int >= y2_int:
        raise ValueError(f"Invalid bbox after clipping: [{x1_int}, {y1_int}, {x2_int}, {y2_int}]")
    
    depth_roi = depth_map[y1_int:y2_int, x1_int:x2_int]
    valid_depth = depth_roi[depth_roi > 0]
    
    if len(valid_depth) == 0:
        raise ValueError(f"No valid depth values in bbox {bbox}")
    
    Z = (np.median(valid_depth) if use_median else np.mean(valid_depth)) * depth_scale
    
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    
    X = Z * (u - cx) / fx
    Y = Z * (v - cy) / fy
    
    return np.array([X, Y, Z], dtype=np.float32)


def compute_translation_pinhole_batch(
    bboxes: np.ndarray,
    depth_paths: list,
    camera_intrinsics: Dict[str, float],
    depth_scale: float = 1.0,
    use_median: bool = True
) -> np.ndarray:
    """
    Batch version of compute_translation_pinhole for multiple images.
    
    Args:
        bboxes: Array of shape [N, 4] with N bounding boxes
        depth_paths: List of N paths to depth images
        camera_intrinsics: Dict with keys {fx, fy, cx, cy}
        depth_scale: Scale factor for depth values
        use_median: Use median (True) or mean (False) for depth aggregation
    
    Returns:
        Array of shape [N, 3] with N translations
    
    Raises:
        ValueError: If bboxes and depth_paths have different lengths
    """
    if len(bboxes) != len(depth_paths):
        raise ValueError(
            f"Number of bboxes ({len(bboxes)}) != "
            f"number of depth paths ({len(depth_paths)})"
        )
    
    translations = []
    for bbox, depth_path in zip(bboxes, depth_paths):
        try:
            translation = compute_translation_pinhole(
                bbox, depth_path, camera_intrinsics, depth_scale, use_median
            )
            translations.append(translation)
        except (FileNotFoundError, ValueError):
            translations.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    
    return np.array(translations)


def verify_pinhole_computation(
    bbox: Union[list, tuple],
    depth_path: Union[str, Path],
    camera_intrinsics: Dict[str, float],
    gt_translation: Optional[np.ndarray] = None
) -> Dict:
    """
    Verify pinhole computation and optionally compare with ground truth.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        depth_path: Path to depth image
        camera_intrinsics: Dict with keys {fx, fy, cx, cy}
        gt_translation: Ground truth translation for comparison (optional)
    
    Returns:
        Dict containing: translation, center_pixel, depth_z, bbox, intrinsics,
        and optionally gt_translation and error metrics if gt_translation provided
    """
    translation = compute_translation_pinhole(bbox, depth_path, camera_intrinsics)
    
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    
    result = {
        'translation': translation,
        'center_pixel': (u, v),
        'depth_z': translation[2],
        'bbox': bbox,
        'intrinsics': camera_intrinsics
    }
    
    if gt_translation is not None:
        error = np.linalg.norm(translation - gt_translation)
        result['gt_translation'] = gt_translation
        result['error_mm'] = error
        result['error_percent'] = (error / np.linalg.norm(gt_translation)) * 100
    
    return result