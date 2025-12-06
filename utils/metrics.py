"""
Evaluation Metrics for 6D Pose Estimation

This module implements evaluation metrics for 6D pose estimation:
- ADD (Average Distance of Model Points)
- ADD-S (Symmetric version for symmetric objects)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


def load_3d_model(model_path: Union[str, Path]) -> np.ndarray:
    """
    Load 3D model points from PLY file.
    
    Args:
        model_path: Path to .ply file
        
    Returns:
        3D model points (N, 3)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Read PLY file
    points = []
    with open(model_path, 'r') as f:
        # Skip header
        in_header = True
        vertex_count = 0
        
        for line in f:
            line = line.strip()
            
            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_header = False
                continue
            
            # Parse vertex coordinates
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    continue
            
            if len(points) >= vertex_count and vertex_count > 0:
                break
    
    points = np.array(points, dtype=np.float32)
    
    if len(points) == 0:
        raise ValueError(f"No points loaded from {model_path}")
    
    return points


def load_models_info(info_path: Union[str, Path]) -> Dict:
    """
    Load models information from models_info.yml.
    
    Args:
        info_path: Path to models_info.yml
        
    Returns:
        Dictionary with object information
    """
    info_path = Path(info_path)
    
    if not info_path.exists():
        raise FileNotFoundError(f"Models info file not found: {info_path}")
    
    with open(info_path, 'r') as f:
        models_info = yaml.safe_load(f)
    
    return models_info


def compute_add(
    pred_R: np.ndarray,
    pred_t: np.ndarray,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    model_points: np.ndarray,
    diameter: float,
    threshold: float = 0.1,
    symmetric: bool = False
) -> Dict[str, float]:
    """
    Compute ADD (Average Distance of Model Points) metric.
    
    ADD = mean( || (R_pred * p + t_pred) - (R_gt * p + t_gt) || )
    
    For symmetric objects, uses ADD-S which finds minimum distance to any point.
    
    Args:
        pred_R: Predicted rotation matrix (3, 3)
        pred_t: Predicted translation vector (3,)
        gt_R: Ground truth rotation matrix (3, 3)
        gt_t: Ground truth translation vector (3,)
        model_points: 3D model points (N, 3)
        diameter: Object diameter (for threshold calculation)
        threshold: Threshold as fraction of diameter (default: 0.1 = 10%)
        symmetric: Whether to use ADD-S for symmetric objects
        
    Returns:
        Dictionary with 'add', 'add_threshold', 'is_correct' keys
    """
    # Transform model points with predicted pose
    pred_points = (pred_R @ model_points.T).T + pred_t  # (N, 3)
    
    # Transform model points with ground truth pose
    gt_points = (gt_R @ model_points.T).T + gt_t  # (N, 3)
    
    if symmetric:
        # ADD-S: For each predicted point, find closest GT point
        # This handles symmetry ambiguity
        from scipy.spatial.distance import cdist
        distances = cdist(pred_points, gt_points)  # (N, N)
        min_distances = np.min(distances, axis=1)  # (N,)
        add_value = np.mean(min_distances)
    else:
        # Standard ADD: Point-to-point distance
        distances = np.linalg.norm(pred_points - gt_points, axis=1)  # (N,)
        add_value = np.mean(distances)
    
    # Compute threshold
    add_threshold = diameter * threshold
    
    # Check if prediction is correct
    is_correct = add_value < add_threshold
    
    return {
        'add': float(add_value),
        'add_threshold': float(add_threshold),
        'is_correct': bool(is_correct)
    }


def compute_add_batch(
    pred_R_batch: Union[np.ndarray, torch.Tensor],
    pred_t_batch: Union[np.ndarray, torch.Tensor],
    gt_R_batch: Union[np.ndarray, torch.Tensor],
    gt_t_batch: Union[np.ndarray, torch.Tensor],
    obj_ids: List[int],
    models_dict: Dict[int, np.ndarray],
    models_info: Dict[int, Dict],
    symmetric_objects: List[int] = [8, 9],
    threshold: float = 0.1
) -> Dict[str, List[float]]:
    """
    Compute ADD metric for a batch of predictions.
    
    Args:
        pred_R_batch: Predicted rotation matrices (B, 3, 3)
        pred_t_batch: Predicted translations (B, 3)
        gt_R_batch: Ground truth rotation matrices (B, 3, 3)
        gt_t_batch: Ground truth translations (B, 3)
        obj_ids: Object IDs for each sample (B,)
        models_dict: Dictionary mapping obj_id to 3D model points
        models_info: Dictionary with object information (diameter, etc.)
        symmetric_objects: List of symmetric object IDs
        threshold: Threshold as fraction of diameter
        
    Returns:
        Dictionary with lists of ADD values and correctness
    """
    # Convert to numpy if torch tensors
    if isinstance(pred_R_batch, torch.Tensor):
        pred_R_batch = pred_R_batch.cpu().numpy()
    if isinstance(pred_t_batch, torch.Tensor):
        pred_t_batch = pred_t_batch.cpu().numpy()
    if isinstance(gt_R_batch, torch.Tensor):
        gt_R_batch = gt_R_batch.cpu().numpy()
    if isinstance(gt_t_batch, torch.Tensor):
        gt_t_batch = gt_t_batch.cpu().numpy()
    
    batch_size = len(obj_ids)
    
    add_values = []
    add_thresholds = []
    is_correct_list = []
    
    for i in range(batch_size):
        obj_id = obj_ids[i]
        
        # Get model points and diameter
        model_points = models_dict[obj_id]
        diameter = models_info[obj_id]['diameter']
        
        # Check if symmetric
        is_symmetric = obj_id in symmetric_objects
        
        # Compute ADD
        result = compute_add(
            pred_R_batch[i],
            pred_t_batch[i],
            gt_R_batch[i],
            gt_t_batch[i],
            model_points,
            diameter,
            threshold=threshold,
            symmetric=is_symmetric
        )
        
        add_values.append(result['add'])
        add_thresholds.append(result['add_threshold'])
        is_correct_list.append(result['is_correct'])
    
    return {
        'add_values': add_values,
        'add_thresholds': add_thresholds,
        'is_correct': is_correct_list,
        'mean_add': float(np.mean(add_values)),
        'accuracy': float(np.mean(is_correct_list))
    }


def load_all_models(
    models_dir: Union[str, Path],
    obj_ids: Optional[List[int]] = None
) -> Dict[int, np.ndarray]:
    """
    Load all 3D models from models directory.
    
    Args:
        models_dir: Path to models directory
        obj_ids: List of object IDs to load (default: all 1-15)
        
    Returns:
        Dictionary mapping obj_id to 3D points
    """
    models_dir = Path(models_dir)
    
    if obj_ids is None:
        obj_ids = list(range(1, 16))  # 1-15
    
    models_dict = {}
    
    for obj_id in obj_ids:
        model_path = models_dir / f"obj_{obj_id:02d}.ply"
        
        if model_path.exists():
            try:
                points = load_3d_model(model_path)
                models_dict[obj_id] = points
                print(f"✅ Loaded model {obj_id:02d}: {len(points)} points")
            except Exception as e:
                print(f"❌ Failed to load model {obj_id:02d}: {e}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
    
    return models_dict


if __name__ == '__main__':
    # Test metrics
    print("Testing ADD metric computation...\n")
    
    # Create dummy data
    model_points = np.random.randn(1000, 3) * 50  # Random 3D points
    
    # Identity pose (should give ADD ≈ 0)
    R_gt = np.eye(3)
    t_gt = np.array([100.0, 200.0, 1000.0])
    
    # Small perturbation
    R_pred = R_gt + np.random.randn(3, 3) * 0.01
    t_pred = t_gt + np.random.randn(3) * 5.0
    
    diameter = 200.0
    
    # Compute ADD
    result = compute_add(R_pred, t_pred, R_gt, t_gt, model_points, diameter)
    
    print(f"📊 ADD Metric:")
    print(f"   ADD value: {result['add']:.2f} mm")
    print(f"   Threshold (10% of diameter): {result['add_threshold']:.2f} mm")
    print(f"   Correct: {result['is_correct']}")
    
    print(f"\n✅ Metrics tests completed")
