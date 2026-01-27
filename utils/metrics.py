"""
ADD / ADD-S metrics for 6D pose estimation models.
Metriche per la 6D Pose Estimation (solo per modelli di pose, non per YOLO)

Questo modulo implementa le metriche ADD/ADD-S per valutare modelli di pose (es. ResNet):

1. ResNet che predice solo la rotazione:
    - Si valuta la rotazione predetta, la traslazione è fissata a zero (dummy).
    - Si usa la metrica ADD considerando solo la rotazione.

2. ResNet che predice rotazione e traslazione:
    - Si valuta sia la rotazione che la traslazione predetta.
    - Si usa la metrica ADD completa.

3. Versioni avanzate/future:
    - Supporto per modelli che usano anche depth, mask, o strategie avanzate.
    - Funzioni batch ottimizzate e GPU-ready.

Nota: Le metriche ADD/ADD-S non sono usate per la valutazione di modelli YOLO, che si occupano solo di object detection 2D e forniscono i crop per la rete di pose.
"""

from typing import Dict, List, Union
import numpy as np
import torch
import yaml
from typing import Dict, List, Optional, Union
from pathlib import Path

from config import Config


def process_obj_for_add(args):
    obj_id, obj_ids, pred_R_batch, pred_t_batch, gt_R_batch, gt_t_batch, models_dict, models_info, symmetric_objects, threshold = args
    import numpy as np
    indices = [i for i, oid in enumerate(obj_ids) if oid == obj_id]
    if not indices:
        return None
    model_points = models_dict[obj_id]
    diameter = models_info[obj_id]['diameter']
    is_symmetric = obj_id in symmetric_objects

    pred_R = pred_R_batch[indices]
    pred_t = pred_t_batch[indices]
    gt_R = gt_R_batch[indices]
    gt_t = gt_t_batch[indices]

    pred_points = np.einsum('kij,nj->kni', pred_R, model_points) + pred_t[:, None, :]
    gt_points = np.einsum('kij,nj->kni', gt_R, model_points) + gt_t[:, None, :]

    local_add_values = np.zeros(len(indices))
    if is_symmetric:
        try:
            from scipy.spatial.distance import cdist
            for local_idx in range(len(indices)):
                distances = cdist(pred_points[local_idx], gt_points[local_idx])
                min_distances = np.min(distances, axis=1)
                local_add_values[local_idx] = np.mean(min_distances)
        except ImportError:
            for local_idx in range(len(indices)):
                diff = pred_points[local_idx][:, None, :] - gt_points[local_idx][None, :, :]
                distances = np.linalg.norm(diff, axis=2)
                min_distances = np.min(distances, axis=1)
                local_add_values[local_idx] = np.mean(min_distances)
    else:
        distances = np.linalg.norm(pred_points - gt_points, axis=2)
        add_batch = np.mean(distances, axis=1)
        local_add_values = add_batch

    threshold_value = diameter * threshold
    local_add_thresholds = np.full(len(indices), threshold_value)
    local_is_correct = local_add_values < threshold_value

    return indices, local_add_values, local_add_thresholds, local_is_correct

def _to_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array


def _to_tensor(array: Union[np.ndarray, torch.Tensor], device: str) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device)
    return torch.as_tensor(array, device=device, dtype=torch.float32)


# ==================== 1. ResNet: Solo Rotazione ====================

def compute_add_batch_rotation_only(
    pred_R_batch: Union[np.ndarray, torch.Tensor],
    gt_R_batch: Union[np.ndarray, torch.Tensor],
    obj_ids: List[int],
    models_dict: Dict[int, np.ndarray],
    models_info: Dict[int, Dict],
    symmetric_objects: List[int] = None,
    threshold: float = None
) -> Dict[str, List[float]]:
    """Compute ADD for rotation-only predictions using zero translation."""
    batch_size = pred_R_batch.shape[0]
    if Config.GPU_PRESENT:
        device = Config.DEVICE
        pred_R_batch = _to_tensor(pred_R_batch, device)
        gt_R_batch = _to_tensor(gt_R_batch, device)
        pred_t_batch = torch.zeros((batch_size, 3), device=device)
        gt_t_batch = torch.zeros((batch_size, 3), device=device)
        return compute_add_batch_gpu(pred_R_batch, pred_t_batch, gt_R_batch, gt_t_batch,
                                     obj_ids, models_dict, models_info, symmetric_objects, threshold, device=device)

    pred_t_batch = np.zeros((batch_size, 3))
    gt_t_batch = np.zeros((batch_size, 3))
    return compute_add_batch(pred_R_batch, pred_t_batch, gt_R_batch, gt_t_batch,
                             obj_ids, models_dict, models_info, symmetric_objects, threshold)


# ==================== 2. ResNet: Rotazione + Traslazione ====================

def compute_add_batch_full_pose(
    pred_R_batch: Union[np.ndarray, torch.Tensor],
    pred_t_batch: Union[np.ndarray, torch.Tensor],
    gt_R_batch: Union[np.ndarray, torch.Tensor],
    gt_t_batch: Union[np.ndarray, torch.Tensor],
    obj_ids: List[int],
    models_dict: Dict[int, np.ndarray],
    models_info: Dict[int, Dict],
    symmetric_objects: List[int] = None,
    threshold: float = None
) -> Dict[str, List[float]]:
    """Compute ADD for full 6D pose predictions."""
    if Config.GPU_PRESENT:
        device = Config.DEVICE
        pred_R_batch = _to_tensor(pred_R_batch, device)
        pred_t_batch = _to_tensor(pred_t_batch, device)
        gt_R_batch = _to_tensor(gt_R_batch, device)
        gt_t_batch = _to_tensor(gt_t_batch, device)
        return compute_add_batch_gpu(pred_R_batch, pred_t_batch, gt_R_batch, gt_t_batch,
                                     obj_ids, models_dict, models_info, symmetric_objects, threshold, device=device)

    return compute_add_batch(pred_R_batch, pred_t_batch, gt_R_batch, gt_t_batch,
                             obj_ids, models_dict, models_info, symmetric_objects, threshold)

# ==================== 3. Funzioni Generali ====================

def compute_add_batch(
    pred_R_batch: Union[np.ndarray, torch.Tensor],
    pred_t_batch: Union[np.ndarray, torch.Tensor],
    gt_R_batch: Union[np.ndarray, torch.Tensor],
    gt_t_batch: Union[np.ndarray, torch.Tensor],
    obj_ids: List[int],
    models_dict: Dict[int, np.ndarray],
    models_info: Dict[int, Dict],
    symmetric_objects: List[int] = None,
    threshold: float = None
) -> Dict[str, List[float]]:
    """Vectorized ADD / ADD-S for batches of 6D poses."""
    if symmetric_objects is None:
        symmetric_objects = Config.SYMMETRIC_OBJECTS
    if threshold is None:
        threshold = Config.ADD_THRESHOLD
    pred_R_batch = _to_numpy(pred_R_batch)
    pred_t_batch = _to_numpy(pred_t_batch)
    gt_R_batch = _to_numpy(gt_R_batch)
    gt_t_batch = _to_numpy(gt_t_batch)

    if pred_t_batch.ndim == 3 and pred_t_batch.shape[1] == 1:
        pred_t_batch = pred_t_batch.squeeze(1)
    if gt_t_batch.ndim == 3 and gt_t_batch.shape[1] == 1:
        gt_t_batch = gt_t_batch.squeeze(1)


    batch_size = len(obj_ids)
    unique_obj_ids = set(obj_ids)
    add_values = np.zeros(batch_size)
    add_thresholds = np.zeros(batch_size)
    is_correct_array = np.zeros(batch_size, dtype=bool)


    import concurrent.futures
    n_jobs = Config.PARALLEL_JOBS
    if n_jobs > 1 and len(unique_obj_ids) > 1:
        args_list = [
            (obj_id, obj_ids, pred_R_batch, pred_t_batch, gt_R_batch, gt_t_batch, models_dict, models_info, symmetric_objects, threshold)
            for obj_id in unique_obj_ids
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_obj_for_add, args_list))
        for res in results:
            if res is None:
                continue
            indices, local_add_values, local_add_thresholds, local_is_correct = res
            add_values[indices] = local_add_values
            add_thresholds[indices] = local_add_thresholds
            is_correct_array[indices] = local_is_correct
    else:
        for obj_id in unique_obj_ids:
            res = process_obj_for_add((obj_id, obj_ids, pred_R_batch, pred_t_batch, gt_R_batch, gt_t_batch, models_dict, models_info, symmetric_objects, threshold))
            if res is None:
                continue
            indices, local_add_values, local_add_thresholds, local_is_correct = res
            add_values[indices] = local_add_values
            add_thresholds[indices] = local_add_thresholds
            is_correct_array[indices] = local_is_correct

    return {
        'add_values': add_values.tolist(),
        'add_thresholds': add_thresholds.tolist(),
        'is_correct': is_correct_array.tolist(),
        'mean_add': float(np.mean(add_values)),
        'accuracy': float(np.mean(is_correct_array))
    }


def compute_add_batch_gpu(
    pred_R_batch: torch.Tensor,
    pred_t_batch: torch.Tensor,
    gt_R_batch: torch.Tensor,
    gt_t_batch: torch.Tensor,
    obj_ids: Union[List[int], np.ndarray],
    models_dict: Dict[int, np.ndarray],
    models_info: Dict[int, Dict],
    symmetric_objects: List[int] = None,
    threshold: float = None,
    device: str = None
) -> Dict[str, torch.Tensor]:
    """GPU implementation of ADD / ADD-S that keeps computation on device."""
    if device is None:
        device = Config.DEVICE
    if symmetric_objects is None:
        symmetric_objects = Config.SYMMETRIC_OBJECTS
    if threshold is None:
        threshold = Config.ADD_THRESHOLD

    # Make sure all tensors are on the same device as indices that will be created below
    pred_R_batch = _to_numpy(pred_R_batch)
    pred_t_batch = _to_numpy(pred_t_batch)
    gt_R_batch = _to_numpy(gt_R_batch)
    gt_t_batch = _to_numpy(gt_t_batch)

    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.cpu().numpy()

    batch_size = len(obj_ids)
    add_values = torch.zeros(batch_size, device=device)
    add_thresholds = torch.zeros(batch_size, device=device)
    is_correct_array = torch.zeros(batch_size, dtype=torch.bool, device=device)

    unique_obj_ids = np.unique(obj_ids)

    for obj_id in unique_obj_ids:
        mask = obj_ids == obj_id
        indices_np = np.where(mask)[0]
        indices = torch.from_numpy(indices_np).to(device)
        if len(indices) == 0:
            continue

        model_points_np = models_dict[obj_id]
        model_points = torch.from_numpy(model_points_np).float().to(device)

        diameter = models_info[obj_id]['diameter']
        is_symmetric = obj_id in symmetric_objects

        pred_R = pred_R_batch[indices_np]
        pred_t = pred_t_batch[indices_np]
        gt_R = gt_R_batch[indices_np]
        gt_t = gt_t_batch[indices_np]

        pred_points = torch.einsum('kij,nj->kni', pred_R, model_points) + pred_t.unsqueeze(1)
        gt_points = torch.einsum('kij,nj->kni', gt_R, model_points) + gt_t.unsqueeze(1)

        if is_symmetric:
            for i, idx in enumerate(indices_np):
                diff = pred_points[i].unsqueeze(1) - gt_points[i].unsqueeze(0)
                distances = torch.norm(diff, dim=2)
                min_distances = torch.min(distances, dim=1)[0]
                add_values[idx] = torch.mean(min_distances)
        else:
            distances = torch.norm(pred_points - gt_points, dim=2)
            add_batch = torch.mean(distances, dim=1)
            add_values[indices_np] = add_batch

        threshold_value = diameter * threshold
        add_thresholds[indices_np] = threshold_value
        is_correct_array[indices_np] = add_values[indices_np] < threshold_value

    return {
        'add_values': add_values.cpu().numpy(),
        'add_thresholds': add_thresholds.cpu().numpy(),
        'is_correct': is_correct_array.cpu().numpy(),
        'mean_add': float(add_values.mean().item()),
        'accuracy': float(is_correct_array.float().mean().item())
    }