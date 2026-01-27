"""File I/O utilities for 6D Pose Estimation."""

import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union

from config import Config


def load_models_info(info_path: Union[str, Path]) -> Dict:
    """Load models information from models_info.yml."""
    info_path = Path(info_path)
    
    if not info_path.exists():
        raise FileNotFoundError(f"Models info file not found: {info_path}")
    
    with open(info_path, 'r') as f:
        models_info = yaml.safe_load(f)
    
    return models_info


def load_3d_model(model_path: Union[str, Path]) -> np.ndarray:
    """Load 3D model points from PLY file."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    points = []
    in_header = True
    vertex_count = 0
    
    with open(model_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_header = False
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    continue
            
            if vertex_count > 0 and len(points) >= vertex_count:
                break
    
    points = np.array(points, dtype=np.float32)
    
    if len(points) == 0:
        raise ValueError(f"No points loaded from {model_path}")
    
    return points


def load_all_models(
    models_dir: Union[str, Path] = None,
    obj_ids: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict[int, np.ndarray]:
    """Load all 3D models from models directory."""
    if models_dir is None:
        models_dir = Config.MODELS_PATH
    models_dir = Path(models_dir)
    
    if obj_ids is None:
        obj_ids = list(Config.LINEMOD_OBJECTS.keys())
    
    models_dict = {}
    
    if verbose:
        print("[load_all_models] VERBOSE - Loading data")
    
    for obj_id in obj_ids:
        model_path = models_dir / f"obj_{obj_id:02d}.ply"
        
        if model_path.exists():
            try:
                points = load_3d_model(model_path)
                models_dict[obj_id] = points
                if verbose:
                    print(f"[load_all_models] Loaded model {obj_id:02d}: {len(points)} points")
            except Exception as e:
                if verbose:
                    print(f"[load_all_models] Failed to load model {obj_id:02d}: {e}")
        else:
            if verbose:
                print(f"[load_all_models] Model file not found: {model_path}")
    
    return models_dict


def save_validation_results(results_rot_only, results_pinhole, checkpoint_dir, verbose: bool = True):
    """Save validation results to CSV file."""
    validation_results = []
    
    # Extract rotation-only results
    add_values = results_rot_only.get('add_values')
    is_correct = results_rot_only.get('is_correct')
    add_values_6d = results_rot_only.get('add_values_6d')
    is_correct_6d = results_rot_only.get('is_correct_6d')
    obj_ids_rot = results_rot_only.get('obj_ids')
    
    if add_values is not None and is_correct is not None and obj_ids_rot is not None:
        for i in range(len(add_values)):
            row = {
                'obj_id': obj_ids_rot[i],
                'add_value': add_values[i],
                'is_correct': is_correct[i]
            }
            if add_values_6d is not None and i < len(add_values_6d):
                row['add_value_6d'] = add_values_6d[i]
            if is_correct_6d is not None and i < len(is_correct_6d):
                row['is_correct_6d'] = is_correct_6d[i]
            validation_results.append(row)
    
    # Add pinhole results if available
    if results_pinhole is not None:
        pinhole_errors = results_pinhole.get('pinhole_errors')
        pred_translations = results_pinhole.get('pred_translations')
        gt_translations = results_pinhole.get('gt_translations')
        obj_ids_pinhole = results_pinhole.get('obj_ids')
        
        if pinhole_errors is not None and obj_ids_pinhole is not None:
            for i in range(len(pinhole_errors)):
                if i < len(validation_results):
                    validation_results[i]['pinhole_error'] = pinhole_errors[i]
                    if pred_translations is not None:
                        validation_results[i]['pred_translation'] = pred_translations[i]
                    if gt_translations is not None:
                        validation_results[i]['gt_translation'] = gt_translations[i]
                else:
                    validation_results.append({
                        'obj_id': obj_ids_pinhole[i],
                        'pinhole_error': pinhole_errors[i],
                        'pred_translation': pred_translations[i] if pred_translations is not None else None,
                        'gt_translation': gt_translations[i] if gt_translations is not None else None
                    })
    
    if validation_results:
        df_val = pd.DataFrame(validation_results)
        checkpoint_dir = Path(checkpoint_dir)
        val_csv_path = checkpoint_dir / 'validation_result.csv'
        df_val.to_csv(val_csv_path, index=False)
        if verbose:
            print(f"Risultati di validazione salvati in {val_csv_path}")


def load_validation_results(val_csv_path):
    """Load validation results from CSV file."""
    df_val = pd.read_csv(val_csv_path)
    
    results_rot_only = {
        'obj_ids': df_val['obj_id'].values,
        'add_values': df_val['add_value'].values if 'add_value' in df_val.columns else None,
        'is_correct': df_val['is_correct'].values if 'is_correct' in df_val.columns else None,
        'add_values_6d': df_val['add_value_6d'].values if 'add_value_6d' in df_val.columns else None,
        'is_correct_6d': df_val['is_correct_6d'].values if 'is_correct_6d' in df_val.columns else None
    }
    
    results_pinhole = None
    if 'pinhole_error' in df_val.columns:
        results_pinhole = {
            'obj_ids': df_val['obj_id'].values,
            'pinhole_errors': df_val['pinhole_error'].values,
            'pred_translations': df_val['pred_translation'].values if 'pred_translation' in df_val.columns else None,
            'gt_translations': df_val['gt_translation'].values if 'gt_translation' in df_val.columns else None
        }
    
    return results_rot_only, results_pinhole
