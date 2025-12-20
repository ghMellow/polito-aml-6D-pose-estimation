"""
Pinhole Camera Model for Translation Computation

This module implements the Pinhole Camera Model to compute 3D translation
from 2D bounding box coordinates, depth map, and camera intrinsics.

Formula:
    X = Z * (u - cx) / fx
    Y = Z * (v - cy) / fy
    Z = median(depth_map[bbox])

Where:
    - (u, v): Centro del bounding box in pixel
    - (fx, fy, cx, cy): Parametri intrinseci della camera
    - Z: Profondità dalla depth map
"""

import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, Optional, Union


def load_camera_intrinsics(gt_yml_path: Union[str, Path]) -> Dict[str, float]:
    """
    Carica parametri intrinseci della camera dal file info.yml.
    
    LineMOD format in info.yml:
        cam_K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]  # 3x3 matrix flattened
    
    Args:
        gt_yml_path: Path al file gt.yml (es. data/01/gt.yml)
                    La funzione automaticamente carica info.yml nella stessa directory
    
    Returns:
        Dictionary con chiavi: {fx, fy, cx, cy}
    
    Example:
        >>> intrinsics = load_camera_intrinsics('data/01/gt.yml')
        >>> print(intrinsics)
        {'fx': 572.41, 'fy': 573.57, 'cx': 325.26, 'cy': 242.04}
    """
    gt_yml_path = Path(gt_yml_path)
    
    # Camera intrinsics sono in info.yml, non in gt.yml
    info_yml_path = gt_yml_path.parent / 'info.yml'
    
    if not info_yml_path.exists():
        raise FileNotFoundError(f"Info file not found: {info_yml_path}")
    
    with open(info_yml_path, 'r') as f:
        info_data = yaml.safe_load(f)
    
    # Prendi la prima immagine per estrarre cam_K
    # In LineMOD, cam_K è uguale per tutte le immagini dello stesso oggetto
    first_img_id = list(info_data.keys())[0]
    first_info = info_data[first_img_id]
    
    if 'cam_K' not in first_info:
        raise KeyError(f"'cam_K' not found in {info_yml_path}")
    
    # cam_K è una lista di 9 elementi: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    cam_K = first_info['cam_K']
    
    intrinsics = {
        'fx': cam_K[0],  # focal length x
        'fy': cam_K[4],  # focal length y
        'cx': cam_K[2],  # principal point x
        'cy': cam_K[5]   # principal point y
    }
    
    return intrinsics


def compute_translation_pinhole(
    bbox: Union[list, tuple, np.ndarray],
    depth_path: Union[str, Path],
    camera_intrinsics: Dict[str, float],
    depth_scale: float = 1.0,
    use_median: bool = True
) -> np.ndarray:
    """
    Calcola translation 3D usando Pinhole Camera Model.
    
    Formula:
        X = Z * (u - cx) / fx
        Y = Z * (v - cy) / fy
        Z = mediana o media della depth map nel bbox
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixel
        depth_path: Path alla depth image (PNG 16-bit)
        camera_intrinsics: Dict con {fx, fy, cx, cy}
        depth_scale: Fattore di scala per depth (default: 1.0 per millimetri)
        use_median: Se True usa mediana, altrimenti media (default: True per robustezza)
    
    Returns:
        translation: Array [tx, ty, tz] in millimetri (o unità della depth map)
    
    Example:
        >>> bbox = [100, 150, 200, 250]
        >>> depth_path = 'data/01/depth/0000.png'
        >>> intrinsics = {'fx': 572.41, 'fy': 573.57, 'cx': 325.26, 'cy': 242.04}
        >>> translation = compute_translation_pinhole(bbox, depth_path, intrinsics)
        >>> print(translation)  # [tx, ty, tz] in mm
        [45.2, -120.5, 850.3]
    """

    # Gestione formato bbox: accetta sia [x1, y1, x2, y2] sia [x, y, w, h]
    bbox = np.array(bbox, dtype=np.float32)
    if bbox.shape[0] != 4:
        raise ValueError(f"Bbox deve avere 4 elementi, ricevuto: {bbox}")
    # Se bbox è [x, y, w, h] (w, h > 0 e x2 < x1 + w + 2), converti in [x1, y1, x2, y2]
    x, y, w, h = bbox
    if w > 0 and h > 0 and (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
        # Probabile [x, y, w, h]
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        # Già [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
    else:
        # Fallback: se w/h sono troppo piccoli o negativi, assume [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

    # ✅ Centro del bounding box (coordinate pixel)
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    
    # ✅ Carica depth map
    depth_path = Path(depth_path)
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    
    # LineMOD depth è uint16, valori in millimetri
    depth_map = np.array(Image.open(depth_path))
    
    # ✅ Estrai region of interest dalla depth map
    x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
    x2_int, y2_int = min(depth_map.shape[1], int(x2)), min(depth_map.shape[0], int(y2))
    
    if x1_int >= x2_int or y1_int >= y2_int:
        raise ValueError(f"Invalid bbox after clipping: [{x1_int}, {y1_int}, {x2_int}, {y2_int}]")
    
    depth_roi = depth_map[y1_int:y2_int, x1_int:x2_int]
    
    # ✅ Filtra valori invalidi (depth = 0 o troppo grande)
    valid_depth = depth_roi[depth_roi > 0]
    
    if len(valid_depth) == 0:
        raise ValueError(f"No valid depth values in bbox {bbox}")
    
    # ✅ Calcola Z (profondità) - mediana per robustezza a outliers
    if use_median:
        Z = np.median(valid_depth) * depth_scale
    else:
        Z = np.mean(valid_depth) * depth_scale
    
    # ✅ Estrai parametri intrinseci
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    
    # ✅ Pinhole Camera Model
    X = Z * (u - cx) / fx
    Y = Z * (v - cy) / fy
    
    # Ritorna translation come array numpy
    translation = np.array([X, Y, Z], dtype=np.float32)
    
    return translation


def compute_translation_pinhole_batch(
    bboxes: np.ndarray,
    depth_paths: list,
    camera_intrinsics: Dict[str, float],
    depth_scale: float = 1.0,
    use_median: bool = True
) -> np.ndarray:
    """
    Versione batch di compute_translation_pinhole per processare multiple immagini.
    
    Args:
        bboxes: Array [N, 4] con N bounding boxes
        depth_paths: Lista di N path alle depth maps
        camera_intrinsics: Dict con {fx, fy, cx, cy}
        depth_scale: Fattore di scala per depth
        use_median: Se True usa mediana, altrimenti media
    
    Returns:
        translations: Array [N, 3] con N translations
    
    Example:
        >>> bboxes = np.array([[100, 150, 200, 250], [300, 400, 400, 500]])
        >>> depth_paths = ['data/01/depth/0000.png', 'data/01/depth/0001.png']
        >>> intrinsics = {'fx': 572.41, 'fy': 573.57, 'cx': 325.26, 'cy': 242.04}
        >>> translations = compute_translation_pinhole_batch(bboxes, depth_paths, intrinsics)
        >>> print(translations.shape)  # (2, 3)
    """
    if len(bboxes) != len(depth_paths):
        raise ValueError(f"bboxes length ({len(bboxes)}) != depth_paths length ({len(depth_paths)})")
    
    translations = []
    
    for bbox, depth_path in zip(bboxes, depth_paths):
        try:
            translation = compute_translation_pinhole(
                bbox, depth_path, camera_intrinsics, depth_scale, use_median
            )
            translations.append(translation)
        except (FileNotFoundError, ValueError) as e:
            # Se fallisce, usa translation nulla (gestisci errore)
            print(f"⚠️  Warning: Failed to compute translation for {depth_path}: {e}")
            translations.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    
    return np.array(translations)


def verify_pinhole_computation(
    bbox: Union[list, tuple],
    depth_path: Union[str, Path],
    camera_intrinsics: Dict[str, float],
    gt_translation: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict:
    """
    Utility per verificare il calcolo pinhole e confrontare con ground truth.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        depth_path: Path alla depth image
        camera_intrinsics: Dict con {fx, fy, cx, cy}
        gt_translation: Ground truth translation per confronto (opzionale)
        verbose: Se True stampa dettagli
    
    Returns:
        Dict con risultati: {translation, error, center_pixel, depth_z, ...}
    """
    # Calcola translation
    translation = compute_translation_pinhole(bbox, depth_path, camera_intrinsics)
    
    # Calcola centro bbox
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
    
    # Confronto con ground truth se disponibile
    if gt_translation is not None:
        error = np.linalg.norm(translation - gt_translation)
        result['gt_translation'] = gt_translation
        result['error_mm'] = error
        result['error_percent'] = (error / np.linalg.norm(gt_translation)) * 100
    
    # Stampa verbose
    if verbose:
        print("=" * 60)
        print("🔍 PINHOLE CAMERA MODEL - VERIFICATION")
        print("=" * 60)
        print(f"Bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"Center (u, v): ({u:.1f}, {v:.1f}) pixels")
        print(f"Intrinsics: fx={camera_intrinsics['fx']:.2f}, fy={camera_intrinsics['fy']:.2f}")
        print(f"            cx={camera_intrinsics['cx']:.2f}, cy={camera_intrinsics['cy']:.2f}")
        print(f"Depth Z: {translation[2]:.2f} mm")
        print(f"\n✅ Computed Translation:")
        print(f"   X: {translation[0]:.2f} mm")
        print(f"   Y: {translation[1]:.2f} mm")
        print(f"   Z: {translation[2]:.2f} mm")
        
        if gt_translation is not None:
            print(f"\n📊 Ground Truth:")
            print(f"   X: {gt_translation[0]:.2f} mm")
            print(f"   Y: {gt_translation[1]:.2f} mm")
            print(f"   Z: {gt_translation[2]:.2f} mm")
            print(f"\n⚠️  Error: {result['error_mm']:.2f} mm ({result['error_percent']:.1f}%)")
        print("=" * 60)
    
    return result


# ==================== TEST UTILITIES ====================

def test_pinhole_on_sample():
    """
    Test rapido su un sample del dataset LineMOD.
    Eseguibile come: python -c "from utils.pinhole import test_pinhole_on_sample; test_pinhole_on_sample()"
    """
    from config import Config
    
    print("\n🧪 TEST PINHOLE CAMERA MODEL\n")
    
    # Sample data (oggetto 01 - ape)
    obj_id = '01'
    img_id = 0
    data_root = Config.LINEMOD_ROOT / 'data' / obj_id
    
    # Carica GT per avere bbox e translation reale
    gt_path = data_root / 'gt.yml'
    with open(gt_path, 'r') as f:
        gt_data = yaml.safe_load(f)
    
    gt_pose = gt_data[img_id][0]
    gt_t = np.array(gt_pose['cam_t_m2c']).flatten()
    
    # Simula bbox dal GT (in real scenario viene da YOLO)
    # Per test, creiamo un bbox fittizio centrato
    bbox = [200, 150, 400, 350]
    
    # Carica intrinseci
    intrinsics = load_camera_intrinsics(gt_path)
    
    # Path depth
    depth_path = data_root / 'depth' / f'{img_id:04d}.png'
    
    # Verifica pinhole
    result = verify_pinhole_computation(
        bbox, depth_path, intrinsics, gt_t, verbose=True
    )
    
    print(f"\n✅ Test completato!")
    return result


if __name__ == '__main__':
    # Run test se eseguito direttamente
    test_pinhole_on_sample()
