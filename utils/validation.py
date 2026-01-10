import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from utils.pinhole import load_camera_intrinsics, compute_translation_pinhole_batch
from utils.transforms import quaternion_to_rotation_matrix_batch
from utils.metrics import compute_add_batch_rotation_only, load_all_models, load_models_info, compute_add_batch_full_pose


# ============================================================================
# Load & Save
# ============================================================================

def save_validation_results(results_rot_only, results_pinhole, checkpoint_dir):
    """
    Salva i risultati di validazione in un file CSV.
    """
    validation_results = []
    add_values = results_rot_only.get('add_values', None)
    is_correct = results_rot_only.get('is_correct', None)
    add_values_6d = results_rot_only.get('add_values_6d', None)
    is_correct_6d = results_rot_only.get('is_correct_6d', None)
    obj_ids_rot = results_rot_only.get('obj_ids', None)
    if add_values is not None and is_correct is not None and obj_ids_rot is not None:
        for i in range(len(add_values)):
            row = {
                'obj_id': obj_ids_rot[i],
                'add_value': add_values[i],
                'is_correct': is_correct[i]
            }
            # Aggiungi ADD 6D se disponibile
            if add_values_6d is not None and i < len(add_values_6d):
                row['add_value_6d'] = add_values_6d[i]
            if is_correct_6d is not None and i < len(is_correct_6d):
                row['is_correct_6d'] = is_correct_6d[i]
            validation_results.append(row)
    if results_pinhole is not None:
        pinhole_errors = results_pinhole.get('pinhole_errors', None)
        pred_translations = results_pinhole.get('pred_translations', None)
        gt_translations = results_pinhole.get('gt_translations', None)
        obj_ids_pinhole = results_pinhole.get('obj_ids', None)
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
        if hasattr(checkpoint_dir, 'joinpath'):
            val_csv_path = checkpoint_dir.joinpath('validation_result.csv')
        else:
            val_csv_path = os.path.join(str(checkpoint_dir), 'validation_result.csv')
        df_val.to_csv(val_csv_path, index=False)
        print(f"✅ Risultati di validazione salvati in {val_csv_path}")


def load_validation_results(val_csv_path):
    """
    Carica i risultati di validazione dal CSV e restituisce i dizionari results_rot_only e results_pinhole.
    """
    df_val = pd.read_csv(val_csv_path)
    results_rot_only = {
        'obj_ids': df_val['obj_id'].values,
        'add_values': df_val['add_value'].values if 'add_value' in df_val else None,
        'is_correct': df_val['is_correct'].values if 'is_correct' in df_val else None,
        'add_values_6d': df_val['add_value_6d'].values if 'add_value_6d' in df_val else None,
        'is_correct_6d': df_val['is_correct_6d'].values if 'is_correct_6d' in df_val else None
    }
    if 'pinhole_error' in df_val:
        results_pinhole = {
            'obj_ids': df_val['obj_id'].values,
            'pinhole_errors': df_val['pinhole_error'].values,
            'pred_translations': df_val['pred_translation'].values if 'pred_translation' in df_val else None,
            'gt_translations': df_val['gt_translation'].values if 'gt_translation' in df_val else None
        }
    else:
        results_pinhole = None
    return results_rot_only, results_pinhole


def calc_add_accuracy_per_class(results_rot_only, linemod_objects, use_6d=False):
    """
    Calcola media ADD e accuracy per classe e globale.
    
    Args:
        results_rot_only: Dict con 'obj_ids', 'add_values', 'is_correct' (e opzionalmente 'add_values_6d', 'is_correct_6d')
        linemod_objects: Dict con mapping obj_id -> nome oggetto
        use_6d: Se True, usa ADD completa 6D invece di rot-only (default: False)
    
    Returns:
        data: Lista di dict per DataFrame
        global_add: Media globale ADD (mm)
        global_acc: Accuracy globale (%)
    """
    obj_ids_rot = np.array(results_rot_only['obj_ids'])
    
    # Scegli quale ADD usare
    if use_6d and 'add_values_6d' in results_rot_only and results_rot_only['add_values_6d'] is not None:
        add_values = np.array(results_rot_only['add_values_6d'])
        is_correct = np.array(results_rot_only['is_correct_6d'])
        add_label = 'Media ADD (6D)'
    else:
        add_values = np.array(results_rot_only['add_values'])
        is_correct = np.array(results_rot_only['is_correct'])
        add_label = 'Media ADD'
    
    data = []
    for obj_id, obj_name in linemod_objects.items():
        mask = obj_ids_rot == obj_id
        if np.sum(mask) == 0:
            continue
        mean_add = add_values[mask].mean()
        acc = is_correct[mask].mean() * 100
        data.append({
            'Classe': f"{obj_id:02d} - {obj_name.get('name')}",
            add_label: f"{mean_add:.2f}",
            'Accuracy (%)': f"{acc:.1f}"
        })
    global_add = add_values.mean()
    global_acc = is_correct.mean() * 100
    return data, global_add, global_acc


def calc_pinhole_error_per_class(results_pinhole, linemod_objects):
    """
    Calcola errore medio pinhole per classe e globale.
    Restituisce una tabella e la metrica globale.
    """
    obj_ids_pinhole = np.array(results_pinhole['obj_ids'])
    pinhole_errors = np.array(results_pinhole['pinhole_errors'])
    data = []
    for obj_id, obj_name in linemod_objects.items():
        mask = obj_ids_pinhole == obj_id
        if np.sum(mask) == 0:
            continue
        mean_pinhole = pinhole_errors[mask].mean()
        data.append({
            'Classe': f"{obj_id:02d} - {obj_name.get('name')}",
            'Err. Pinhole medio (mm)': f"{mean_pinhole:.2f}"
        })
    global_pinhole = pinhole_errors.mean()
    return data, global_pinhole


def calc_combined_results_per_class(results_rot_only, results_pinhole, linemod_objects):
    """
    Combina ADD completa (6D pose: rotation + translation) e pinhole error in un'unica tabella per classe.
    
    Args:
        results_rot_only: Dict con 'obj_ids', 'add_values_6d', 'is_correct_6d' (ADD completa 6D)
        results_pinhole: Dict con 'obj_ids', 'pinhole_errors'
        linemod_objects: Dict con mapping obj_id -> nome oggetto
    
    Returns:
        data: Lista di dict per DataFrame
        global_add_6d: Media globale ADD 6D completa (mm)
        global_acc_6d: Accuracy globale 6D (%)
        global_pinhole: Errore medio globale pinhole (mm)
    """
    obj_ids_rot = np.array(results_rot_only['obj_ids'])
    
    # Usa ADD completa (6D) se disponibile, altrimenti fallback su ADD rot-only
    if 'add_values_6d' in results_rot_only:
        add_values = np.array(results_rot_only['add_values_6d'])
        is_correct = np.array(results_rot_only['is_correct_6d'])
        add_label = "ADD 6D (mm)"
    else:
        add_values = np.array(results_rot_only['add_values'])
        is_correct = np.array(results_rot_only['is_correct'])
        add_label = "ADD rot (mm)"
    
    obj_ids_pinhole = np.array(results_pinhole['obj_ids'])
    pinhole_errors = np.array(results_pinhole['pinhole_errors'])
    
    data = []
    for obj_id, obj_name in linemod_objects.items():
        mask_rot = obj_ids_rot == obj_id
        mask_pinhole = obj_ids_pinhole == obj_id
        
        if np.sum(mask_rot) == 0:
            continue
        
        mean_add = add_values[mask_rot].mean()
        acc = is_correct[mask_rot].mean() * 100
        
        # Pinhole error (potrebbe non essere disponibile per tutti)
        mean_pinhole = pinhole_errors[mask_pinhole].mean() if np.sum(mask_pinhole) > 0 else 0.0
        
        data.append({
            'Classe': f"{obj_id:02d} - {obj_name.get('name')}",
            add_label: f"{mean_add:.2f}",
            'Accuracy (%)': f"{acc:.1f}",
            'Err. Pinhole (mm)': f"{mean_pinhole:.2f}"
        })
    
    global_add = add_values.mean()
    global_acc = is_correct.mean() * 100
    global_pinhole = pinhole_errors.mean()
    
    return data, global_add, global_acc, global_pinhole


# ============================================================================
# GT Crops Validation (ResNet models on pre-cropped images)
# ============================================================================

def run_pinhole_deep_pipeline(model, test_loader, name='test_rotationonly_1'):
    """
    Valida il modello baseline (rotation-only) su GT crops.
    Pipeline: GT crops → ResNet (rotation) → Pinhole (translation)
    
    Args:
        model: ResNet baseline model (predice solo rotation)
        test_loader: DataLoader con GT crops (LineMODPoseDataset)
        name: Nome del checkpoint
    """
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / name
    checkpoint_path = checkpoint_dir / "weights" / "best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()
    print(f"✅ Modello {name} caricato!")
    
    models_dict = load_all_models()
    models_info = load_models_info(Config.MODELS_INFO_PATH)

    all_pred_quaternions = []
    all_gt_quaternions = []
    all_obj_ids = []
    all_pred_translations = []
    all_gt_translations = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validazione GT crops (baseline)"):
            images = batch['rgb_crop'].to(Config.DEVICE)
            gt_quaternions = batch['quaternion'].to(Config.DEVICE)
            obj_ids = batch['obj_id'].cpu().numpy()
            
            pred_quaternions = model(images)
            all_pred_quaternions.append(pred_quaternions.cpu())
            all_gt_quaternions.append(gt_quaternions.cpu())
            all_obj_ids.append(obj_ids)
            
            # Calcolo translation con pinhole batch
            bboxes = batch['bbox'].cpu().numpy()
            
            # Converti bbox da [x, y, w, h] a [x1, y1, x2, y2] per pinhole
            bboxes_xyxy = bboxes.copy()
            bboxes_xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x2 = x1 + w
            bboxes_xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y2 = y1 + h
            
            depth_paths = batch['depth_path']
            camera_intrinsics = load_camera_intrinsics(
                os.path.join(os.path.dirname(depth_paths[0]), '../gt.yml')
            )
            pred_trans = compute_translation_pinhole_batch(bboxes_xyxy, depth_paths, camera_intrinsics)
            
            # Converti da millimetri a metri per consistenza con GT
            pred_trans = pred_trans / 1000.0
            
            all_pred_translations.append(pred_trans)
            
            if 'translation' in batch:
                all_gt_translations.append(batch['translation'].cpu().numpy())

    # Concatenazione
    print("Concatenazione batch...")
    pred_quaternions = torch.cat(all_pred_quaternions, dim=0)
    gt_quaternions = torch.cat(all_gt_quaternions, dim=0)
    obj_ids = np.concatenate(all_obj_ids, axis=0)
    pred_R = quaternion_to_rotation_matrix_batch(pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(gt_quaternions)
    pred_translations = np.concatenate(all_pred_translations, axis=0)
    gt_translations = np.concatenate(all_gt_translations, axis=0) if all_gt_translations else None

    print("Calcolo metriche: ADD rot-only, ADD completa (6D), traslazione pinhole")
    
    # 1. ADD solo rotazione (rotation + dummy translation)
    results_rot_only = compute_add_batch_rotation_only(
        pred_R, gt_R, obj_ids, models_dict, models_info
    )
    results_rot_only['obj_ids'] = obj_ids

    # 2. Errore traslazione pinhole
    results_pinhole = None
    if gt_translations is not None:
        # Calcola errori in metri, poi converti in mm per visualizzazione
        pinhole_errors_m = np.linalg.norm(pred_translations - gt_translations, axis=1)
        pinhole_errors_mm = pinhole_errors_m * 1000.0
        results_pinhole = {
            'obj_ids': obj_ids,
            'pinhole_errors': pinhole_errors_mm,  # In millimetri
            'pred_translations': pred_translations,  # In metri
            'gt_translations': gt_translations  # In metri
        }
        
        # 3. ADD completa (6D pose: rotation predetta + translation pinhole)
        results_full_6d = compute_add_batch_full_pose(
            pred_R, pred_translations, gt_R, gt_translations, obj_ids, models_dict, models_info
        )
        # Aggiungi ADD completa ai risultati rot_only
        results_rot_only['add_values_6d'] = results_full_6d['add_values']
        results_rot_only['is_correct_6d'] = results_full_6d['is_correct']
    
    print("✅ Metriche calcolate.")
    save_validation_results(results_rot_only, results_pinhole, checkpoint_dir)


def run_deep_pose_pipeline(model, test_loader, name="test_endtoend_pose_1"):
    """
    Valida il modello end-to-end (rotation+translation) su GT crops.
    Pipeline: GT crops → ResNet (rotation + translation)
    
    Args:
        model: ResNet end-to-end model (predice rotation + translation)
        test_loader: DataLoader con GT crops (LineMODPoseDataset)
        name: Nome del checkpoint
    """
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / name
    checkpoint_path = checkpoint_dir / "weights" / "best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()
    print(f"✅ Modello {name} caricato!")

    models_dict = load_all_models()
    models_info = load_models_info(Config.MODELS_INFO_PATH)

    all_pred_quaternions = []
    all_gt_quaternions = []
    all_obj_ids = []
    all_pred_translations = []
    all_gt_translations = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validazione GT crops (end-to-end)"):
            images = batch['rgb_crop'].to(Config.DEVICE)
            gt_quaternions = batch['quaternion'].to(Config.DEVICE)
            obj_ids = batch['obj_id'].cpu().numpy()
            
            pred_quaternions, pred_translations = model(images)
            all_pred_quaternions.append(pred_quaternions.cpu())
            all_gt_quaternions.append(gt_quaternions.cpu())
            all_obj_ids.append(obj_ids)
            all_pred_translations.append(pred_translations.cpu().numpy())
            
            if 'translation' in batch:
                all_gt_translations.append(batch['translation'].cpu().numpy())

    # Concatenazione
    print("Concatenazione batch...")
    pred_quaternions = torch.cat(all_pred_quaternions, dim=0)
    gt_quaternions = torch.cat(all_gt_quaternions, dim=0)
    obj_ids = np.concatenate(all_obj_ids, axis=0)
    pred_R = quaternion_to_rotation_matrix_batch(pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(gt_quaternions)
    pred_translations = np.concatenate(all_pred_translations, axis=0)
    gt_translations = np.concatenate(all_gt_translations, axis=0) if all_gt_translations else None

    print("Calcolo metriche: ADD full pose (end-to-end)")
    results_full_pose = compute_add_batch_full_pose(
        pred_R, pred_translations, gt_R, gt_translations, obj_ids, models_dict, models_info
    )
    results_full_pose['obj_ids'] = obj_ids

    save_validation_results(results_full_pose, None, checkpoint_dir)
    print(f"✅ Risultati salvati in {checkpoint_dir / 'validation_result.csv'}")


# ============================================================================
# Full Pipeline Validation (YOLO + ResNet on full images)
# ============================================================================
def run_yolo_baseline_pipeline(yolo_model, pose_model, image_loader, name, max_samples=None):
    """
    Valida la pipeline completa: YOLO detection → crop → ResNet baseline → Pinhole
    
    Args:
        yolo_model: YoloDetector model per detection (required)
        pose_model: ResNet baseline model (predice solo rotation) - già caricato e in eval mode (required)
        image_loader: DataLoader con immagini full-size + GT annotations (required)
        name: Nome del checkpoint (per salvare risultati)
        max_samples: Numero massimo di campioni da processare (None = tutti). Utile per debug rapido.
    
    Raises:
        ValueError: Se yolo_model, pose_model o image_loader sono None
    """
    # Validazione parametri obbligatori
    if yolo_model is None:
        raise ValueError("yolo_model non può essere None")
    if pose_model is None:
        raise ValueError("pose_model non può essere None")
    if image_loader is None:
        raise ValueError("image_loader non può essere None")
    
    import cv2
    from torchvision import transforms
    from utils.bbox_utils import crop_bbox_optimized
    
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / name
    
    pose_model.eval()
    print(f"✅ Usando modello {name} (già caricato)!")
    
    models_dict = load_all_models()
    models_info = load_models_info(Config.MODELS_INFO_PATH)

    all_pred_quaternions = []
    all_gt_quaternions = []
    all_obj_ids = []
    all_pred_translations = []
    all_gt_translations = []
    detection_failures = 0
    samples_processed = 0
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    desc = f"Validazione YOLO pipeline (baseline{f', max {max_samples} samples' if max_samples else ''})"
    with torch.no_grad():
        for batch in tqdm(image_loader, desc=desc):
            for i in range(len(batch['rgb_path'])):
                rgb_path = batch['rgb_path'][i]
                depth_path = batch['depth_path'][i]
                gt_quaternion = batch['quaternion'][i].unsqueeze(0).to(Config.DEVICE)
                gt_translation = batch['translation'][i].numpy() if 'translation' in batch else None
                obj_id = batch['obj_id'][i].item()
                
                image_BGR = cv2.imread(rgb_path)
                
                detections = yolo_model.detect_objects(image_BGR, conf_threshold=0.3)
                if len(detections) == 0:
                    detection_failures += 1
                    continue
                
                bbox_xyxy = detections[0]['bbox']
                cropped_img = crop_bbox_optimized(image_BGR, bbox_xyxy, margin=0.15, output_size=(224, 224))
                cropped_tensor = transform(cropped_img).unsqueeze(0).to(Config.DEVICE)
                
                pred_quaternion = pose_model(cropped_tensor)
                all_pred_quaternions.append(pred_quaternion.cpu().squeeze(0))
                all_gt_quaternions.append(gt_quaternion.cpu().squeeze(0))
                all_obj_ids.append([obj_id])

                # Pinhole translation
                bbox_xywh = np.array([
                    bbox_xyxy[0], bbox_xyxy[1],
                    bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]
                ])
                camera_intrinsics = load_camera_intrinsics(
                    os.path.join(os.path.dirname(depth_path), '../gt.yml')
                )
                pred_trans = compute_translation_pinhole_batch(
                    np.array([bbox_xywh]), [depth_path], camera_intrinsics
                )
                # Converti da millimetri a metri per consistenza con GT
                pred_trans = pred_trans / 1000.0
                all_pred_translations.append(pred_trans[0])

                if gt_translation is not None:
                    all_gt_translations.append(gt_translation)
                
                samples_processed += 1
            
            if max_samples is not None and samples_processed >= max_samples:
                break

    print(f"📊 Campioni processati: {samples_processed}")
    print(f"⚠️  Detection failures: {detection_failures}")

    if len(all_pred_quaternions) == 0:
        print("❌ No successful detections! All detections failed.")
        return None

    # Concatenazione
    print("Concatenazione batch...")
    pred_quaternions = torch.stack(all_pred_quaternions, dim=0)
    gt_quaternions = torch.stack(all_gt_quaternions, dim=0)
    obj_ids = np.concatenate(all_obj_ids, axis=0)
    pred_R = quaternion_to_rotation_matrix_batch(pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(gt_quaternions)
    pred_translations = np.stack(all_pred_translations, axis=0)
    gt_translations = np.stack(all_gt_translations, axis=0) if all_gt_translations else None

    # Validazione parametri per compute_add_batch_full_pose
    if gt_translations is None:
        raise ValueError("gt_translations è None - impossibile calcolare ADD completa senza ground truth translation")

    print("Calcolo metriche: ADD completa 6D pose (rotation predetta + translation pinhole)")
    results_full_pose = compute_add_batch_full_pose(
        pred_R, pred_translations, gt_R, gt_translations, obj_ids, models_dict, models_info
    )
    results_full_pose['obj_ids'] = obj_ids

    # Calcolo errore traslazione pinhole
    pinhole_errors_m = np.linalg.norm(pred_translations - gt_translations, axis=1)
    pinhole_errors_mm = pinhole_errors_m * 1000.0
    results_pinhole = {
        'obj_ids': obj_ids,
        'pinhole_errors': pinhole_errors_mm,
        'pred_translations': pred_translations,
        'gt_translations': gt_translations
    }

    print("✅ Metriche calcolate.")
    save_validation_results(results_full_pose, results_pinhole, checkpoint_dir)
    print(f"✅ Risultati salvati in {checkpoint_dir / 'validation_result.csv'}")

def run_yolo_endtoend_pipeline(yolo_model, pose_model, image_loader, name='yolo_endtoend_pipeline', max_samples=None):
    """
    Valida la pipeline completa: YOLO detection → crop → ResNet end-to-end
    
    Args:
        yolo_model: YoloDetector model per detection
        pose_model: ResNet end-to-end model (predice rotation + translation) - già caricato e in eval mode
        image_loader: DataLoader con immagini full-size + GT annotations
        name: Nome del checkpoint (per salvare risultati)
        max_samples: Numero massimo di campioni da processare (None = tutti). Utile per debug rapido.
    """
    import cv2
    from torchvision import transforms
    from utils.bbox_utils import crop_bbox_optimized
    
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / name

    # ✅ FIX: Non ricaricare il modello, è già stato caricato nel notebook
    pose_model.eval()
    print(f"✅ Usando modello {name} (già caricato)!")
    
    models_dict = load_all_models()
    models_info = load_models_info(Config.MODELS_INFO_PATH)

    all_pred_quaternions = []
    all_gt_quaternions = []
    all_obj_ids = []
    all_pred_translations = []
    all_gt_translations = []
    detection_failures = 0
    samples_processed = 0
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    desc = f"Validazione YOLO pipeline (end-to-end{f', max {max_samples} samples' if max_samples else ''})"
    with torch.no_grad():
        for batch in tqdm(image_loader, desc="Validazione YOLO pipeline (end-to-end)"):
            for i in range(len(batch['rgb_path'])):
                rgb_path = batch['rgb_path'][i]
                gt_quaternion = batch['quaternion'][i].unsqueeze(0).to(Config.DEVICE)
                gt_translation = batch['translation'][i].numpy() if 'translation' in batch else None
                obj_id = batch['obj_id'][i].item()
                
                image_BGR = cv2.imread(rgb_path)
                
                detections = yolo_model.detect_objects(image_BGR, conf_threshold=0.3)
                if len(detections) == 0:
                    detection_failures += 1
                    continue
                
                bbox_xyxy = detections[0]['bbox']
                cropped_img = crop_bbox_optimized(image_BGR, bbox_xyxy, margin=0.15, output_size=(224, 224))
                cropped_tensor = transform(cropped_img).unsqueeze(0).to(Config.DEVICE)
                
                pred_quaternion, pred_translation = pose_model(cropped_tensor)
                all_pred_quaternions.append(pred_quaternion.cpu().squeeze(0))
                all_gt_quaternions.append(gt_quaternion.cpu().squeeze(0))
                all_obj_ids.append([obj_id])
                # ✅ FIX: Mantieni shape 2D per translations (1, 3) → squeeze solo batch
                all_pred_translations.append(pred_translation.cpu().numpy())  # Shape (1, 3)
                
                if gt_translation is not None:
                    all_gt_translations.append(gt_translation)
                
                samples_processed += 1
            
            # ✅ Early exit if max_samples reached
            if max_samples is not None and samples_processed >= max_samples:
                break

    print(f"📊 Campioni processati: {samples_processed}")
    print(f"⚠️  Detection failures: {detection_failures}")

    # Check if we have any successful predictions
    if len(all_pred_quaternions) == 0:
        print("❌ No successful detections! All detections failed.")
        return None

    # ✅ FIX: stack invece di concatenate per array 1D
    pred_translations = np.stack(all_pred_translations, axis=0)
    gt_translations = np.stack
    pred_quaternions = torch.stack(all_pred_quaternions, dim=0)
    gt_quaternions = torch.stack(all_gt_quaternions, dim=0)
    obj_ids = np.concatenate(all_obj_ids, axis=0)
    pred_R = quaternion_to_rotation_matrix_batch(pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(gt_quaternions)
    # ✅ FIX: stack invece di concatenate per array 1D
    pred_translations = np.stack(all_pred_translations, axis=0)
    gt_translations = np.stack(all_gt_translations, axis=0) if all_gt_translations else None
    
    print("Calcolo metriche: ADD full pose (YOLO pipeline end-to-end)")
    results_full_pose = compute_add_batch_full_pose(
        pred_R, pred_translations, gt_R, gt_translations, obj_ids, models_dict, models_info
    )
    results_full_pose['obj_ids'] = obj_ids

    save_validation_results(results_full_pose, None, checkpoint_dir)
    print(f"✅ Risultati salvati in {checkpoint_dir / 'validation_result.csv'}")