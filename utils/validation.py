import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from utils.pinhole import load_camera_intrinsics, compute_translation_pinhole_batch
from utils.transforms import quaternion_to_rotation_matrix_batch
from utils.metrics import compute_add_batch_rotation_only, compute_add_batch_full_pose
from utils.file_io import save_validation_results, load_all_models, load_models_info


# GT Crops Validation (ResNet models on pre-cropped images)

def run_pinhole_deep_pipeline(model, test_loader, name='pose_rgb_baseline'):
    """
    Validates baseline model on GT crops using pinhole projection for translation.
    Pipeline: GT crops → ResNet (rotation) → Pinhole (translation)
    """
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / name
    checkpoint_path = checkpoint_dir / "weights" / "best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()
    print(f"Model {name} loaded")
    
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
            
            bboxes = batch['bbox'].cpu().numpy()
            bboxes_xyxy = bboxes.copy()
            bboxes_xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes_xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3]
            
            depth_paths = batch['depth_path']
            camera_intrinsics = load_camera_intrinsics(
                os.path.join(os.path.dirname(depth_paths[0]), '../gt.yml')
            )
            pred_trans = compute_translation_pinhole_batch(bboxes_xyxy, depth_paths, camera_intrinsics)
            pred_trans = pred_trans / 1000.0
            all_pred_translations.append(pred_trans)
            
            if 'translation' in batch:
                all_gt_translations.append(batch['translation'].cpu().numpy())

    print("Concatenating batches...")
    pred_quaternions = torch.cat(all_pred_quaternions, dim=0)
    gt_quaternions = torch.cat(all_gt_quaternions, dim=0)
    obj_ids = np.concatenate(all_obj_ids, axis=0)
    pred_R = quaternion_to_rotation_matrix_batch(pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(gt_quaternions)
    pred_translations = np.concatenate(all_pred_translations, axis=0)
    gt_translations = np.concatenate(all_gt_translations, axis=0) if all_gt_translations else None

    print("Computing metrics: rotation-only ADD, full 6D ADD, pinhole translation error")
    
    results_rot_only = compute_add_batch_rotation_only(
        pred_R, gt_R, obj_ids, models_dict, models_info
    )
    results_rot_only['obj_ids'] = obj_ids

    results_pinhole = None
    if gt_translations is not None:
        pinhole_errors_m = np.linalg.norm(pred_translations - gt_translations, axis=1)
        pinhole_errors_mm = pinhole_errors_m * 1000.0
        results_pinhole = {
            'obj_ids': obj_ids,
            'pinhole_errors': pinhole_errors_mm,
            'pred_translations': pred_translations,
            'gt_translations': gt_translations
        }
        
        results_full_6d = compute_add_batch_full_pose(
            pred_R, pred_translations, gt_R, gt_translations, obj_ids, models_dict, models_info
        )
        results_rot_only['add_values_6d'] = results_full_6d['add_values']
        results_rot_only['is_correct_6d'] = results_full_6d['is_correct']
    
    print("Metrics computed")
    save_validation_results(results_rot_only, results_pinhole, checkpoint_dir)

def run_deep_pose_pipeline(model, test_loader, name="pose_rgb_endtoend"):
    """
    Validates end-to-end model on GT crops.
    Pipeline: GT crops → ResNet (rotation + translation)
    """
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / name
    checkpoint_path = checkpoint_dir / "weights" / "best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()
    print(f"Model {name} loaded")

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

    print("Concatenating batches...")
    pred_quaternions = torch.cat(all_pred_quaternions, dim=0)
    gt_quaternions = torch.cat(all_gt_quaternions, dim=0)
    obj_ids = np.concatenate(all_obj_ids, axis=0)
    pred_R = quaternion_to_rotation_matrix_batch(pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(gt_quaternions)
    pred_translations = np.concatenate(all_pred_translations, axis=0)
    gt_translations = np.concatenate(all_gt_translations, axis=0) if all_gt_translations else None

    print("Computing full pose ADD metrics")
    results_full_pose = compute_add_batch_full_pose(
        pred_R, pred_translations, gt_R, gt_translations, obj_ids, models_dict, models_info
    )
    results_full_pose['obj_ids'] = obj_ids

    save_validation_results(results_full_pose, None, checkpoint_dir)
    print(f"Results saved to {checkpoint_dir / 'validation_result.csv'}")


# Full Pipeline Validation (YOLO + ResNet on full images)

def run_yolo_baseline_pipeline(yolo_model, pose_model, image_loader, name='pose_rgb_baseline', max_samples=None):
    """
    Validates full pipeline: YOLO detection → crop → ResNet baseline → Pinhole.
    
    Args:
        yolo_model: YOLO detector model
        pose_model: ResNet baseline model (rotation only), already loaded in eval mode
        image_loader: DataLoader with full-size images and GT annotations
        name: Checkpoint name for saving results
        max_samples: Maximum number of samples to process (None for all)
    """
    if not all([yolo_model, pose_model, image_loader]):
        raise ValueError("yolo_model, pose_model, and image_loader are required")
    
    import cv2
    from torchvision import transforms
    from utils.bbox_utils import crop_bbox_optimized
    
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / name
    pose_model.eval()
    print(f"Using model {name}")
    
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
            rgb_paths = batch['rgb_path']
            depth_paths = batch['depth_path']
            gt_quaternions = batch['quaternion']
            obj_ids = batch['obj_id']
            gt_translations = batch['translation'] if 'translation' in batch else None

            images_BGR = [cv2.imread(p) for p in rgb_paths]
            # Batch YOLO detection
            detections_batch = yolo_model.detect_objects_batch(images_BGR, conf_threshold=0.3)

            crop_tensors = []
            valid_indices = []
            crop_gt_quaternions = []
            crop_obj_ids = []
            crop_depth_paths = []
            crop_gt_translations = []

            for i, detections in enumerate(detections_batch):
                if len(detections) == 0:
                    detection_failures += 1
                    continue
                bbox_xyxy = detections[0]['bbox']
                cropped_img = crop_bbox_optimized(images_BGR[i], bbox_xyxy, margin=0.15, output_size=(224, 224))
                crop_tensors.append(transform(cropped_img))
                crop_gt_quaternions.append(gt_quaternions[i].unsqueeze(0))
                crop_obj_ids.append([obj_ids[i].item()])
                crop_depth_paths.append(depth_paths[i])
                if gt_translations is not None:
                    crop_gt_translations.append(gt_translations[i].numpy())
                valid_indices.append(i)
                samples_processed += 1
                if max_samples is not None and samples_processed >= max_samples:
                    break

            if not crop_tensors:
                if max_samples is not None and samples_processed >= max_samples:
                    break
                continue

            crop_tensors_batch = torch.stack(crop_tensors).to(Config.DEVICE)
            pred_quaternions_batch = pose_model(crop_tensors_batch)
            crop_gt_quaternions_batch = torch.cat(crop_gt_quaternions, dim=0).to(Config.DEVICE)

            for j in range(len(crop_tensors)):
                all_pred_quaternions.append(pred_quaternions_batch[j].cpu())
                all_gt_quaternions.append(crop_gt_quaternions_batch[j].cpu())
                all_obj_ids.append(crop_obj_ids[j])

                bbox_xyxy = detections_batch[valid_indices[j]][0]['bbox']
                bbox_xywh = np.array([
                    bbox_xyxy[0], bbox_xyxy[1],
                    bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]
                ])
                camera_intrinsics = load_camera_intrinsics(
                    os.path.join(os.path.dirname(crop_depth_paths[j]), '../gt.yml')
                )
                pred_trans = compute_translation_pinhole_batch(
                    np.array([bbox_xywh]), [crop_depth_paths[j]], camera_intrinsics
                )
                pred_trans = pred_trans / 1000.0
                all_pred_translations.append(pred_trans[0])
                if gt_translations is not None:
                    all_gt_translations.append(crop_gt_translations[j])

            if max_samples is not None and samples_processed >= max_samples:
                break

    print(f"Samples processed: {samples_processed}")
    print(f"Detection failures: {detection_failures}")

    if len(all_pred_quaternions) == 0:
        print("No successful detections")
        return None

    print("Concatenating batches...")
    pred_quaternions = torch.stack(all_pred_quaternions, dim=0)
    gt_quaternions = torch.stack(all_gt_quaternions, dim=0)
    obj_ids = np.concatenate(all_obj_ids, axis=0)
    pred_R = quaternion_to_rotation_matrix_batch(pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(gt_quaternions)
    pred_translations = np.stack(all_pred_translations, axis=0)
    gt_translations = np.stack(all_gt_translations, axis=0) if all_gt_translations else None

    if gt_translations is None:
        raise ValueError("Ground truth translations required for full 6D pose evaluation")

    print("Computing full 6D pose ADD metrics")
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

    print("Metrics computed")
    save_validation_results(results_full_pose, results_pinhole, checkpoint_dir)
    print(f"Results saved to {checkpoint_dir / 'validation_result.csv'}")

def run_yolo_endtoend_pipeline(yolo_model, pose_model, image_loader, name='pose_rgb_endtoend', max_samples=None):
    """
    Validates full pipeline: YOLO detection → crop → ResNet end-to-end.
    
    Args:
        yolo_model: YOLO detector model
        pose_model: ResNet end-to-end model (rotation + translation), already loaded in eval mode
        image_loader: DataLoader with full-size images and GT annotations
        name: Checkpoint name for saving results
        max_samples: Maximum number of samples to process (None for all)
    """
    import cv2
    from torchvision import transforms
    from utils.bbox_utils import crop_bbox_optimized
    
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / name
    pose_model.eval()
    print(f"Using model {name}")
    
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

    desc = f"Validating YOLO pipeline (end-to-end{f', max {max_samples} samples' if max_samples else ''})"
    with torch.no_grad():
        for batch in tqdm(image_loader, desc=desc):
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
                all_pred_translations.append(pred_translation.cpu().numpy())
                
                if gt_translation is not None:
                    all_gt_translations.append(gt_translation)
                
                samples_processed += 1
            
            if max_samples is not None and samples_processed >= max_samples:
                break

    print(f"Samples processed: {samples_processed}")
    print(f"Detection failures: {detection_failures}")

    if len(all_pred_quaternions) == 0:
        print("No successful detections")
        return None
    pred_quaternions = torch.stack(all_pred_quaternions, dim=0)
    gt_quaternions = torch.stack(all_gt_quaternions, dim=0)
    obj_ids = np.concatenate(all_obj_ids, axis=0)
    pred_R = quaternion_to_rotation_matrix_batch(pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(gt_quaternions)
    pred_translations = np.stack(all_pred_translations, axis=0)
    gt_translations = np.stack(all_gt_translations, axis=0) if all_gt_translations else None
    
    print("Computing full pose ADD metrics")
    results_full_pose = compute_add_batch_full_pose(
        pred_R, pred_translations, gt_R, gt_translations, obj_ids, models_dict, models_info
    )
    results_full_pose['obj_ids'] = obj_ids

    save_validation_results(results_full_pose, None, checkpoint_dir)
    print(f"Results saved to {checkpoint_dir / 'validation_result.csv'}")