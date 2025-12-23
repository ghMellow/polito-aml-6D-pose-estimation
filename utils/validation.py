import pandas as pd
import numpy as np
from config import Config


def save_validation_results(results_rot_only, results_pinhole, checkpoint_dir):
    """
    Salva i risultati di validazione in un file CSV.
    """
    validation_results = []
    add_values = results_rot_only.get('add_values', None)
    is_correct = results_rot_only.get('is_correct', None)
    obj_ids_rot = results_rot_only.get('obj_ids', None)
    if add_values is not None and is_correct is not None and obj_ids_rot is not None:
        for i in range(len(add_values)):
            validation_results.append({
                'obj_id': obj_ids_rot[i],
                'add_value': add_values[i],
                'is_correct': is_correct[i]
            })
    if results_pinhole is not None:
        pinhole_errors = results_pinhole.get('pinhole_errors', None)
        pred_translations = results_pinhole.get('pred_translations', None)
        gt_translations = results_pinhole.get('gt_translations', None)
        obj_ids_pinhole = results_pinhole.get('obj_ids', None)
        if pinhole_errors is not None and obj_ids_pinhole is not None:
            for i in range(len(pinhole_errors)):
                # Aggiorna la riga corrispondente se esiste, altrimenti aggiungi nuova
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
        import os
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
        'is_correct': df_val['is_correct'].values if 'is_correct' in df_val else None
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


def calc_add_accuracy_per_class(results_rot_only, linemod_objects):
    """
    Calcola media ADD e accuracy per classe e globale.
    Restituisce una tabella e le metriche globali.
    """
    obj_ids_rot = np.array(results_rot_only['obj_ids'])
    add_values = np.array(results_rot_only['add_values'])
    is_correct = np.array(results_rot_only['is_correct'])
    data = []
    for obj_id, obj_name in linemod_objects.items():
        mask = obj_ids_rot == obj_id
        if np.sum(mask) == 0:
            continue
        mean_add = add_values[mask].mean()
        acc = is_correct[mask].mean() * 100
        data.append({
            'Classe': f"{obj_id:02d} - {obj_name.get('name')}",
            'Media ADD (rot-only)': f"{mean_add:.2f}",
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

def run_pinhole_deep_pipeline(model, test_loader):
    """
    Esegue la pipeline pinhole+deep per la valutazione su test set.
    Salva le metriche di validazione.
    """
    import torch
    import numpy as np
    from tqdm import tqdm
    from utils.pinhole import load_camera_intrinsics, compute_translation_pinhole
    import os
    from utils.transforms import quaternion_to_rotation_matrix_batch
    from utils.metrics import compute_add_batch_rotation_only, load_all_models, load_models_info

    NAME = "test_rotationonly_1"
    checkpoint_dir = Config.CHECKPOINT_DIR / "pose" / NAME
    checkpoint_weights_dir = checkpoint_dir / "weights"
    best_path = checkpoint_weights_dir / "best.pt"

    # Carica il modello trained (se necessario)
    try:
        model.load_state_dict(torch.load(best_path, map_location=Config.DEVICE))
        model.eval()
        print(f"✅ Modello {NAME} caricato e in modalità eval!")
    except Exception as e:
        print(f"⚠️  Modello non trovato o già caricato. Errore: {e}")
        raise SystemExit("Stop right there!")    

    models_dict = load_all_models()
    models_info = load_models_info(Config.MODELS_INFO_PATH)

    all_pred_quaternions = []
    all_gt_quaternions = []
    all_obj_ids = []
    all_pred_translations = []
    all_gt_translations = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Valutazione su test set"):
            images = batch['rgb_crop'].to(Config.DEVICE)
            gt_quaternions = batch['quaternion'].to(Config.DEVICE)
            obj_ids = batch['obj_id'].cpu().numpy()
            pred_quaternions = model(images)
            all_pred_quaternions.append(pred_quaternions.cpu())
            all_gt_quaternions.append(gt_quaternions.cpu())
            all_obj_ids.append(obj_ids)
            # --- Calcolo translation con pinhole ---
            bboxes = batch['bbox'].cpu().numpy()  # [batch, 4]
            depth_paths = batch['depth_path']     # lista di path (stringhe)
            camera_intrinsics = load_camera_intrinsics(
                os.path.join(os.path.dirname(depth_paths[0]), '../gt.yml')
            )
            pred_translations = []
            gt_translations = batch['translation'].cpu().numpy() if 'translation' in batch else None
            for i in range(len(bboxes)):
                try:
                    pred_t = compute_translation_pinhole(bboxes[i], depth_paths[i], camera_intrinsics)
                except Exception as e:
                    print(f"⚠️  Pinhole failed: {e}")
                    pred_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                pred_translations.append(pred_t)
            all_pred_translations.append(np.stack(pred_translations))
            if gt_translations is not None:
                all_gt_translations.append(gt_translations)

    print("\nconcatenazione batch")
    all_pred_quaternions = torch.cat(all_pred_quaternions, dim=0)
    all_gt_quaternions = torch.cat(all_gt_quaternions, dim=0)
    all_obj_ids = np.concatenate(all_obj_ids, axis=0)
    all_pred_translations = np.concatenate(all_pred_translations, axis=0)
    if all_gt_translations:
        all_gt_translations = np.concatenate(all_gt_translations, axis=0)
    else:
        all_gt_translations = None

    print("conversione da quaternoni a matrici di rotazione")
    pred_R = quaternion_to_rotation_matrix_batch(all_pred_quaternions)
    gt_R = quaternion_to_rotation_matrix_batch(all_gt_quaternions)

    print("calcolo metriche: ADD rot-only, traslazione pinhole")
    # 1. ADD solo rotazione (traslazione dummy)
    results_rot_only = compute_add_batch_rotation_only(
        pred_R, gt_R, all_obj_ids, models_dict, models_info
    )
    results_rot_only['obj_ids'] = all_obj_ids

    # 2. Statistiche traslazione pinhole (vs GT)
    results_pinhole = {}
    if all_gt_translations is not None:
        pinhole_errors = np.linalg.norm(all_pred_translations - all_gt_translations, axis=1)
        results_pinhole['obj_ids'] = all_obj_ids
        results_pinhole['pinhole_errors'] = pinhole_errors
        results_pinhole['pred_translations'] = all_pred_translations
        results_pinhole['gt_translations'] = all_gt_translations
    else:
        results_pinhole = None
    print("✅ Metriche calcolate.")

    # Salva i risultati
    save_validation_results(results_rot_only, results_pinhole, checkpoint_dir)