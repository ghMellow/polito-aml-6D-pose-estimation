"""
Visualization utilities for 6D Pose Estimation training and evaluation.
Centralizes plotting of losses, learning rates, and per-class metrics.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(history):
    """
    Plot training loss, translation loss, rotation loss, and learning rates.
    Args:
        history: dict with keys 'train_loss', 'train_trans', 'train_rot', 'lr_backbone', 'lr_head'
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Total loss
    axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    # Translation loss
    axes[0, 1].plot(history['train_trans'], 'g-', linewidth=2)
    axes[0, 1].set_title('Translation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    # Rotation loss
    axes[1, 0].plot(history['train_rot'], 'r-', linewidth=2)
    axes[1, 0].set_title('Rotation Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    # Learning rates
    if 'lr_backbone' in history and 'lr_head' in history:
        axes[1, 1].plot(history['lr_backbone'], 'purple', linewidth=2, label='Backbone')
        axes[1, 1].plot(history['lr_head'], 'orange', linewidth=2, label='Head')
        axes[1, 1].set_title('Learning Rates', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def show_per_class_table(results, class_dict):
    """
    Show table of mean ADD and accuracy per class.
    Args:
        results: dict with 'obj_ids', 'add_values', 'is_correct'
        class_dict: dict mapping obj_id to class info (e.g., Config.LINEMOD_OBJECTS)
    """
    import numpy as np
    obj_ids = np.array(results['obj_ids'])
    add_values = np.array(results['add_values'])
    is_correct = np.array(results['is_correct'])
    data = []
    for obj_id, obj_name in class_dict.items():
        mask = obj_ids == obj_id
        if np.sum(mask) == 0:
            continue
        mean_add = add_values[mask].mean()
        acc = is_correct[mask].mean() * 100
        data.append({
            'Classe': f"{obj_id:02d} - {obj_name.get('name')}",
            'Media ADD': f"{mean_add:.2f}",
            'Accuracy (%)': f"{acc:.1f}"
        })
    df = pd.DataFrame(data)
    display(df)
    print("\nMedia globale ADD:", f"{add_values.mean():.2f}")
    print("Accuracy globale (%):", f"{is_correct.mean()*100:.1f}")

def show_pose_samples(batch, n=4):
    """
    Visualizza alcuni sample del batch del pose dataset (immagini già croppate, quaternion, obj_id).
    Args:
        batch: dict con chiavi 'rgb_crop' (o 'rgb'), 'quaternion', 'obj_id'
        n: numero di immagini da mostrare (default 4)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    images = batch['rgb'] if 'rgb' in batch else batch['rgb_crop']
    quaternions = batch['quaternion']
    obj_ids = batch['obj_id']
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(min(n, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        quat = quaternions[i].cpu().numpy()
        obj_id = obj_ids[i].item()
        axes[i].imshow(img)
        axes[i].set_title(
            f"Object {obj_id:02d}\n"
            f"Quat: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]",
            fontsize=10
        )
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Visualizzazione sample del training set (immagini già croppate dal dataset)")
    print(f"Quaternion normalizzato (||q|| = 1)")

def plot_training_validation_loss_from_csv(results_path):
    """
    Plotta la curva della training loss dal file results.csv generato dal training baseline.
    """
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        plt.figure(figsize=(8,5))
        if 'train_loss' in df.columns:
            plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
        if 'val_loss' in df.columns:
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(f"File {results_path} non trovato.")

def show_pose_samples_with_add(images, gt_quaternions, pred_quaternions, obj_ids, add_values, threshold=0.02):
    """
    Visualizza 4 sample con info su GT, predizione e ADD (errore di rotazione).
    Args:
        images: batch di immagini torch [B, 3, H, W]
        gt_quaternions: batch torch [B, 4]
        pred_quaternions: batch torch [B, 4]
        obj_ids: batch torch o numpy [B]
        add_values: lista/array di ADD per sample
        threshold: soglia per colorazione (default 0.02)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    for i in range(min(4, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        gt_q = gt_quaternions[i].cpu().numpy()
        pred_q = pred_quaternions[i].cpu().numpy()
        obj_id = obj_ids[i].item() if hasattr(obj_ids[i], 'item') else obj_ids[i]
        add_val = add_values[i]
        axes[i].imshow(img)
        title = (
            f"Object {obj_id:02d}\n"
            f"GT Quat: [{gt_q[0]:.2f}, {gt_q[1]:.2f}, {gt_q[2]:.2f}, {gt_q[3]:.2f}]\n"
            f"Pred Quat: [{pred_q[0]:.2f}, {pred_q[1]:.2f}, {pred_q[2]:.2f}, {pred_q[3]:.2f}]\n"
            f"ADD: {add_val:.2f}"
        )
        color = 'green' if add_val < threshold else 'orange' if add_val < threshold*2.5 else 'red'
        axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_add_per_class(results, class_dict):
    """
    Bar plot of mean ADD per class.
    Args:
        results: dict with 'obj_ids', 'add_values'
        class_dict: dict mapping obj_id to class info (e.g., Config.LINEMOD_OBJECTS)
    """
    import numpy as np
    obj_ids = np.array(results['obj_ids'])
    add_values = np.array(results['add_values'])
    class_names = []
    mean_adds = []
    for obj_id, obj_name in class_dict.items():
        mask = obj_ids == obj_id
        if np.sum(mask) == 0:
            continue
        class_names.append(f"{obj_id:02d} - {obj_name.get('name')}")
        mean_adds.append(add_values[mask].mean())
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, mean_adds, color='skyblue')
    plt.ylabel('Media ADD')
    plt.xlabel('Classe')
    plt.title('Media ADD per Classe (LineMOD)')
    plt.xticks(rotation=45, ha='right')
    for bar, value in zip(bars, mean_adds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_pinhole_error_per_class(results_pinhole, class_dict):
    """
    Bar plot of mean pinhole translation error per class.
    Args:
        results_pinhole: dict with 'obj_ids', 'pinhole_errors'
        class_dict: dict mapping obj_id to class info (e.g., Config.LINEMOD_OBJECTS)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    obj_ids = np.array(results_pinhole['obj_ids'])
    pinhole_errors = np.array(results_pinhole['pinhole_errors'])
    class_names = []
    mean_pinhole = []
    for obj_id, obj_name in class_dict.items():
        mask = obj_ids == obj_id
        if np.sum(mask) == 0:
            continue
        class_names.append(f"{obj_id:02d} - {obj_name.get('name')}")
        mean_pinhole.append(pinhole_errors[mask].mean())
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, mean_pinhole, color='salmon')
    plt.ylabel('Err. Pinhole medio (mm)')
    plt.xlabel('Classe')
    plt.title('Errore medio Pinhole per Classe (LineMOD)')
    plt.xticks(rotation=45, ha='right')
    for bar, value in zip(bars, mean_pinhole):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()