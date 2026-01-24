"""Visualization utilities for 6D Pose Estimation training and evaluation."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize_image(img_tensor):
    """Denormalize ImageNet-normalized tensor to [0, 1] range."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def _get_add_data(results, use_6d=False):
    """Extract ADD values and correctness from results dict."""
    if use_6d and 'add_values_6d' in results and results['add_values_6d'] is not None:
        return (np.array(results['add_values_6d']), 
                np.array(results['is_correct_6d']), 
                'ADD 6D (mm)')
    return (np.array(results['add_values']), 
            np.array(results['is_correct']), 
            'ADD (mm)')


def _format_class_name(obj_id, obj_info):
    """Format class name consistently."""
    return f"{obj_id:02d} - {obj_info.get('name')}"


def calc_add_accuracy_per_class(results, linemod_objects, use_6d=False):
    """Calculate mean ADD and accuracy per class and globally.
    
    Returns:
        tuple: (per_class_data, global_add, global_accuracy)
    """
    obj_ids = np.array(results['obj_ids'])
    add_values, is_correct, add_label = _get_add_data(results, use_6d)
    
    data = []
    for obj_id, obj_info in linemod_objects.items():
        mask = obj_ids == obj_id
        if not mask.any():
            continue
        data.append({
            'Class': _format_class_name(obj_id, obj_info),
            add_label: f"{add_values[mask].mean():.2f}",
            'Accuracy (%)': f"{is_correct[mask].mean() * 100:.1f}"
        })
    
    return data, add_values.mean(), is_correct.mean() * 100


def calc_pinhole_error_per_class(results, linemod_objects):
    """Calculate mean pinhole error per class and globally.
    
    Returns:
        tuple: (per_class_data, global_error)
    """
    obj_ids = np.array(results['obj_ids'])
    errors = np.array(results['pinhole_errors'])
    
    data = []
    for obj_id, obj_info in linemod_objects.items():
        mask = obj_ids == obj_id
        if not mask.any():
            continue
        data.append({
            'Class': _format_class_name(obj_id, obj_info),
            'Pinhole Error (mm)': f"{errors[mask].mean():.2f}"
        })
    
    return data, errors.mean()


def calc_combined_results_per_class(results_add, results_pinhole, linemod_objects):
    """Combine ADD and pinhole errors per class.
    
    Returns:
        tuple: (per_class_data, global_add, global_accuracy, global_pinhole_error)
    """
    obj_ids_add = np.array(results_add['obj_ids'])
    obj_ids_pinhole = np.array(results_pinhole['obj_ids'])
    
    add_values, is_correct, add_label = _get_add_data(results_add, use_6d=True)
    pinhole_errors = np.array(results_pinhole['pinhole_errors'])
    
    data = []
    for obj_id, obj_info in linemod_objects.items():
        mask_add = obj_ids_add == obj_id
        if not mask_add.any():
            continue
        
        mask_pinhole = obj_ids_pinhole == obj_id
        mean_pinhole = pinhole_errors[mask_pinhole].mean() if mask_pinhole.any() else 0.0
        
        data.append({
            'Class': _format_class_name(obj_id, obj_info),
            add_label: f"{add_values[mask_add].mean():.2f}",
            'Accuracy (%)': f"{is_correct[mask_add].mean() * 100:.1f}",
            'Pinhole Error (mm)': f"{mean_pinhole:.2f}"
        })
    
    return data, add_values.mean(), is_correct.mean() * 100, pinhole_errors.mean()


def show_pose_samples(batch, n=4):
    """Display pose dataset samples with quaternions."""
    images = batch.get('rgb', batch.get('rgb_crop'))
    quaternions = batch['quaternion']
    obj_ids = batch['obj_id']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(n, len(images))):
        img = _denormalize_image(images[i])
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


def plot_training_validation_loss_from_csv(results_path):
    """Plot training and validation loss curves from CSV file."""
    if not os.path.exists(results_path):
        print(f"File not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    plt.figure(figsize=(8, 5))
    
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


def show_pose_samples_with_add(images, gt_quaternions, pred_quaternions, 
                                 obj_ids, add_values, threshold=0.02):
    """Display samples with ground truth, predictions, and ADD error."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for i in range(min(4, len(images))):
        img = _denormalize_image(images[i])
        gt_q = gt_quaternions[i].cpu().numpy()
        pred_q = pred_quaternions[i].cpu().numpy()
        obj_id = obj_ids[i].item() if hasattr(obj_ids[i], 'item') else obj_ids[i]
        add_val = add_values[i]
        
        axes[i].imshow(img)
        title = (
            f"Object {obj_id:02d}\n"
            f"GT: [{gt_q[0]:.2f}, {gt_q[1]:.2f}, {gt_q[2]:.2f}, {gt_q[3]:.2f}]\n"
            f"Pred: [{pred_q[0]:.2f}, {pred_q[1]:.2f}, {pred_q[2]:.2f}, {pred_q[3]:.2f}]\n"
            f"ADD: {add_val:.2f}"
        )
        
        if add_val < threshold:
            color = 'green'
        elif add_val < threshold * 2.5:
            color = 'orange'
        else:
            color = 'red'
        
        axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def _plot_per_class_metric(results, class_dict, metric_key, ylabel, title, color='skyblue'):
    """Generic function to plot per-class metrics as bar chart."""
    obj_ids = np.array(results['obj_ids'])
    metric_values = np.array(results[metric_key])
    
    class_names = []
    mean_values = []
    
    for obj_id, obj_info in class_dict.items():
        mask = obj_ids == obj_id
        if not mask.any():
            continue
        class_names.append(_format_class_name(obj_id, obj_info))
        mean_values.append(metric_values[mask].mean())
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, mean_values, color=color)
    plt.ylabel(ylabel)
    plt.xlabel('Class')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    
    for bar, value in zip(bars, mean_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_add_per_class(results, class_dict):
    """Plot mean ADD per class."""
    _plot_per_class_metric(
        results, class_dict, 'add_values',
        ylabel='Mean ADD (mm)',
        title='Mean ADD per Class (LineMOD)',
        color='skyblue'
    )


def plot_pinhole_error_per_class(results, class_dict):
    """Plot mean pinhole error per class."""
    _plot_per_class_metric(
        results, class_dict, 'pinhole_errors',
        ylabel='Mean Pinhole Error (mm)',
        title='Mean Pinhole Error per Class (LineMOD)',
        color='salmon'
    )