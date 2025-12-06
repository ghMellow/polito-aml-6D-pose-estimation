"""
Test Utilities for YOLO Notebooks

Reusable helper functions for testing and visualizing YOLO detections.
Eliminates code duplication across notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def test_yolo_detection(
    detector,
    image_path: str,
    conf_threshold: float = 0.25
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Test YOLO detector on a single image.
    
    Args:
        detector: YOLODetector instance
        image_path: Path to image file
        conf_threshold: Confidence threshold for detections
        
    Returns:
        tuple: (image_array, detections_list)
    """
    from models.yolo_detector import visualize_detections
    
    # Load image
    image_path = Path(image_path)
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Run detection
    detections = detector.detect_objects(image, conf_threshold=conf_threshold)
    
    # Print results
    print(f"🔍 Detected {len(detections)} objects in {image_path.name}:")
    for i, det in enumerate(detections):
        print(f"   {i+1}. {det['class_name']}: {det['confidence']:.2f}")
    
    return image, detections


def visualize_comparison(
    image: np.ndarray,
    detections_before: List[Dict],
    detections_after: List[Dict],
    title1: str = "Before",
    title2: str = "After",
    figsize: Tuple[int, int] = (16, 8)
) -> None:
    """
    Compare two detection results side-by-side.
    
    Args:
        image: Input image array (H, W, 3)
        detections_before: List of detections for left plot
        detections_after: List of detections for right plot
        title1: Title for left plot
        title2: Title for right plot
        figsize: Figure size (width, height)
    """
    from models.yolo_detector import visualize_detections
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Before
    vis_before = visualize_detections(image.copy(), detections_before)
    axes[0].imshow(vis_before)
    axes[0].set_title(f"{title1}\n{len(detections_before)} detections", fontsize=14)
    axes[0].axis('off')
    
    # After
    vis_after = visualize_detections(image.copy(), detections_after)
    axes[1].imshow(vis_after)
    axes[1].set_title(f"{title2}\n{len(detections_after)} detections", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_image_with_detections(
    image: np.ndarray,
    detections: List[Dict],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """
    Display image with detection overlays.
    
    Args:
        image: Input image array (H, W, 3)
        detections: List of detections
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    from models.yolo_detector import visualize_detections
    
    vis_img = visualize_detections(image.copy(), detections)
    
    plt.figure(figsize=figsize)
    plt.imshow(vis_img)
    
    if title is None:
        title = f"Detections ({len(detections)} objects)"
    plt.title(title)
    plt.axis('off')
    plt.show()


def print_model_summary(detector, show_layers: int = 20) -> None:
    """
    Print model summary with trainable parameter statistics.
    
    Args:
        detector: YOLODetector instance
        show_layers: Number of layers to display
    """
    print("📊 MODEL SUMMARY")
    print("=" * 80)
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    
    for name, param in detector.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\n📈 Parameter Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    print(f"   Trainable percentage: {100 * trainable_params / total_params:.1f}%")
    
    # Show layer details
    if show_layers > 0:
        print(f"\n🔍 First {show_layers} layers:")
        print("-" * 80)
        for i, (name, param) in enumerate(list(detector.model.named_parameters())[:show_layers]):
            status = "✅ TRAINABLE" if param.requires_grad else "❄️  FROZEN"
            print(f"   {i:2d}. {name:50s} {status}  Shape: {tuple(param.shape)}")


def compare_model_stats(detector_before, detector_after) -> None:
    """
    Compare parameter statistics between two model states.
    
    Args:
        detector_before: YOLODetector before freezing
        detector_after: YOLODetector after freezing
    """
    def count_params(detector):
        total = 0
        trainable = 0
        for param in detector.model.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        return total, trainable
    
    total_before, trainable_before = count_params(detector_before)
    total_after, trainable_after = count_params(detector_after)
    
    print("📊 MODEL COMPARISON")
    print("=" * 80)
    print(f"\nBEFORE FREEZING:")
    print(f"   Total: {total_before:,}")
    print(f"   Trainable: {trainable_before:,} ({100 * trainable_before / total_before:.1f}%)")
    
    print(f"\nAFTER FREEZING:")
    print(f"   Total: {total_after:,}")
    print(f"   Trainable: {trainable_after:,} ({100 * trainable_after / total_after:.1f}%)")
    print(f"   Frozen: {total_after - trainable_after:,} ({100 * (total_after - trainable_after) / total_after:.1f}%)")
    
    print(f"\n💡 Reduction in trainable parameters: {trainable_before - trainable_after:,}")
    print(f"   Speed improvement: ~{trainable_before / trainable_after:.1f}x faster training")
