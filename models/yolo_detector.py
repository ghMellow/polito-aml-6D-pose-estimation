"""
YOLOv8 Object Detection Wrapper

This module provides a wrapper around Ultralytics YOLOv8 for object detection
as a baseline model. This will be extended later to include rotation prediction.
"""

import torch
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import cv2


class YOLODetector:
    """
    YOLOv8 Object Detection Wrapper.
    
    This class wraps the Ultralytics YOLOv8 model for object detection.
    It serves as the baseline model for 6D pose estimation.
    
    Args:
        model_name (str): YOLOv8 model variant ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
        pretrained (bool): Whether to use pretrained weights
        num_classes (int): Number of object classes in your dataset
        device (str): Device to run the model on ('cpu', 'cuda', 'mps')
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n',
        pretrained: bool = True,
        num_classes: int = 13,  # LineMOD has 13 objects
        device: str = 'cpu',
        weights_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        
        # Set weights directory (use checkpoints dir to avoid cluttering project root)
        if weights_dir is None:
            weights_dir = Path(__file__).parent.parent / 'checkpoints' / 'pretrained'
            weights_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = Path(weights_dir)
        
        # Set Ultralytics settings to use our custom weights directory
        # This prevents downloading to the current directory
        from ultralytics.utils import SETTINGS
        original_weights_dir = SETTINGS.get('weights_dir', None)
        
        # Initialize YOLOv8 model
        if pretrained:
            # Check if weights already exist locally
            weights_path = self.weights_dir / f'{model_name}.pt'
            
            if weights_path.exists():
                print(f"✅ Loading pretrained {model_name} from cache: {weights_path}")
                self.model = YOLO(str(weights_path))
            else:
                # Temporarily change working directory to weights_dir for download
                original_cwd = os.getcwd()
                try:
                    os.chdir(self.weights_dir)
                    print(f"📥 Downloading pretrained {model_name} model to: {self.weights_dir}")
                    self.model = YOLO(f'{model_name}.pt')
                    print(f"💾 Weights saved to: {self.weights_dir / f'{model_name}.pt'}")
                finally:
                    os.chdir(original_cwd)
            
            # Warning if using pretrained COCO weights with different num_classes
            if num_classes != 80:
                print(f"⚠️  WARNING: Using COCO pretrained weights (80 classes) for {num_classes} classes")
                print(f"   You'll need to fine-tune or retrain for your custom dataset")
        else:
            # Load architecture only (for training from scratch)
            self.model = YOLO(f'{model_name}.yaml')
            print(f"✅ Initialized {model_name} architecture (no pretrained weights)")
        
        # Move model to device
        self.model.to(device)
        
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        project: str = './runs/detect',
        name: str = 'yolo_baseline',
        **kwargs
    ):
        """
        Train the YOLO model.
        
        Args:
            data_yaml (str): Path to data.yaml configuration file
            epochs (int): Number of training epochs
            imgsz (int): Input image size
            batch_size (int): Batch size
            project (str): Project directory for saving results
            name (str): Name of the training run
            **kwargs: Additional arguments passed to YOLO.train()
        
        Returns:
            Results object containing training metrics
        """
        print(f"\n🚂 Starting YOLOv8 training...")
        print(f"   Model: {self.model_name}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project,
            name=name,
            device=self.device,
            **kwargs
        )
        
        print(f"✅ Training completed!")
        return results
    
    def predict(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        save: bool = False,
        **kwargs
    ):
        """
        Run inference on images/video.
        
        Args:
            source: Input source (image path, folder, video, etc.)
            conf (float): Confidence threshold
            iou (float): IoU threshold for NMS
            imgsz (int): Input image size
            save (bool): Whether to save results
            **kwargs: Additional arguments passed to YOLO.predict()
        
        Returns:
            List of Results objects
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            device=self.device,
            **kwargs
        )
        
        return results
    
    def detect_objects(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25
    ) -> List[Dict]:
        """
        Detect objects in a single image and return structured results.
        
        Args:
            image (np.ndarray): Input image (H, W, 3) in RGB format
            conf_threshold (float): Confidence threshold
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        # Run prediction
        results = self.predict(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes.xyxy[i].cpu().numpy(),  # [x1, y1, x2, y2]
                    'confidence': float(boxes.conf[i].cpu().numpy()),
                    'class_id': int(boxes.cls[i].cpu().numpy()),
                    'class_name': result.names[int(boxes.cls[i])]
                }
                detections.append(detection)
        
        return detections
    
    def validate(
        self,
        data_yaml: str,
        batch_size: int = 16,
        imgsz: int = 640,
        **kwargs
    ):
        """
        Validate the model on validation dataset.
        
        Args:
            data_yaml (str): Path to data.yaml configuration file
            batch_size (int): Batch size
            imgsz (int): Input image size
            **kwargs: Additional arguments passed to YOLO.val()
        
        Returns:
            Validation metrics
        """
        print(f"\n📊 Validating YOLOv8 model...")
        
        metrics = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            **kwargs
        )
        
        print(f"✅ Validation completed!")
        return metrics
    
    def export(
        self,
        format: str = 'onnx',
        **kwargs
    ):
        """
        Export model to different formats.
        
        Args:
            format (str): Export format ('onnx', 'torchscript', 'coreml', etc.)
            **kwargs: Additional arguments passed to YOLO.export()
        
        Returns:
            Path to exported model
        """
        print(f"\n📦 Exporting model to {format}...")
        
        path = self.model.export(format=format, **kwargs)
        
        print(f"✅ Model exported to: {path}")
        return path
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        self.model = YOLO(checkpoint_path)
        self.model.to(self.device)
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
    
    def save_checkpoint(self, save_path: str):
        """
        Save model checkpoint.
        
        Args:
            save_path (str): Path to save checkpoint
        """
        # YOLOv8 automatically saves checkpoints during training
        # This method is for manual saving if needed
        print(f"💾 Model checkpoints are saved during training")
        print(f"   Check: runs/detect/{self.model_name}/weights/")
    
    @property
    def model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'device': self.device,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def __repr__(self):
        return f"YOLODetector(model={self.model_name}, classes={self.num_classes}, device={self.device})"


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize detections on image.
    
    Args:
        image (np.ndarray): Input image (H, W, 3)
        detections (List[Dict]): List of detections from detect_objects()
        save_path (Optional[str]): Path to save visualization
    
    Returns:
        Image with drawn detections
    """
    vis_image = image.copy()
    
    for det in detections:
        # Extract bbox coordinates
        x1, y1, x2, y2 = det['bbox'].astype(int)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(
            vis_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image
