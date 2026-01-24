"""
YOLO11 Object Detection Wrapper

This module provides a wrapper around Ultralytics YOLO11 for object detection.
"""

import torch
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Optional
import cv2

try:
    from config import Config
    DEFAULT_DEVICE = Config.DEVICE
except ImportError:
    DEFAULT_DEVICE = 'cpu'

COCO_NUM_CLASSES = 80


class YOLODetector:
    """YOLO11 Object Detection Wrapper for 6D pose estimation."""
    
    def __init__(
        self,
        model_name: str = 'yolo11n',
        pretrained: bool = True,
        num_classes: int = 13,
        device: Optional[str] = None,
        weights_dir: Optional[str] = None
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLO11 variant ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x')
            pretrained: Use pretrained weights if True
            num_classes: Number of object classes (LineMOD has 13)
            device: Device to run on ('cpu', 'cuda', 'mps')
            weights_dir: Directory for model weights
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device if device is not None else DEFAULT_DEVICE
        
        if weights_dir is None:
            weights_dir = Config.PRETRAINED_DIR
            weights_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = Path(weights_dir)
        
        self.model = self._load_model(model_name, pretrained, num_classes)
        self.model.to(self.device)
    
    def _load_model(self, model_name: str, pretrained: bool, num_classes: int):
        """Load YOLO model with appropriate weights and configuration."""
        model_path = Path(model_name)
        
        if model_path.suffix == '.pt' and (model_path.exists() or model_path.is_absolute()):
            print(f"Loading custom weights: {model_path}")
            return YOLO(str(model_path))
        
        if pretrained:
            return self._load_pretrained(model_name, num_classes)
        else:
            print(f"Initializing {model_name} architecture (no pretrained weights)")
            return YOLO(f'{model_name}.yaml')
    
    def _load_pretrained(self, model_name: str, num_classes: int):
        """Load pretrained YOLO model, downloading if necessary."""
        weights_path = self.weights_dir / f'{model_name}.pt'
        
        if weights_path.exists():
            print(f"Loading pretrained {model_name} from cache: {weights_path}")
            model = YOLO(str(weights_path))
        else:
            original_cwd = os.getcwd()
            try:
                os.chdir(self.weights_dir)
                print(f"Downloading pretrained {model_name} to: {self.weights_dir}")
                model = YOLO(f'{model_name}.pt')
            finally:
                os.chdir(original_cwd)
        
        if num_classes != COCO_NUM_CLASSES:
            self._modify_detection_head(model, num_classes)
        
        return model
    
    def _modify_detection_head(self, model, new_num_classes: int):
        """Modify detection head to match dataset classes."""
        import torch.nn as nn
        from ultralytics.nn.modules import Detect
        
        for module in model.model.model:
            if isinstance(module, Detect):
                old_nc = module.nc
                module.nc = new_num_classes
                self._replace_cv3_layers(module, new_num_classes)
                print(f"Modified detection head: {old_nc} -> {new_num_classes} classes")
                break
    
    def _replace_cv3_layers(self, detect_module, new_num_classes: int):
        """Replace cv3 classification layers with new output channels."""
        import torch.nn as nn
        
        for j in range(len(detect_module.cv3)):
            old_module = detect_module.cv3[j]
            in_channels = self._get_input_channels(old_module)
            
            detect_module.cv3[j] = nn.Conv2d(
                in_channels, new_num_classes, kernel_size=1, bias=True
            )
            nn.init.normal_(detect_module.cv3[j].weight, mean=0.0, std=0.01)
            nn.init.constant_(detect_module.cv3[j].bias, 0.0)
    
    @staticmethod
    def _get_input_channels(module) -> int:
        """Extract input channels from conv module."""
        import torch.nn as nn
        
        if hasattr(module, 'conv'):
            return module.conv.in_channels
        if hasattr(module, 'in_channels'):
            return module.in_channels
        
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                return child.in_channels
        
        raise ValueError("Could not determine input channels")
    
    def freeze_backbone(self, freeze_until_layer: int = 10) -> Dict:
        """
        Freeze backbone layers for fine-tuning.
        
        Args:
            freeze_until_layer: Freeze layers 0 to freeze_until_layer-1
        
        Returns:
            Dictionary with total, frozen, and trainable parameter counts
        """
        model = self.model.model
        
        for name, param in model.named_parameters():
            if name.startswith('model.'):
                parts = name.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_num = int(parts[1])
                    param.requires_grad = layer_num >= freeze_until_layer
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = trainable + frozen
        
        print(f"Frozen backbone: {frozen:,} params | "
              f"Trainable: {trainable:,} params | Total: {total:,}")
        
        return {'total': total, 'frozen': frozen, 'trainable': trainable}
    
    def train(
        self,
        data_yaml: str,
        epochs: int = None,
        imgsz: int = None,
        batch_size: int = None,
        lr0: float = None,
        lrf: float = None,
        project: str = None,
        name: str = 'yolo_baseline',
        **kwargs
    ):
        """
        Train the YOLO model.
        
        Args:
            data_yaml: Path to data.yaml configuration file
            epochs: Number of epochs (default: Config.YOLO_EPOCHS)
            imgsz: Image size (default: Config.YOLO_IMG_SIZE)
            batch_size: Batch size (default: Config.YOLO_BATCH_SIZE)
            lr0: Initial learning rate (default: Config.YOLO_LR_INITIAL)
            lrf: Final learning rate (default: Config.YOLO_LR_FINAL)
            project: Project directory (default: Config.CHECKPOINT_DIR/yolo)
            name: Training run name
            **kwargs: Additional YOLO.train() arguments
        """
        params = {
            'data': data_yaml,
            'epochs': epochs or Config.YOLO_EPOCHS,
            'imgsz': imgsz or Config.YOLO_IMG_SIZE,
            'batch': batch_size or Config.YOLO_BATCH_SIZE,
            'lr0': lr0 or Config.YOLO_LR_INITIAL,
            'lrf': lrf or Config.YOLO_LR_FINAL,
            'project': project or str(Config.CHECKPOINT_DIR / 'yolo'),
            'name': name,
            'device': self.device,
            'warmup_epochs': Config.YOLO_WARMUP_EPOCHS,
            'warmup_momentum': Config.YOLO_WARMUP_MOMENTUM,
            'warmup_bias_lr': Config.YOLO_WARMUP_BIAS_LR,
            'cos_lr': Config.YOLO_COS_LR,
            'optimizer': Config.YOLO_OPTIMIZER,
            'momentum': Config.YOLO_MOMENTUM,
            'weight_decay': Config.YOLO_WEIGHT_DECAY,
            'patience': Config.YOLO_PATIENCE,
        }
        params.update(kwargs)
        
        print(f"Training {self.model_name} | Epochs: {params['epochs']} | "
              f"LR: {params['lr0']} -> {params['lrf']}")
        
        results = self.model.train(**params)
        return results
    
    def predict(
        self,
        source,
        conf: float = None,
        iou: float = None,
        imgsz: int = None,
        save: bool = False,
        **kwargs
    ):
        """
        Run inference on images/video.
        
        Args:
            source: Input (image path, folder, video, etc.)
            conf: Confidence threshold (default: Config.YOLO_CONF_THRESHOLD)
            iou: IoU threshold for NMS (default: Config.YOLO_IOU_THRESHOLD)
            imgsz: Image size (default: Config.YOLO_IMG_SIZE)
            save: Save results
            **kwargs: Additional predict() arguments
        """
        conf = conf or Config.YOLO_CONF_THRESHOLD
        iou = iou or Config.YOLO_IOU_THRESHOLD
        imgsz = imgsz or Config.YOLO_IMG_SIZE
        
        try:
            return self.model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                save=save,
                device=self.device,
                **kwargs
            )
        except NotImplementedError as e:
            if 'torchvision::nms' in str(e):
                print(f"NMS not available for {self.device}, using CPU")
                self.model.to('cpu')
                results = self.model.predict(
                    source=source, conf=conf, iou=iou, imgsz=imgsz,
                    save=save, device='cpu', **kwargs
                )
                self.model.to(self.device)
                return results
            raise
    
    def detect_objects(
        self,
        image: np.ndarray,
        conf_threshold: float = None
    ) -> List[Dict]:
        """
        Detect objects in image.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            conf_threshold: Confidence threshold
        
        Returns:
            List of detections with bbox, confidence, class_id, class_name
        """
        conf_threshold = conf_threshold or Config.YOLO_CONF_THRESHOLD
        results = self.predict(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            for i in range(len(result.boxes)):
                detections.append({
                    'bbox': result.boxes.xyxy[i].cpu().numpy(),
                    'confidence': float(result.boxes.conf[i].cpu().numpy()),
                    'class_id': int(result.boxes.cls[i].cpu().numpy()),
                    'class_name': result.names[int(result.boxes.cls[i])]
                })
        
        return detections
    
    def validate(
        self,
        data_yaml: str,
        batch_size: int = None,
        imgsz: int = None,
        conf: float = None,
        iou: float = None,
        **kwargs
    ):
        """
        Validate model on validation dataset.
        
        Args:
            data_yaml: Path to data.yaml file
            batch_size: Batch size (default: Config.YOLO_BATCH_SIZE)
            imgsz: Image size (default: Config.YOLO_IMG_SIZE)
            conf: Confidence threshold (default: Config.YOLO_CONF_THRESHOLD)
            iou: IoU threshold (default: Config.YOLO_IOU_THRESHOLD)
            **kwargs: Additional val() arguments
        
        Note:
            Higher conf values (e.g., 0.5) can speed up validation by reducing NMS candidates
        """
        print(f"Validating {self.model_name}...")
        
        return self.model.val(
            data=data_yaml,
            batch=batch_size or Config.YOLO_BATCH_SIZE,
            imgsz=imgsz or Config.YOLO_IMG_SIZE,
            conf=conf or Config.YOLO_CONF_THRESHOLD,
            iou=iou or Config.YOLO_IOU_THRESHOLD,
            device=self.device,
            **kwargs
        )
    
    def export(self, format: str = 'onnx', **kwargs):
        """Export model to different formats."""
        print(f"Exporting model to {format}...")
        path = self.model.export(format=format, **kwargs)
        print(f"Exported to: {path}")
        return path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        self.model = YOLO(checkpoint_path)
        self.model.to(self.device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    @property
    def model_info(self) -> Dict:
        """Return model information dictionary."""
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
    Draw detections on image.
    
    Args:
        image: Input image (H, W, 3)
        detections: List of detections from detect_objects()
        save_path: Optional path to save visualization
    
    Returns:
        Image with drawn bounding boxes and labels
    """
    vis_image = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox'].astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(
            vis_image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image
