"""
YOLO11 Object Detection Wrapper

This module provides a wrapper around Ultralytics YOLO11 for object detection.
"""

import torch
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import cv2

# Import Config for device management
try:
    from config import Config
    DEFAULT_DEVICE = Config.DEVICE
except ImportError:
    DEFAULT_DEVICE = 'cpu'


class YOLODetector:
    """
    YOLO11 Object Detection Wrapper.
    
    This class wraps the Ultralytics YOLO11 model for object detection.
    It serves as the baseline model for 6D pose estimation.
    
    Args:
        model_name (str): YOLO11 model variant ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x')
        pretrained (bool): Whether to use pretrained weights
        num_classes (int): Number of object classes in your dataset
        device (str): Device to run the model on ('cpu', 'cuda', 'mps')
    """
    
    def __init__(
        self,
        model_name: str = 'yolo11n',
        pretrained: bool = True,
        num_classes: int = 13,  # LineMOD has 13 objects
        device: Optional[str] = None,
        weights_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device if device is not None else DEFAULT_DEVICE
        
        # Set weights directory (usa Config.PRETRAINED_DIR per evitare cluttering)
        if weights_dir is None:
            weights_dir = Config.PRETRAINED_DIR
            weights_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = Path(weights_dir)
        
        # Set Ultralytics settings to use our custom weights directory
        # This prevents downloading to the current directory
        from ultralytics.utils import SETTINGS
        original_weights_dir = SETTINGS.get('weights_dir', None)
        
        # Initialize yolo model
        # Check if model_name is a path to a custom weights file
        model_path = Path(model_name)
        is_custom_path = (model_path.suffix == '.pt') and (model_path.exists() or model_path.is_absolute())
        
        if is_custom_path:
            # Loading custom weights (e.g., fine-tuned model)
            print(f"✅ Loading custom weights from: {model_path}")
            self.model = YOLO(str(model_path))
        elif pretrained:
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
            
            # Modify detection head if using different number of classes
            if num_classes != 80:
                print(f"⚠️  COCO pretrained weights have 80 classes, but you have {num_classes} classes")
                print(f"🔄 Modifying detection head...")
                self._modify_detection_head(num_classes)
        else:
            # Load architecture only (for training from scratch)
            self.model = YOLO(f'{model_name}.yaml')
            print(f"✅ Initialized {model_name} architecture (no pretrained weights)")
        
        # Move model to device
        self.model.to(self.device)
    
    def _modify_detection_head(self, new_num_classes: int):
        """
        Modify the detection head to match the number of classes.
        
        Args:
            new_num_classes (int): Number of classes in your dataset
        """
        import torch.nn as nn
        from ultralytics.nn.modules import Detect
        
        model = self.model.model
        
        # Find the Detect module
        for i, module in enumerate(model.model):
            if isinstance(module, Detect):
                old_nc = module.nc
                module.nc = new_num_classes
                
                print(f"\n🔍 Inspecting cv3 structure:")
                for j, conv_block in enumerate(module.cv3):
                    print(f"   cv3[{j}]: {type(conv_block).__name__}")
                    print(f"   Structure: {conv_block}")
                
                # YOLO11 cv3 structure is typically a Conv module wrapper
                # We need to replace the final conv layer
                for j in range(len(module.cv3)):
                    # Get the old module to extract input channels
                    old_module = module.cv3[j]
                    
                    # YOLO uses a custom Conv wrapper, access the actual conv layer
                    if hasattr(old_module, 'conv'):
                        in_channels = old_module.conv.in_channels
                    elif hasattr(old_module, 'weight'):
                        # It's already a Conv2d
                        in_channels = old_module.in_channels
                    else:
                        # Navigate through children
                        for child in old_module.modules():
                            if isinstance(child, nn.Conv2d):
                                in_channels = child.in_channels
                                break
                    
                    print(f"   cv3[{j}]: {in_channels} → {new_num_classes} channels")
                    
                    # Replace with new Conv2d
                    module.cv3[j] = nn.Conv2d(in_channels, new_num_classes, kernel_size=1, bias=True)
                    
                    # Initialize weights
                    nn.init.normal_(module.cv3[j].weight, mean=0.0, std=0.01)
                    nn.init.constant_(module.cv3[j].bias, 0.0)
                
                print(f"\n   ✅ Modified detection head: {old_nc} → {new_num_classes} classes")
                print(f"   🔧 Reinitialized {len(module.cv3)} cv3 classification layers")
                
                break
    
    def freeze_backbone(self, freeze_until_layer: int = 10):
        """
        Freeze backbone layers for fine-tuning.
        Only the detection head will be trainable.
        
        Args:
            freeze_until_layer (int): Freeze layers 0 to freeze_until_layer-1
                                     YOLO structure: layers 0-9 are backbone,
                                     layers 10+ are neck/head
        
        Returns:
            dict: Parameter statistics (total, frozen, trainable)
        """
        print(f"🔒 Freezing backbone until layer {freeze_until_layer}...")
        
        # Access the underlying model
        model = self.model.model
        
        # Freeze parameters by layer name
        # YOLO structure: model.0, model.1, ..., model.22
        # Backbone is typically layers 0-9
        for name, param in model.named_parameters():
            # Extract layer number from parameter name (e.g., "model.3.conv.weight" -> 3)
            if name.startswith('model.'):
                parts = name.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_num = int(parts[1])
                    if layer_num < freeze_until_layer:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params
        
        print(f"\n📊 Parameter Statistics:")
        print(f"   Total: {total_params:,}")
        print(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"\n✅ Backbone frozen! Only detection head will be trained.")
        
        return {
            'total': total_params,
            'frozen': frozen_params,
            'trainable': trainable_params
        }
        
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
            data_yaml (str): Path to data.yaml configuration file
            epochs (int): Number of training epochs (default: Config.YOLO_EPOCHS)
            imgsz (int): Input image size (default: Config.YOLO_IMG_SIZE)
            batch_size (int): Batch size (default: Config.YOLO_BATCH_SIZE)
            lr0 (float): Initial learning rate (default: Config.YOLO_LR_INITIAL)
            lrf (float): Final learning rate (default: Config.YOLO_LR_FINAL)
            project (str): Project directory for saving results (default: Config.CHECKPOINT_DIR/yolo)
            name (str): Name of the training run
            **kwargs: Additional arguments passed to YOLO.train()
        
        Returns:
            Results object containing training metrics
        """
        # Use Config defaults if not specified
        epochs = epochs if epochs is not None else Config.YOLO_EPOCHS
        imgsz = imgsz if imgsz is not None else Config.YOLO_IMG_SIZE
        batch_size = batch_size if batch_size is not None else Config.YOLO_BATCH_SIZE
        lr0 = lr0 if lr0 is not None else Config.YOLO_LR_INITIAL
        lrf = lrf if lrf is not None else Config.YOLO_LR_FINAL
        project = project if project is not None else str(Config.CHECKPOINT_DIR / 'yolo')
        
        print(f"\n🚂 Starting YOLO training...")
        print(f"   Model: {self.model_name}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch_size}")
        print(f"   LR (initial → final): {lr0} → {lrf}")
        print(f"   Optimizer: {Config.YOLO_OPTIMIZER}")
        print(f"   Device: {self.device}")
        print("")
        
        # Set default training parameters optimized for fine-tuning
        default_params = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'lr0': lr0,
            'lrf': lrf,
            'project': project,
            'name': name,
            'device': self.device,
            
            # Warmup strategy (from Config)
            'warmup_epochs': Config.YOLO_WARMUP_EPOCHS,
            'warmup_momentum': Config.YOLO_WARMUP_MOMENTUM,
            'warmup_bias_lr': Config.YOLO_WARMUP_BIAS_LR,
            
            # Scheduler
            'cos_lr': Config.YOLO_COS_LR,
            
            # Optimizer
            'optimizer': Config.YOLO_OPTIMIZER,
            'momentum': Config.YOLO_MOMENTUM,
            'weight_decay': Config.YOLO_WEIGHT_DECAY,
            
            # Early stopping
            'patience': Config.YOLO_PATIENCE,
        }
        
        # Merge with user-provided kwargs (user kwargs override defaults)
        default_params.update(kwargs)
        
        results = self.model.train(**default_params)
        
        print(f"✅ Training completed!")
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
            source: Input source (image path, folder, video, etc.)
            conf (float): Confidence threshold (default: Config.YOLO_CONF_THRESHOLD)
            iou (float): IoU threshold for NMS (default: Config.YOLO_IOU_THRESHOLD)
            imgsz (int): Input image size (default: Config.YOLO_IMG_SIZE)
            save (bool): Whether to save results
            **kwargs: Additional arguments passed to YOLO.predict()
        
        Returns:
            List of Results objects
        """
        # Use Config defaults if not specified
        conf = conf if conf is not None else Config.YOLO_CONF_THRESHOLD
        iou = iou if iou is not None else Config.YOLO_IOU_THRESHOLD
        imgsz = imgsz if imgsz is not None else Config.YOLO_IMG_SIZE
        
        # 🚀 WORKAROUND: Force CPU for NMS if CUDA to avoid torchvision incompatibility
        # Some torchvision builds don't have CUDA NMS kernel compiled
        nms_device = 'cpu' if self.device == 'cuda' else self.device
        
        try:
            results = self.model.predict(
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
                print(f"⚠️  torchvision NMS not available for {self.device}, falling back to CPU NMS")
                # Fallback: move model to CPU, predict, then move back
                self.model.to('cpu')
                results = self.model.predict(
                    source=source,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    save=save,
                    device='cpu',
                    **kwargs
                )
                self.model.to(self.device)
            else:
                raise
        
        return results
    
    def detect_objects(
        self,
        image: np.ndarray,
        conf_threshold: float = None
    ) -> List[Dict]:
        """
        Detect objects in a single image and return structured results.
        
        Args:
            image (np.ndarray): Input image (H, W, 3) in RGB format
            conf_threshold (float): Confidence threshold (default: Config.YOLO_CONF_THRESHOLD)
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        # Use Config default if not specified
        conf_threshold = conf_threshold if conf_threshold is not None else Config.YOLO_CONF_THRESHOLD
        
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
        batch_size: int = None,
        imgsz: int = None,
        conf: float = None,
        iou: float = None,
        **kwargs
    ):
        """
        Validate the model on validation dataset.
        
        Args:
            data_yaml (str): Path to data.yaml configuration file
            batch_size (int): Batch size (default: Config.YOLO_BATCH_SIZE)
            imgsz (int): Input image size (default: Config.YOLO_IMG_SIZE)
            conf (float): Confidence threshold for detections (default: Config.YOLO_CONF_THRESHOLD)
                         ⚠️ CRITICAL: Higher values (e.g., 0.5) drastically reduce NMS overhead
            iou (float): IoU threshold for NMS (default: Config.YOLO_IOU_THRESHOLD)
            **kwargs: Additional arguments passed to YOLO.val()
        
        Returns:
            Validation metrics
        
        Note:
            Using conf=0.5 can speed up validation by 5-10x by reducing candidate detections.
        """
        # Use Config defaults if not specified
        batch_size = batch_size if batch_size is not None else Config.YOLO_BATCH_SIZE
        imgsz = imgsz if imgsz is not None else Config.YOLO_IMG_SIZE
        conf = conf if conf is not None else Config.YOLO_CONF_THRESHOLD
        iou = iou if iou is not None else Config.YOLO_IOU_THRESHOLD
        
        print(f"\n📊 Validating YOLO model...")
        
        metrics = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
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
        # YOLO automatically saves checkpoints during training
        # This method is for manual saving if needed
        print(f"💾 Model checkpoints are saved during training")
        print(f"   Check: checkpoints/yolo/{self.model_name}/weights/")
    
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
