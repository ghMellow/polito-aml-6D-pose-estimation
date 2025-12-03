"""
Configuration File

Central configuration for training, evaluation, and inference.
"""

import os
from pathlib import Path


class Config:
    """Base configuration class."""
    
    # ==================== Paths ====================
    PROJECT_ROOT = Path(__file__).parent
    DATA_ROOT = PROJECT_ROOT / 'data' / 'Linemod_preprocessed'
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    RUNS_DIR = PROJECT_ROOT / 'runs'
    
    # ==================== Dataset ====================
    NUM_CLASSES = 13  # LineMOD has 13 objects
    TRAIN_RATIO = 0.8
    RANDOM_SEED = 42
    
    # Object class names (LineMOD)
    CLASS_NAMES = [
        'ape', 'benchvise', 'camera', 'can', 'cat',
        'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
        'iron', 'lamp', 'phone'
    ]
    
    # ==================== YOLO Model ====================
    YOLO_MODEL = 'yolov8n'  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    PRETRAINED = True
    
    # ==================== Training ====================
    EPOCHS = 100
    BATCH_SIZE = 16
    IMAGE_SIZE = 640
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.937
    
    # Data augmentation
    AUGMENT = True
    MOSAIC = 1.0
    MIXUP = 0.0
    
    # ==================== Inference ====================
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    # ==================== Device ====================
    DEVICE = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'
    NUM_WORKERS = 4
    
    # ==================== Logging ====================
    USE_WANDB = False
    WANDB_PROJECT = '6d-pose-estimation'
    WANDB_ENTITY = None
    
    SAVE_PERIOD = 10  # Save checkpoint every N epochs
    VERBOSE = True
    
    # ==================== Evaluation ====================
    EVAL_BATCH_SIZE = 32
    
    @classmethod
    def get_model_config(cls):
        """Get model-specific configuration."""
        return {
            'model_name': cls.YOLO_MODEL,
            'pretrained': cls.PRETRAINED,
            'num_classes': cls.NUM_CLASSES,
            'device': cls.DEVICE
        }
    
    @classmethod
    def get_train_config(cls):
        """Get training configuration."""
        return {
            'epochs': cls.EPOCHS,
            'batch_size': cls.BATCH_SIZE,
            'imgsz': cls.IMAGE_SIZE,
            'lr0': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'momentum': cls.MOMENTUM,
            'augment': cls.AUGMENT,
            'mosaic': cls.MOSAIC,
            'mixup': cls.MIXUP,
            'workers': cls.NUM_WORKERS,
            'device': cls.DEVICE,
            'verbose': cls.VERBOSE
        }
    
    @classmethod
    def get_inference_config(cls):
        """Get inference configuration."""
        return {
            'conf': cls.CONF_THRESHOLD,
            'iou': cls.IOU_THRESHOLD,
            'imgsz': cls.IMAGE_SIZE,
            'device': cls.DEVICE
        }


# Create directories if they don't exist
Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
Config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
