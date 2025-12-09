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
    
    # ==================== Dataset ====================
    NUM_CLASSES = 13  # LineMOD has 13 objects
    
    # LineMOD objects unified mapping (single source of truth)
    # Folders 03 and 07 are missing in LineMOD dataset
    LINEMOD_OBJECTS = {
        1: {'name': 'ape', 'yolo_class': 0, 'symmetric': False},
        2: {'name': 'benchvise', 'yolo_class': 1, 'symmetric': False},
        4: {'name': 'camera', 'yolo_class': 2, 'symmetric': False},
        5: {'name': 'can', 'yolo_class': 3, 'symmetric': False},
        6: {'name': 'cat', 'yolo_class': 4, 'symmetric': False},
        8: {'name': 'driller', 'yolo_class': 5, 'symmetric': False},
        9: {'name': 'duck', 'yolo_class': 6, 'symmetric': False},
        10: {'name': 'eggbox', 'yolo_class': 7, 'symmetric': True},
        11: {'name': 'glue', 'yolo_class': 8, 'symmetric': True},
        12: {'name': 'holepuncher', 'yolo_class': 9, 'symmetric': False},
        13: {'name': 'iron', 'yolo_class': 10, 'symmetric': False},
        14: {'name': 'lamp', 'yolo_class': 11, 'symmetric': False},
        15: {'name': 'phone', 'yolo_class': 12, 'symmetric': False}
    }
    
    # Derived properties (auto-generated from LINEMOD_OBJECTS)
    CLASS_NAMES = [obj['name'] for obj in sorted(LINEMOD_OBJECTS.values(), key=lambda x: x['yolo_class'])]
    FOLDER_ID_TO_CLASS_ID = {fid: obj['yolo_class'] for fid, obj in LINEMOD_OBJECTS.items()}
    OBJ_ID_TO_NAME = {fid: obj['name'] for fid, obj in LINEMOD_OBJECTS.items()}
    SYMMETRIC_OBJECTS = [obj['yolo_class'] for obj in LINEMOD_OBJECTS.values() if obj['symmetric']]
    
    # ==================== YOLO Model ====================
    YOLO_MODEL = 'yolo11n'  # Options: yolo11n, yolo11s, yolo11m (11n is nano - smallest and fastest)
    
    # ==================== Adaptive Helpers ====================
    @staticmethod
    def get_optimal_workers():
        """Get optimal worker count based on device and CPU cores."""
        import torch
        import os
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon: 4 workers optimal for unified memory
            return 4
        elif torch.cuda.is_available():
            # CUDA: Can handle more workers
            return min(8, os.cpu_count() // 2) if os.cpu_count() else 4
        else:
            # CPU (including Colab): Conservative
            return 2
    
    @staticmethod
    def should_pin_memory():
        """Determine if pin_memory should be used for DataLoader."""
        import torch
        # Only beneficial for CUDA
        return torch.cuda.is_available()
    
    @staticmethod
    def should_cache_images():
        """Determine if images should be cached based on available RAM."""
        try:
            import psutil
            # Get available RAM in GB
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            # Enable caching if ≥6GB available
            return available_ram_gb >= 6.0
        except ImportError:
            # If psutil not available, assume enough RAM
            return True
    
    # YOLO Fine-tuning parameters (used in notebooks)
    YOLO_FREEZE_UNTIL_LAYER = 10  # Freeze layers 0-9 (backbone), train from 10 onwards (neck/head)
    
    # Device-specific optimizations
    PIN_MEMORY = should_pin_memory()  # pin_memory for DataLoader
    CACHE_IMAGES = should_cache_images()  # Cache images in RAM
    
    # ==================== Device ====================
    # Auto-detect best available device: CUDA > MPS (Apple Silicon) > CPU
    @staticmethod
    def get_device():
        """Get the best available device for training."""
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Test MPS availability
                torch.zeros(1).to('mps')
                return 'mps'
            except Exception:
                # MPS not working properly, fall back to CPU
                return 'cpu'
        else:
            return 'cpu'
    
    DEVICE = get_device()
    NUM_WORKERS = get_optimal_workers()  # Adaptive worker count for dataloaders
    
    # ==================== Logging (used by scripts) ====================
    USE_WANDB = False
    WANDB_PROJECT = '6d-pose-estimation'
    
    # ==================== Pose Estimation ====================
    # Model parameters
    POSE_IMAGE_SIZE = 224  # Input size for ResNet-50
    POSE_BACKBONE = 'resnet50'
    POSE_DROPOUT = 0.3
    
    # Training parameters
    POSE_EPOCHS = 50
    POSE_BATCH_SIZE = 8  # Effective batch size = 8 * 4 (gradient accumulation) = 32
    POSE_LR = 1e-4
    POSE_WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUM_STEPS = 4  # Gradient accumulation for larger effective batch size
    USE_AMP = True  # Use automatic mixed precision (FP16)
    
    # Data augmentation for pose
    POSE_CROP_MARGIN = 0.1  # 10% margin around bbox
    POSE_COLOR_JITTER = True
    
    # Loss weights
    LAMBDA_TRANS = 1.0  # Translation loss weight
    LAMBDA_ROT = 10.0  # Rotation loss weight
    
    # Evaluation
    ADD_THRESHOLD = 0.1  # 10% of object diameter
    # SYMMETRIC_OBJECTS now derived from LINEMOD_OBJECTS (defined above)
    
    # Paths for 3D models
    MODELS_PATH = DATA_ROOT / 'models'
    MODELS_INFO_PATH = MODELS_PATH / 'models_info.yml'


# Create directories if they don't exist
Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
