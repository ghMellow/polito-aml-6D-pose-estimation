"""
Configuration File

Central configuration for training, evaluation, and inference.
"""

import os
from pathlib import Path


class Config:
    """Base configuration class."""
    
    # ==================== Paths ====================
    # Config è in root/, quindi parent è PROJECT_ROOT
    # resolve():
    # - Garantisce coerenza su Windows/macOS/Linux
    # - Converte il path in assoluto
    # - Elimina . e ..
    # - Risolve i symlink
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATASETS_DIR = PROJECT_ROOT / 'data'
    LINEMOD_ROOT = DATASETS_DIR / 'Linemod_preprocessed'
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    PRETRAINED_DIR = CHECKPOINT_DIR / 'pretrained'  # For pretrained model weights
    
    # ==================== Dataset ====================
    NUM_CLASSES = 13  # LineMOD has 13 objects
    
    # Dataset splits and sampling
    TRAIN_TEST_RATIO = 0.8  # 80% train, 20% test (for random splits)
    RANDOM_SEED = 42  # For reproducibility
    
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
    
    # Detection thresholds
    YOLO_CONF_THRESHOLD = 0.25  # Confidence threshold for detections
    YOLO_IOU_THRESHOLD = 0.45   # IoU threshold for NMS
    
    # Architecture
    YOLO_FREEZE_UNTIL_LAYER = 10  # Freeze layers 0-9 (backbone), train from 10 onwards (neck/head)
    
    # Training hyperparameters
    YOLO_EPOCHS = 10  # Poche epoche per test veloce
    YOLO_BATCH_SIZE = 32
    YOLO_IMG_SIZE = 416
    YOLO_PATIENCE = 10  # Early stopping patience
    
    # Learning rate (ottimizzato per fine-tuning)
    YOLO_LR_INITIAL = 0.01   # Learning rate iniziale (lr0) - 10x più basso di 0.1 per fine-tuning
    YOLO_LR_FINAL = 0.01     # Learning rate finale (lrf)
    
    # Warmup strategy
    YOLO_WARMUP_EPOCHS = 3
    YOLO_WARMUP_MOMENTUM = 0.8
    YOLO_WARMUP_BIAS_LR = 0.1
    
    # Scheduler
    YOLO_COS_LR = True  # Cosine annealing learning rate scheduler
    
    # Optimizer
    YOLO_OPTIMIZER = 'SGD'  # Options: 'SGD', 'Adam', 'AdamW'
    YOLO_MOMENTUM = 0.937
    YOLO_WEIGHT_DECAY = 0.0005
    
    # ==================== Adaptive Helpers ====================
    @staticmethod
    def get_optimal_workers(force_safe_mode=False):
        """
        Get optimal worker count based on device and CPU cores.
        
        Args:
            force_safe_mode: If True, uses 0 workers on MPS (safe for complex data transforms)
        
        Returns:
            int: Number of workers for DataLoader
        """
        import torch
        import os
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon MPS:
            # - Use 0 workers for pose estimation (avoids crashes with rotation/quaternion ops)
            # - Can use 4 workers for simple YOLO data loading
            return 0 if force_safe_mode else 4
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
            # 🚀 OPTIMIZATION: Adaptive cache strategy
            # - ≥8GB: Full cache (all 4,700+ images)
            # - 4-8GB: Partial cache (most frequent 50%)
            # - <4GB: No image cache (metadata only)
            return available_ram_gb >= 4.0
        except ImportError:
            # If psutil not available, assume enough RAM
            return True
    
    @staticmethod
    def get_cache_strategy():
        """
        🚀 OPTIMIZATION: Get adaptive cache strategy based on available RAM.
        
        Returns:
            str: 'full' (cache all), 'partial' (LRU cache 50%), or 'none' (no image cache)
        """
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_ram_gb >= 8.0:
                return 'full'  # ~1.5GB for full dataset
            elif available_ram_gb >= 4.0:
                return 'partial'  # ~750MB for 50% most used
            else:
                return 'none'  # Metadata only (~10MB)
        except ImportError:
            return 'full'
        
    # Adaptive device-specific helper optimizations
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
    NUM_WORKERS = get_optimal_workers()  # Adaptive worker count for YOLO dataloaders
    NUM_WORKERS_POSE = get_optimal_workers(force_safe_mode=True)  # Safe mode for pose (0 on MPS)
    
    # ==================== Logging (used by scripts) ====================
    USE_WANDB = False
    WANDB_PROJECT = '6d-pose-estimation'
    
    # ==================== Pose Estimation ====================
    # Model parameters
    POSE_IMAGE_SIZE = 224  # Input size for ResNet-50
    POSE_BACKBONE = 'resnet50'
    POSE_PRETRAINED = True  # Use ImageNet pretrained weights for ResNet-50
    POSE_FREEZE_BACKBONE = False  # If True, freeze backbone and only train head (faster)
    POSE_DROPOUT = 0.5
    
    # Training parameters
    POSE_EPOCHS = 100
    POSE_BATCH_SIZE = 64 
    ACCUMULATION_STEPS = 2
    POSE_LR = 1e-4
    POSE_WEIGHT_DECAY = 5e-4  # da 1e-4 a 5e-4
    USE_AMP = True  # Use automatic mixed precision (FP16)
    
    # Data augmentation for pose
    POSE_CROP_MARGIN = 0.15  # da 0.1 a 0.15 (più variabilità)
    POSE_COLOR_JITTER = True
    
    # Loss weights
    LAMBDA_TRANS = 1.0  # Translation loss weight
    LAMBDA_ROT = 50.0   # Rotation loss weight (10 -> 50)
    
    # Evaluation
    ADD_THRESHOLD = 0.1  # 10% of object diameter
    # SYMMETRIC_OBJECTS now derived from LINEMOD_OBJECTS (defined above)
    
    # Paths for 3D models
    MODELS_PATH = LINEMOD_ROOT / 'models'
    MODELS_INFO_PATH = MODELS_PATH / 'models_info.yml'


# Create directories if they don't exist
Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
