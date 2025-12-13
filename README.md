# 6D Object Pose Estimation

End-to-end pipeline for 6D object pose estimation using RGB-D images. The project implements object detection and pose prediction techniques, progressively incorporating depth information to improve estimation accuracy.

## 🎯 Project Overview

This project focuses on 6D pose estimation, which determines both the **3D position** (translation vector) and **3D orientation** (rotation matrix) of objects in space. The pipeline combines:

- **Object Detection**: Localizing objects in RGB images using pretrained models (e.g., YOLO11)
- **Pose Estimation**: Predicting 6D pose from detected regions using CNN-based architectures
- **RGB-D Fusion**: Enhancing predictions by incorporating depth information

The implementation follows a modular structure with clear separation of concerns, enabling easy experimentation and extension.

## 📁 Project Structure

```
polito-aml-6D_pose_estimation/
├── checkpoints/                  # 💾 MODEL CHECKPOINTS (created during training)
│   ├── pretrained/               # Pretrained weights (yolo11n.pt, yolov8n.pt)
│   ├── yolo/                     # YOLO fine-tuned models (organized structure)
│   │   └── yolo_head_only/       # Training run folder (auto-organized after training)
│   │       ├── plots/            # Training curves (F1, PR, confusion matrix)
│   │       ├── training_samples/ # Sample training batches (JPG)
│   │       ├── validation_samples/  # Sample validation batches (JPG)
│   │       ├── weights/          # Model weights (best.pt, last.pt)
│   │       ├── args.yaml         # Training configuration
│   │       └── results.csv       # Per-epoch metrics
│   ├── best_model.pth            # Best PoseEstimator model (gitignored)
│   └── checkpoint_epoch_N.pth    # Periodic PoseEstimator checkpoints (gitignored)
│
├── data/                         # 📁 DATASET FILES (LineMOD subset - download separately)
│   ├── .gitkeep
│   └── Linemod_preprocessed/     # LineMOD dataset
│       ├── data/                 # RGB-D images (01-15 objects)
│       ├── models/               # 3D object models (.ply)
│       └── yolo_symlinks/        # YOLO-format dataset (symlinks)
│           ├── images/           # train/, val/ splits
│           ├── labels/           # YOLO annotations
│           └── data.yaml         # Dataset config
│
├── dataset/                      # 📦 DATASET MODULE
│   ├── __init__.py               # Dataset exports
│   ├── custom_dataset.py         # PyTorch Dataset for pose estimation
│   └── linemod_yolo_dataset.py   # YOLO dataset preparation
│
├── models/                       # 🧠 MODELS MODULE
│   ├── __init__.py               # Model exports
│   ├── yolo_detector.py          # YOLO11-based object detection (freeze/train/validate)
│   └── pose_estimator.py         # 6D pose estimation (ResNet-50 + regression head)
│
├── notebooks/                    # 📓 JUPYTER NOTEBOOKS
│   ├── colab_training.ipynb      # Google Colab training workflow
│   └── Enhancing_6DPose_Estimation.ipynb  # Educational notebook
│
├── notebooks test/               # 🧪 TEST NOTEBOOKS
│   ├── test_explore_dataset.ipynb       # Dataset exploration & statistics
│   ├── test_yolo1_pretrained.ipynb      # YOLO pretrained detection baseline
│   ├── test_yolo2_finetuning.ipynb      # YOLO fine-tuning & validation (mAP metrics)
│   └── test_yolo3_pose_estimation.ipynb # Pose estimation testing & 3D visualization
|
|
├── utils/                        # 🛠️ UTILITIES MODULE
│   ├── __init__.py               # Utility exports
│   ├── download_dataset.py       # Dataset downloader
│   ├── transforms.py             # Pose transformations (quaternion, rotation, cropping)
│   ├── losses.py                 # Loss functions (translation + rotation)
│   ├── metrics.py                # Evaluation metrics (ADD, ADD-S)
│   ├── bbox_utils.py             # Bounding box utilities
│   ├── prepare_yolo_symlinks.py  # Create YOLO dataset with symlinks
│   └── organize_yolo_results.py  # Auto-organize YOLO outputs into subdirectories
│
├── config.py                     # ⚙️ CONFIGURATION (hyperparameters, paths, M4 optimizations)
├── requirements.txt              # 📋 PYTHON DEPENDENCIES (pip install -r requirements.txt)
├── .gitignore                    # 🚫 GIT IGNORE (data/, checkpoints/*.pth, wandb/)
│
└── README.md
```

## 🎯 Key Components

✅ **Modularity**: Code split into reusable modules (dataset, models, utils)

✅ **CLI Interface**: Argparse for flexible script execution

✅ **Reproducibility**: requirements.txt + config.py for consistent experiments

✅ **Checkpoint Management**: Automatic model saving

✅ **Logging**: Wandb integration for experiment tracking

✅ **Documentation**: Clear structure and documentation

✅ **Git-friendly**: Proper .gitignore for large files

## 🔍 Module Overview

**Dataset Module** (`dataset/`): Handles data loading for RGB-D images, bounding boxes, and 6D pose annotations. Includes `PoseDataset` class that loads LineMOD samples from official train/test splits, crops objects using bounding boxes, and converts rotation matrices to quaternions.

**Models Module** (`models/`):

- `yolo_detector.py`: yolo-based object detection wrapper
- `pose_estimator.py`: 6D pose estimation using ResNet-50 backbone + regression head outputting quaternion (4D) + translation (3D)

**Utils Module** (`utils/`):

- `download_dataset.py`: Dataset downloader
- `transforms.py`: Pose transformations (rotation matrix ↔ quaternion, bbox cropping, 3D point projection)
- `losses.py`: Combined loss function (L1 smooth for translation + geodesic distance for rotation)
- `metrics.py`: ADD and ADD-S metrics with 3D model loading

**Notebooks** (`notebooks/`): Jupyter notebooks for Colab training and educational purposes

**Test** (`test/`):

- `test_yolo.ipynb`: Detection baseline testing with ground truth comparison
- `test_pose_estimation.ipynb`: Pose prediction visualization with 3D bounding boxes, per-object ADD analysis

**Config** (`config.py`): Centralized configuration including detection parameters (YOLO), pose estimation parameters (batch size, learning rate, loss weights), and object information (symmetric objects, ID-to-name mapping)

## 🔄 Typical Workflow

### 1. Initial Setup

```bash
git clone <repo-url>
cd polito-aml-6D_pose_estimation
pip install -r requirements.txt
python utils/download_dataset.py
```

> **📝 Note on Checkpoints**: All models save in `checkpoints/`:
> - YOLO models: `checkpoints/yolo/`
> - Pose models: `checkpoints/*.pth`

**Device Detection:**
The system automatically detects the best available device (CUDA > MPS > CPU).
Test your device with:

```bash
python test_device.py
```

On **Apple Silicon Mac** (M1/M2/M3), MPS (Metal Performance Shaders) will be automatically enabled for ~5-10x speedup vs CPU.

### 2. Training (6D Pose Estimation)

**Training Modes:**

| Mode | Command | Time (Mac M1/M2) | Params Trained | Quality | Use Case |
|------|---------|------------------|----------------|---------|----------|
| **Quick Test** | `--freeze_backbone --epochs 2` | 2-3 min | ~3M (head only) | Basic | Fast prototyping |
| **Medium** | `--epochs 5` | 10-15 min | ~26M (full) | Good | Quick experiments |
| **Full** | `--epochs 50` | 2-4 hours | ~26M (full) | Best | Final model |

**Key Training Features:**

- **Gradient Accumulation**: Effective batch size = batch_size × gradient_accum_steps
- **Mixed Precision (FP16)**: Faster training on Apple Silicon / CUDA GPUs
- **Validation with ADD Metric**: Computed every 5 epochs using official test split
- **Automatic Checkpointing**: Best model saved based on validation ADD
- **Wandb Logging**: Track experiments with Weights & Biases

### 3. Testing & Evaluation

```bash
# Test detection baseline
jupyter notebook test/test_yolo.ipynb

# Evaluate pose estimation on test set
jupyter notebook test/test_pose_estimation.ipynb
```

**Evaluation Metrics:**

- **ADD (Average Distance of Model Points)**: Mean distance between transformed 3D points
- **ADD-S**: Symmetric variant for objects like eggbox (obj_08) and glue (obj_09)
- **Accuracy**: Percentage of predictions with ADD < 10% of object diameter

## 📢 Release Information

**📅 Last update:** November 2025  
**🏷️ Version:** v1.0.0

*For details on changes and fixes, see the changelog in the repository.*
