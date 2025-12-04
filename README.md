# 6D Object Pose Estimation

End-to-end pipeline for 6D object pose estimation using RGB-D images. The project implements object detection and pose prediction techniques, progressively incorporating depth information to improve estimation accuracy.

## 🎯 Project Overview

This project focuses on 6D pose estimation, which determines both the **3D position** (translation vector) and **3D orientation** (rotation matrix) of objects in space. The pipeline combines:

- **Object Detection**: Localizing objects in RGB images using pretrained models (e.g., YOLOv8)
- **Pose Estimation**: Predicting 6D pose from detected regions using CNN-based architectures
- **RGB-D Fusion**: Enhancing predictions by incorporating depth information

The implementation follows a modular structure with clear separation of concerns, enabling easy experimentation and extension.

## 📁 Project Structure

```
polito-aml-6D_pose_estimation/
├── checkpoints/                  # 💾 MODEL CHECKPOINTS (created during training)
│   ├── .gitkeep                  # Keeps folder in git
│   ├── best_model.pth            # Best model saved automatically (gitignored)
│   └── checkpoint_epoch_N.pth    # Periodic checkpoints (gitignored)
│
├── data/                         # 📁 DATASET FILES (LineMOD subset - download separately)
│   ├── .gitkeep
│   └── ...                       # RGB-D images, bounding boxes, masks, 3D models
|
├── dataset/                      # 📦 DATASET MODULE
│   ├── __init__.py               # Dataset exports
│   └── custom_dataset.py         # PyTorch Dataset class for data loading
│
├── models/                       # 🧠 MODELS MODULE
│   ├── __init__.py               # Model exports
│   └── yolo_detector.py          # YOLO-based object detection
│
├── utils/                        # 🛠️ UTILITIES MODULE
│   ├── __init__.py               # Utility exports
│   └── download_dataset.py       # Dataset downloader
│
├── scripts/                      # 🚀 EXECUTABLE SCRIPTS
│   ├── train.py                  # 🚂 Training script (main training loop with CLI)
│   └── eval.py                   # 📊 Evaluation script (evaluation with CLI)
│
├── notebooks/                    # 📓 JUPYTER NOTEBOOKS
│   ├── colab_training.ipynb      # Google Colab training workflow
│   └── Enhancing_6DPose_Estimation.ipynb  # Educational notebook
│
├── test/                         # 🧪 TEST NOTEBOOKS
│   ├── test_local_dataset.ipynb  # Dataset testing
│   └── test_yolo.ipynb           # YOLO testing
│
├── config.py                     # ⚙️ CONFIGURATION (hyperparameters and settings)
├── pyproject.toml                # 📦 PROJECT METADATA (Poetry configuration)
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

**Dataset Module** (`dataset/`): Handles data loading and preprocessing for RGB-D images, bounding boxes, masks, and 3D models

**Models Module** (`models/`): Contains pose estimation architectures and model creation functions

**Utils Module** (`utils/`): Provides data download utilities and helper functions

**Scripts** (`scripts/`): Executable training and evaluation scripts with command-line interface

**Notebooks** (`notebooks/`): Jupyter notebooks for Colab training and educational purposes

**Config** (`config.py`): Centralized hyperparameters and configuration

## 🔄 Typical Workflow

### 1. Initial Setup

```bash
git clone <repo-url>
cd polito-aml-6D_pose_estimation
pip install -r requirements.txt
python utils/download_dataset.py
```

### 2. Training

```bash
python scripts/train.py --data_dir ./data --epochs 50 --batch_size 32 --use_wandb
```

### 3. Evaluation

```bash
python scripts/eval.py --checkpoint ./checkpoints/best_model.pth --data_dir ./data
```

## 📢 Release Information

**📅 Last update:** November 2025  
**🏷️ Version:** v1.0.0

*For details on changes and fixes, see the changelog in the repository.*
