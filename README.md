# 6D Pose Estimation on LineMOD

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-red.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

6D object pose estimation on LineMOD dataset through modular pipelines: detection, rotation/translation estimation, deep learning model training and validation.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project provides a complete pipeline for 6D object pose estimation on the LineMOD dataset, combining detection (YOLO), rotation estimation (ResNet-50/ResNet-18), and translation (pinhole model, end-to-end RGB, or RGB-D fusion). It is designed for:
- Researchers and students in computer vision and robotics
- Developers interested in modular pipelines for 6D pose estimation
- Those who want to reproduce, extend, or compare solutions on LineMOD

**Problem solved:** accurate estimation of 3D position and orientation of known objects in RGB images, with an easily adaptable and reproducible pipeline.

---

## Features
- **Modular pipelines:** baseline (YOLO + pinhole + ResNet), end-to-end RGB (YOLO + ResNet), and RGB-D fusion (YOLO + multi-modal fusion model)
- **Training and fine-tuning:** YOLOv11 for object detection, ResNet-50 for rotation and translation (RGB), ResNet-18 + DepthEncoder for RGB-D fusion
- **Dataset handlers:** loading, parsing, and splitting LineMOD, conversion to YOLO format
- **Metrics and visualization:** ADD/ADD-S, plots, prediction overlays, per-class analysis
- **Checkpoints and reproducibility:** weight saving, logs, YAML configurations
- **Example notebooks:** training, validation, pipelines, model comparison
- **Centralized configuration:** parameters in `config.py`
- **GPU/CPU/MPS support:** automatic device detection
- **RGB-D fusion architecture:** combines RGB features, depth encoding, and metadata (bbox + camera intrinsics) for improved pose estimation

---

## Architecture

The project is divided into main modules, each documented with specific READMEs:

- **config.py**: Centralized configuration (paths, hyperparameters, object mappings)
- **checkpoints/** ([README](checkpoints/README.md)): Model checkpoints, weights, logs, configurations
- **data/** ([README](data/README.md)): Pre-processed LineMOD dataset, annotations, 3D models, YOLO symlinks
- **dataset/** ([README](dataset/README.md)): Loaders, parsers, conversions, and DataLoaders for LineMOD
- **models/** ([README](models/README.md)): YOLO implementations, ResNet-50 baseline, end-to-end RGB, RGB-D fusion model
- **utils/** ([README](utils/README.md)): Utilities for bbox, losses, metrics, pinhole, training, visualization
- **notebooks/** ([README](notebooks/readme.md)): Jupyter notebooks for training, pipelines, validation, analysis
- **experimental_notebooks/** ([README](experimental_notebooks/README.md)): Exploration and comparison notebooks

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Ultralytics YOLO (v11)
- Other: numpy, pandas, matplotlib, pyyaml, tqdm, PIL, opencv-python, gdown

### Installation
1. **Clone the repository:**
	```bash
	git clone https://github.com/[user]/[repo].git
	cd polito-aml-6D_pose_estimation
	```
2. **Create a virtual environment (optional but recommended):**
	```bash
	python -m venv .venv
	source .venv/bin/activate
	```
3. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	# or use pyproject.toml/poetry if preferred
	```
4. **Download the pre-processed LineMOD dataset:**
	```bash
	python utils/download_dataset.py
	# or follow the instructions in data/README.md
	```

### Configuration
- All paths and parameters are managed in `config.py`.
- Modify parameters according to your needs (e.g., device, batch size, learning rate).

### First Run
- Execute one of the notebooks in `notebooks/` for pipelines, training, or validation.
- Or launch a custom script using the `models/`, `utils/`, `dataset/` modules.

---

## Usage

### Baseline Pipeline (YOLO + Pinhole + ResNet)
```python
from models.yolo_detector import YOLODetector
from models.pose_estimator_baseline import PoseEstimatorBaseline
from utils.pinhole import compute_translation_pinhole

yolo = YOLODetector(model_name='yolo11n.pt', num_classes=13)
model = PoseEstimatorBaseline(pretrained=True)
# ...load batch, run detection, estimate rotation and translation...
```

### End-to-End RGB Pipeline (YOLO + ResNet)
```python
from models.yolo_detector import YOLODetector
from models.pose_estimator_endtoend import PoseEstimator

yolo = YOLODetector(model_name='yolo11n.pt', num_classes=13)
model = PoseEstimator(pretrained=True)
# ...load batch, run detection, estimate full 6D pose...
```

### RGB-D Fusion Pipeline (YOLO + RGB-D Fusion Model)
```python
from models.yolo_detector import YOLODetector
from models.pose_estimator_RGBD import RGBDFusionModel

yolo = YOLODetector(model_name='yolo11n.pt', num_classes=13)
model = RGBDFusionModel(pretrained_rgb=True)
model.load_weights('checkpoints/pose/fusion_rgbd_512/best.pt')
# ...load batch with RGB, depth, and metadata, estimate full 6D pose...
```

### Fine-tuning YOLO
```python
from models.yolo_detector import YOLODetector
detector = YOLODetector(model_name='yolo11n', pretrained=True, num_classes=13)
detector.freeze_backbone(freeze_until_layer=10)
results = detector.train(data_yaml='path/to/data.yaml', epochs=20)
```

### Training Baseline Model (Rotation Only)
```python
from models.pose_estimator_baseline import PoseEstimatorBaseline
model = PoseEstimatorBaseline(pretrained=True)
# ...setup optimizer, loss, train loop...
```

### Training RGB-D Fusion Model
```python
from models.pose_estimator_RGBD import RGBDFusionModel
model = RGBDFusionModel(pretrained_rgb=True)
# ...setup optimizer, loss, train loop with RGB, depth, and metadata...
```

For more detailed examples, see the notebooks in [notebooks/](notebooks/readme.md).

---

## Project Structure

```
polito-aml-6D_pose_estimation/
├── config.py
├── checkpoints/         # Model checkpoints, weights, logs
├── data/                # LineMOD dataset, annotations, 3D models
├── dataset/             # Dataset loaders and parsers
├── models/              # YOLO, ResNet-50, RGB-D fusion models
├── utils/               # Utilities, metrics, losses, training
├── notebooks/           # Training, validation, pipeline notebooks
├── experimental_notebooks/ # Exploration notebooks
├── requirements.txt
├── pyproject.toml
└── README.md
```

Each directory contains a detailed README explaining its contents and usage.

---

## Documentation
- [checkpoints/README.md](checkpoints/README.md): Model checkpoints, weights, logs
- [data/README.md](data/README.md): Pre-processed LineMOD dataset
- [dataset/README.md](dataset/README.md): Dataset loaders and parsers
- [models/README.md](models/README.md): YOLO, ResNet-50, RGB-D fusion models
- [utils/README.md](utils/README.md): Utilities, metrics, pinhole, training
- [notebooks/readme.md](notebooks/readme.md): Pipeline, training, validation notebooks
- [experimental_notebooks/README.md](experimental_notebooks/README.md): Exploration and comparison notebooks

---

## Contributing

Contributions, bug reports, and improvement proposals are welcome! Open an issue or pull request following GitHub best practices.

---

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

- Alessandro - Politecnico di Torino
- Advanced Machine Learning Course Project

For questions, suggestions, or collaborations, feel free to reach out!
