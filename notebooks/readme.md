
# Notebooks Directory - Examples, Pipelines, and Experimentation

## 1. Overview
This directory contains all Jupyter notebooks for examples, validation, and experimentation of the 6D pose estimation pipeline. The notebooks cover the entire workflow: from YOLO fine-tuning, to pose model training, through to evaluation and results visualization. They are designed for both experiment reproducibility and interactive project exploration.

## 2. Directory Structure
- `pipeline_rgb_baseline.ipynb` - Complete baseline pipeline: YOLO + Pinhole + ResNet (rotation only)
- `pipeline_rgb_endtoend.ipynb` - Complete end-to-end RGB pipeline: YOLO + ResNet (rotation + translation)
- `pipeline_rgbd_endtoend.ipynb` - Complete RGB-D fusion pipeline: YOLO + RGB-D Fusion Model (rotation + translation)
- `finetuning_Yolo.ipynb` - YOLO fine-tuning and validation on LineMOD
- `finetuning_rgb_baseline.ipynb` - Training and validation of baseline model (rotation only)
- `finetuning_rgb_endtoend.ipynb` - Training and validation of end-to-end RGB model (rotation + translation)
- `training_rgbd_endtoend.ipynb` - Training and validation of RGB-D fusion model (rotation + translation)
- `figures/` - Folder containing result figures and plots
  - `rgbd_e2e_pipeline/` - Figures for the RGB-D end-to-end pipeline
    - `accuracy_vs_threshold.png`, `best_predictions_yolo.png`, `error_distributions.png`, `per_object_analysis.png`, `roi_crops_demo.png`, `worst_predictions_yolo.png`, `yolo_detection_demo.png`
- `README.md` - This documentation file describing the contents of the notebooks directory


## 3. Main Notebooks

### `pipeline_rgb_baseline.ipynb`
- **Purpose:** Runs the classic pipeline: object detection with YOLO, translation estimation with the pinhole model, and rotation estimation with ResNet-50.
- **Main steps:**
  1. Load pre-trained models
  2. Extract test batch (with GT or YOLO bounding boxes)
  3. Compute and visualize rotation and translation
  4. Evaluate the ADD metric and visualize per-class results
- **Key dependencies:** `models/`, `utils/`, `dataset/`, `config.py`

### `pipeline_rgb_endtoend.ipynb`
- **Purpose:** Runs the end-to-end RGB pipeline: YOLO for detection, ResNet-50 for joint rotation and translation estimation.
- **Main steps:**
  1. Load pre-trained models
  2. Extract test batch (with GT or YOLO bounding boxes)
  3. Compute and visualize rotation and translation
  4. Evaluate the ADD metric and visualize per-class results
- **Key dependencies:** `models/`, `utils/`, `dataset/`, `config.py`

### `pipeline_rgbd_endtoend.ipynb`
- **Purpose:** Runs the RGB-D fusion pipeline: YOLO for detection, RGB-D Fusion Model for joint rotation and translation estimation using RGB, depth, and metadata.
- **Main steps:**
  1. Load pre-trained YOLO and RGB-D fusion models
  2. Extract test batch with RGB, depth, and bounding boxes
  3. Compute and visualize rotation and translation
  4. Evaluate the ADD metric and visualize per-class results
- **Key dependencies:** `models/`, `utils/`, `dataset/`, `config.py`

### `finetuning_Yolo.ipynb`
- **Purpose:** Demonstrates how to fine-tune YOLO on LineMOD, organize results, and evaluate performance.
- **Main steps:**
  1. Load pre-trained YOLO model
  2. Prepare dataset and data.yaml
  3. Fine-tune the detection head
  4. Evaluate on the test set and visualize predictions
- **Key dependencies:** `models.yolo_detector`, `utils.prepare_yolo_symlinks`, `config.py`

### `finetuning_rgb_baseline.ipynb`
- **Purpose:** Trains and evaluates the baseline model (rotation only) on LineMOD.
- **Main steps:**
  1. Load and visualize the dataset
  2. Initialize the `PoseEstimatorBaseline` model
  3. Training and checkpoint saving
  4. Evaluate on the test set (ADD rotation)
  5. Visualize metrics and plots
- **Key dependencies:** `models.pose_estimator_baseline`, `utils/`, `config.py`

### `finetuning_rgb_endtoend.ipynb`
- **Purpose:** Trains and evaluates the end-to-end RGB model (rotation + translation) on LineMOD.
- **Main steps:**
  1. Load and visualize the dataset
  2. Initialize the `PoseEstimator` model
  3. Training and checkpoint saving
  4. Evaluate on the test set (ADD full pose)
  5. Visualize metrics and plots
- **Key dependencies:** `models.pose_estimator_endtoend`, `utils/`, `config.py`

### `training_rgbd_endtoend.ipynb`
- **Purpose:** Trains and evaluates the RGB-D fusion model (rotation + translation) on LineMOD.
- **Main steps:**
  1. Load and visualize the dataset with RGB, depth, and metadata
  2. Initialize the `RGBDFusionModel` model
  3. Training with multi-modal inputs and checkpoint saving
  4. Evaluate on the test set (ADD full pose)
  5. Visualize metrics and plots
- **Key dependencies:** `models.pose_estimator_RGBD`, `utils/`, `config.py`


## 4. Technical Notes and Conventions
- **Modular structure:** Each notebook focuses on a specific phase (fine-tuning, training, pipeline, evaluation).
- **Reproducibility:** Notebooks save checkpoints, intermediate results, and plots to facilitate experiment replication.
- **Centralized configuration:** All key parameters are managed through `config.py`.
- **Visualization:** Extensive use of plots, tables, and images to interpret results.
