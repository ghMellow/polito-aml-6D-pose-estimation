
# Utils - Utility Scripts for 6D Pose Estimation

## 1. Overview

This directory contains all shared utilities and support modules for the LineMOD 6D Pose Estimation project. Common functions, transformations, metrics, visualization, data management, and support scripts for training, validation, and dataset preparation are centralized here. The goal is to avoid code duplication and provide reusable tools for training, validation, and analysis pipelines.

## 2. Directory Structure

- `bbox_utils.py` - Bounding box utilities (cropping, format conversions, padding)
- `download_dataset.py` - Script to download and extract the pre-processed LineMOD dataset
- `file_io.py` - File I/O utilities for loading and saving data
- `losses.py` - Loss functions for pose estimation (rotation, translation, combined)
- `metrics.py` - ADD/ADD-S metrics for pose model evaluation
- `model_loader.py` - Model loading and checkpoint management utilities
- `organize_yolo_results.py` - Automatic organization of YOLO results into subfolders
- `pinhole.py` - Pinhole camera model for 3D translation computation from bounding box and depth
- `prepare_yolo_symlinks.py` - YOLO dataset preparation via symlinks (or file copy)
- `training.py` - Centralized training loops for pose models
- `transforms.py` - Image transformations, quaternion conversions, augmentation
- `validation.py` - Validation pipeline and result analysis functions
- `visualization.py` - Visualization utilities for training, metrics, and samples
- `README.md` - This documentation file describing the contents of the utils directory
- `__pycache__/` - Python cache directory for compiled files (not for direct use)

## 3. Main Components


### `bbox_utils.py`
- **Purpose**: Handles common bounding box operations: centered cropping, padding, and format conversions (e.g., YOLO format).
- **Main Functions**:
    - `crop_and_pad(img, bbox, output_size, margin)` - Square crop centered on the bounding box with optional padding.
    - `convert_bbox_to_yolo_format(bbox, img_width, img_height)` - Convert bounding box to normalized YOLO format.
    - `crop_bbox_optimized(img, bbox_xyxy, margin, output_size)` - Optimized crop from bounding box in xyxy format.
- **Dependencies**: `numpy`, `cv2`


### `download_dataset.py`
- **Purpose**: Downloads and extracts the pre-processed LineMOD dataset from Google Drive.
- **Main Functions**:
    - `download_linemod_dataset(output_dir)` - Downloads and extracts the dataset to the desired directory.
- **Dependencies**: `gdown`, `zipfile`, `os`, `pathlib`, `config.py`
- **Usage Example**:
    ```bash
    python utils/download_dataset.py --output_dir ./data
    ```


### `file_io.py`
- **Purpose**: Provides utilities for file I/O operations, loading and saving various data formats.
- **Main Functions**:
    - File loading and saving utilities for common data formats
    - Path management and file system operations
- **Dependencies**: `pathlib`, `pickle`, `json`, `yaml`


### `losses.py`
- **Purpose**: Implements loss functions for pose estimation (rotation, translation, combined).
- **Main Classes**:
    - `PoseLoss` - Combined loss (rotation + translation) with configurable weights.
    - `PoseLossBaseline` - Loss only for rotation (baseline, translation not learned).
- **Dependencies**: `torch`, `config.py`
- **Pattern**: Uses Smooth L1 for translation, geodesic distance on quaternions for rotation.


### `metrics.py`
- **Purpose**: Calculates ADD/ADD-S evaluation metrics for 6D pose (both batch and per sample), loads 3D models and object info.
- **Main Functions**:
    - `compute_add_batch_rotation_only`, `compute_add_batch_full_pose`, `compute_add_batch`, `compute_add_batch_gpu` - ADD computation for different scenarios.
    - `load_models_info`, `load_3d_model`, `load_all_models` - Loading info and 3D meshes.
- **Dependencies**: `numpy`, `torch`, `config.py`


### `model_loader.py`
- **Purpose**: Utilities for loading models and managing checkpoints.
- **Main Functions**:
    - Model checkpoint loading and saving
    - Weight management and model initialization
- **Dependencies**: `torch`, `pathlib`, `config.py`


### `organize_yolo_results.py`
- **Purpose**: Organizes YOLO results (plots, samples, weights) into ordered subfolders.
- **Main Functions**:
    - `organize_yolo_output(run_dir, destination_dir)` - Moves and orders output files.
    - `print_organization_summary(project_dir, stats)` - Prints organization summary.
- **Dependencies**: `pathlib`, `shutil`, `config.py`


### `pinhole.py`
- **Purpose**: Implements the pinhole camera model to calculate 3D translation from bounding box, depth map, and camera intrinsics.
- **Main Functions**:
    - `load_camera_intrinsics(gt_yml_path)` - Loads intrinsic parameters from info.yml.
    - `compute_translation_pinhole(bbox, depth_path, camera_intrinsics, ...)` - Calculates 3D translation.
    - `compute_translation_pinhole_batch(...)` - Batch version.
- **Dependencies**: `numpy`, `PIL`, `yaml`, `config.py`


### `prepare_yolo_symlinks.py`
- **Purpose**: Prepares dataset structure in YOLO format using symlinks (or file copy if not supported).
- **Main Functions**:
    - `prepare_yolo_dataset_symlinks(dataset_root, output_root, use_symlinks)` - Creates symlinks/copies and YOLO labels.
    - `create_data_yaml(output_root, dataset_root)` - Generates data.yaml file for YOLO.
- **Dependencies**: `pathlib`, `tqdm`, `PIL`, `config.py`, `bbox_utils.py`


### `training.py`
- **Purpose**: Centralizes training loops for pose models (baseline and end-to-end).
- **Main Functions**:
    - `train_pose_baseline(...)` - Training for models that predict only rotation.
    - `train_pose_full(...)` - Training for models that predict rotation and translation.
- **Dependencies**: `torch`, `tqdm`, `config.py`


### `transforms.py`
- **Purpose**: Image and pose transformations, conversions between rotation representations, and data augmentation.
- **Main Functions**:
    - `rotation_matrix_to_quaternion`, `quaternion_to_rotation_matrix_batch` - Conversions between rotation and quaternion.
    - `crop_image_from_bbox` - Crop images from bounding box.
    - `get_pose_transforms` - Compose transformations for training/validation.
- **Dependencies**: `torch`, `numpy`, `PIL`, `config.py`


### `validation.py`
- **Purpose**: Validation pipeline for pose models and complete pipelines (YOLO+ResNet), metrics calculation, and results saving.
- **Main Functions**:
    - `run_pinhole_deep_pipeline`, `run_deep_pose_pipeline`, `run_yolo_baseline_pipeline`, `run_yolo_endtoend_pipeline` - Validation pipelines.
    - `save_validation_results`, `load_validation_results` - Results management.
    - `calc_add_accuracy_per_class`, `calc_pinhole_error_per_class` - Per-class metrics analysis.
- **Dependencies**: `torch`, `config.py`, utils modules


### `visualization.py`
- **Purpose**: Visualization utilities for training, metrics, samples, loss and ADD curves.
- **Main Functions**:
    - `show_pose_samples`, `show_pose_samples_with_add` - Visualize images and predictions.
- **Dependencies**: `matplotlib`, `pandas`, `torch`, `config.py`

### `README.md`
- **Purpose**: This documentation file, describing the contents and usage of the utils directory.

### `__pycache__/`
- **Purpose**: Python cache directory for compiled files. Not intended for direct use.


## 4. Usage Examples

### Example: Crop and bbox conversion
```python
from utils.bbox_utils import crop_and_pad, convert_bbox_to_yolo_format
img_crop = crop_and_pad(img, bbox=[x, y, w, h], output_size=(224, 224), margin=0.1)
yolo_bbox = convert_bbox_to_yolo_format([x, y, w, h], img_width=640, img_height=480)
```

### Example: Translation computation with pinhole
```python
from utils.pinhole import compute_translation_pinhole, load_camera_intrinsics
intrinsics = load_camera_intrinsics('data/01/gt.yml')
t = compute_translation_pinhole([x1, y1, x2, y2], 'data/01/depth/0000.png', intrinsics)
```

### Example: Training loop
```python
from utils.training import train_pose_baseline
history, best_loss, best_epoch = train_pose_baseline(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50)
```

### Example: ADD metric computation
```python
from utils.metrics import compute_add_batch_rotation_only
results = compute_add_batch_rotation_only(pred_R_batch, gt_R_batch, obj_ids, models_dict, models_info)
```


## 5. Technical Notes and Conventions

- **Pattern**: All modules are designed to be importable and reusable in different pipelines.
- **Config**: Many modules depend on `config.py` for global parameters (paths, weights, class mappings, device, etc.).
- **Batch and GPU**: Metrics and loss functions are optimized for batch processing and support GPU.
- **Formats**: Functions accept both `numpy` and `torch` where possible, and convert internally.
- **Visualization**: Plotting functions use `matplotlib` and `pandas` for tables and charts.
- **Dataset**: YOLO dataset preparation uses symlinks for efficiency and space savings.
- **Documentation**: Each file and main function is documented with detailed docstrings.

---

For additional details, refer to the docstrings in individual files or the main project documentation.
