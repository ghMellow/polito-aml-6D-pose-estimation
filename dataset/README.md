# Dataset Module — LineMOD Utilities

## 1. Overview
This folder provides classes and utilities for handling, exploring, and pre-processing the LineMOD dataset commonly used for 6D pose estimation and object detection tasks. It implements shared logic for loading images and annotations, parsing ground-truth files, converting annotations to formats compatible with training pipelines (e.g., PyTorch, YOLO), and helper functions to create DataLoaders.

## 2. Folder structure
- `__init__.py` — Makes the folder a Python package.
- `linemod_base.py` — Abstract base class for LineMOD dataset handling (image/depth loading, annotations, info files, sample collection).
- `linemod_pose.py` — Extension for pose estimation tasks (rotation/translation handling, cropping, DataLoader helpers).
- `linemod_yolo.py` — Utilities to convert / explore LineMOD annotations in YOLO format (normalized bounding boxes).
- `README.md` — This file.

## 3. Main components

### `linemod_base.py`
- Purpose:
  - Implements `LineMODDatasetBase`, which inherits from `torch.utils.data.Dataset` and centralizes common logic for loading RGB, depth, ground-truth (`gt.yml`), camera info (`info.yml`), and collecting sample indices.
- Key class/function:
  - `LineMODDatasetBase`: methods to load images, depth maps, ground-truth entries, camera info, and iterate samples.
- Primary dependencies:
  - `torch.utils.data.Dataset`, `PIL.Image`, `yaml`, `numpy`, `config.Config`

### `linemod_pose.py`
- Purpose:
  - Extends the base dataset for pose estimation tasks, adding cropping, normalization, rotation-to-quaternion conversion, and utilities for creating PyTorch DataLoaders with train/val/test splits.
- Key classes/functions:
  - `LineMODPoseDataset`: returns crops, full RGB, quaternion, translation, camera intrinsics, bbox, and path metadata per sample.
  - `create_pose_dataloaders`: helper that builds train/val/test DataLoaders (supports reproducible random split via `Config` values).
- Primary dependencies:
  - `LineMODDatasetBase`, `torch`, `PIL.Image`, `utils.bbox_utils`, `utils.transforms`, `config.Config`

### `linemod_yolo.py`
- Purpose:
  - Provides a lightweight dataset wrapper to inspect and convert LineMOD annotations into YOLO-style normalized bounding boxes and class IDs. Useful for visualization and debugging prior to exporting labels to disk.
- Key class/function:
  - `LineMODYOLODataset`: yields `PIL.Image`, normalized bounding boxes, class IDs, and sample metadata.
- Primary dependencies:
  - `LineMODDatasetBase`, `utils.bbox_utils`

## 4. Usage — Examples

Loading dataset for pose estimation:
```python
from dataset.linemod_pose import LineMODPoseDataset, create_pose_dataloaders

dataset_root = 'data/Linemod_preprocessed'  # or your custom path
train_loader, val_loader, test_loader = create_pose_dataloaders(
    dataset_root=dataset_root,
    batch_size=16,
    crop_margin=30,
    output_size=128,
    num_workers=4
)

for batch in train_loader:
    rgb_crop = batch['rgb_crop']  # Tensor [B, C, H, W]
    quaternion = batch['quaternion']
    # ... training loop ...
```

Inspecting annotations in YOLO format:
```python
from dataset.linemod_yolo import LineMODYOLODataset

dataset = LineMODYOLODataset('data/Linemod_preprocessed')
sample = dataset[0]
print('Bboxes (YOLO):', sample['bboxes'])
print('Class IDs:', sample['class_ids'])
```

## 5. Technical notes
- Patterns:
  - All dataset classes inherit from `LineMODDatasetBase` to maximize reuse and follow DRY principles.
  - Train/test splits are handled using `train.txt` / `test.txt` files located inside each object subfolder.
  - Metadata files (`gt.yml`, `info.yml`) are preloaded into a lightweight cache for efficiency.
  - Cropping and image normalization rely on helper utilities (e.g., `crop_and_pad`, `get_pose_transforms`).
  - Rotation conversions use matrix → quaternion functions (e.g., `rotation_matrix_to_quaternion`).
- Conventions:
  - Folder IDs correspond to LineMOD object IDs (01, 02, ...).
  - Ground-truth annotations are returned as Python dicts.
  - Images may be returned as `PIL.Image` or converted to `torch.Tensor` depending on the dataset class and transform pipeline.
- Details:
  - These classes do not directly write YOLO label files to disk; they provide normalized annotation data suitable for exporting.
  - Paths to RGB/depth files are centralized and compatible with external training pipelines.

For implementation details of helper utilities (e.g., `crop_and_pad`, `rotation_matrix_to_quaternion`), see the `utils/` folder.
