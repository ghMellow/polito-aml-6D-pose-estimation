# Directory data

## 1. Overview

This `data` folder contains the datasets used by the project; its primary subfolder is `Linemod_preprocessed`, which holds a pre-processed version of the LINEMOD dataset for 6D object pose estimation. The `data` folder provides structured and annotated data for training, validation and testing of detection and pose estimation models, as well as auxiliary resources for segmentation and YOLO pipelines.

## 2. Repository structure

- **data/**: Subfolders for each object (01, 02, ..., 15). Each object folder contains RGB images, depth maps, masks and annotation files.
- **models/**: 3D object models and metadata files.
- **segnet_results/**: Segmentation outputs per object (label masks from segmentation networks).
- **yolo_symlinks/**: Reorganized dataset layout for YOLO training, using symlinks to avoid data duplication.

## 3. Contents (detailed)

### data/[ID]/

- `gt.yml`: Ground-truth annotations for each image. Typically contains object poses, bounding boxes, and other per-image metadata.
- `info.yml`: Additional image and dataset information (image sizes, camera intrinsics, etc.).
- `train.txt` / `test.txt`: Lists of image filenames used for training and testing splits.
- `rgb/`: RGB images (files typically zero-padded, e.g. `0001.png`).
- `depth/`: Depth maps aligned to RGB images.
- `mask/`: Binary object masks (object=255, background=0).
- `mask_all/` (present for some objects): Masks including all objects in the scene.

These files are consumed by the project's dataset loaders (see `dataset/linemod_base.py` and `dataset/linemod_pose.py`).

### models/

- `models_info.yml`: Metadata for the objects (real-world sizes, symmetries, etc.).
- `obj_XX.ply`: 3D mesh files for each object used for pose evaluation and synthetic rendering.

These are used during pose evaluation and for generating synthetic training data or visualizations.

### segnet_results/[ID]_label/

- Contains segmentation label images produced by segmentation networks (example: SegNet). Each folder holds per-frame label masks that can be used for training or evaluating segmentation components of the pipeline.

### yolo_symlinks/

- `data.yaml`: YOLO dataset configuration (class names and paths to images/labels).
- `images/train/`, `images/val/`: Symlinked or copied images for YOLO training and validation.
- `labels/train/`, `labels/val/`: Corresponding YOLO label files (one `.txt` per image) in the format `[class x_center y_center width height]` normalized to image size.

For example, label files in `yolo_symlinks/labels/train/` follow the standard YOLO format and may include filenames like `01_0004.txt`, `01_0009.txt`, etc. These label files list bounding boxes and classes for the YOLO detector.

## 4. Usage examples

Loading ground-truth annotations and an RGB image:

```python
import yaml
from PIL import Image

# Load ground truth annotations
with open('data/Linemod_preprocessed/data/01/gt.yml', 'r') as f:
    gt = yaml.safe_load(f)

# Load an RGB image
img = Image.open('data/Linemod_preprocessed/data/01/rgb/0001.png')

# Load a binary mask
mask = Image.open('data/Linemod_preprocessed/data/01/mask/0001.png')
```

Training YOLO with the prepared symlinked dataset:

```bash
yolo train data=data/Linemod_preprocessed/yolo_symlinks/data.yaml model=yolov8n.pt
```

If you use YOLOv5/v8 directly, ensure `data.yaml` points to the correct `images` and `labels` directories and contains the `nc` (number of classes) and `names` entries.

## 5. Technical notes and conventions

- Pattern: Structure inspired by the BOP (Benchmark for 6D Object Pose Estimation) format.
- File and naming conventions:
  - Images and annotation files are zero-padded (e.g. `0001.png`).
  - YOLO labels are formatted as `class x_center y_center width height` with coordinates normalized to image dimensions.
- Masks are binary images (object=255, background=0).
- YAML files follow standard syntax and can be read with PyYAML (`yaml.safe_load`).
- `yolo_symlinks` uses filesystem symlinks where possible to avoid duplicating large image files.

## 6. Extensibility

The layout supports adding new objects or additional preprocessing / postprocessing pipelines. To add another object: create a new `data/XX/` folder with the same structure (`rgb/`, `depth/`, `mask/`, `gt.yml`, `info.yml`, train/test lists) and update `models/models_info.yml` if a new 3D model is required.

---

This README describes the contents of the `Linemod_preprocessed` folder and how the files are used by the repository.
