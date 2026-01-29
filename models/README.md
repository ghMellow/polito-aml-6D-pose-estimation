# Models Directory - 6D Pose Estimation

## 1. Overview
This directory contains all main model implementations for the 6D pose estimation pipeline. It includes **object detection** models (YOLO) and **6D pose estimation** models (baseline, end-to-end RGB, and RGB-D fusion), as well as supporting modules for encoding depth, metadata, and regressing the final pose. The models are modular, extensible, and designed for easy integration into the project pipeline.

## 2. Directory Structure
- `yolo_detector.py` - Wrapper for YOLOv11 (Ultralytics) for object detection
- `pose_estimator_baseline.py` - Baseline model: estimates only rotation (quaternion) with ResNet-50, translation computed geometrically
- `pose_estimator_endtoend.py` - End-to-end RGB model: estimates both rotation (quaternion) and translation (3D vector) with ResNet-50
- `pose_estimator_RGBD.py` - RGB-D fusion model: multi-modal architecture combining RGB, depth, and metadata for joint rotation and translation estimation
- `depth_encoder.py` - Custom CNN encoder for depth image features
- `meta_encoder.py` - MLP encoder for bounding box and camera intrinsics metadata
- `pose_regressor.py` - MLP regressor head for final pose prediction
- `__init__.py` - Makes the directory a Python module
- `README.md` - This file

## 3. Additional Files
- `__pycache__/` - Python bytecode cache

## 4. Main Components

### `yolo_detector.py`
- **Purpose:**
  - Provides a `YOLODetector` class that encapsulates the logic for loading, training, validation, and inference of YOLOv11 models via Ultralytics.
  - Allows customization of number of classes, backbone freezing, model export, and prediction visualization.
- **Main Classes/Functions:**
  - `YOLODetector`: complete wrapper for YOLOv11 (initialization, train, predict, validate, export, freeze_backbone, etc.)
  - `visualize_detections`: function to visualize predictions on images
- **Key Dependencies:**
  - `ultralytics` (YOLO), `torch`, `numpy`, `config.Config`

### `pose_estimator_baseline.py`
- **Purpose:**
  - Implements the baseline model: rotation is estimated by ResNet-50, translation is computed with the pinhole camera model (not learned).
- **Main Classes/Functions:**
  - `PoseEstimatorBaseline`: PyTorch module that predicts only rotation (quaternion)
  - `create_pose_estimator_baseline`: factory function to instantiate and configure the model
- **Key Dependencies:**
  - `torch`, `torchvision`, `config.Config`, `utils.pinhole` (for translation)

### `pose_estimator_endtoend.py`
- **Purpose:**
  - Implements an end-to-end RGB model for 6D pose estimation: both rotation (quaternion) and translation are learned by ResNet-50.
- **Main Classes/Functions:**
  - `PoseEstimator`: PyTorch module that predicts rotation and translation
  - `create_pose_estimator`: factory function to instantiate and configure the model
- **Key Dependencies:**
  - `torch`, `torchvision`, `config.Config`

### `pose_estimator_RGBD.py`
- **Purpose:**
  - Implements a multi-modal RGB-D fusion architecture for 6D pose estimation, combining RGB features (ResNet-18), depth features (DepthEncoder), and metadata features (MetaEncoder) for joint rotation and translation prediction.
- **Main Classes/Functions:**
  - `RGBDFusionModel`: complete RGB-D fusion model with three parallel encoders and a pose regressor
  - `build_crop_meta`: utility function to construct metadata tensors from bounding boxes and camera intrinsics
- **Key Dependencies:**
  - `torch`, `torchvision`, `models.depth_encoder`, `models.meta_encoder`, `models.pose_regressor`

### `depth_encoder.py`
- **Purpose:**
  - Implements a lightweight CNN encoder for extracting features from single-channel depth images, inspired by DenseFusion but simplified for 2D depth crops.
- **Main Classes/Functions:**
  - `DepthEncoder`: PyTorch module for depth feature extraction
- **Key Dependencies:**
  - `torch`, `torch.nn`

### `meta_encoder.py`
- **Purpose:**
  - Encodes scalar metadata (bounding box, camera intrinsics) for translation disambiguation using a multi-layer perceptron.
- **Main Classes/Functions:**
  - `MetaEncoder`: PyTorch module for metadata encoding
- **Key Dependencies:**
  - `torch`, `torch.nn`

### `pose_regressor.py`
- **Purpose:**
  - MLP regressor head for final 6D pose prediction from fused features (quaternion + translation).
- **Main Classes/Functions:**
  - `PoseRegressor`: PyTorch module for pose regression
- **Key Dependencies:**
  - `torch`, `torch.nn`, `torch.nn.functional`

### `__init__.py`
- **Purpose:**
  - Makes the directory a Python module.

## 5. Usage Examples

### YOLODetector: object detection
```python
from models.yolo_detector import YOLODetector

yolo = YOLODetector(model_name='yolo11n', pretrained=True, num_classes=13)
results = yolo.predict('path/to/image.jpg')
# results: list of detected objects
```

### PoseEstimatorBaseline: rotation estimation (baseline)
```python
from models.pose_estimator_baseline import create_pose_estimator_baseline

model = create_pose_estimator_baseline(pretrained=True, freeze_backbone=False)
model.eval()
# x = batch of torch images (B, 3, H, W)
quaternion = model(x)  # (B, 4)
```

### PoseEstimator (end-to-end RGB): rotation and translation estimation
```python
from models.pose_estimator_endtoend import create_pose_estimator

model = create_pose_estimator(pretrained=True, freeze_backbone=False)
model.eval()
# x = batch of torch images (B, 3, H, W)
pred = model.predict(x)
# pred['quaternion']: (B, 4), pred['translation']: (B, 3)
```

### RGBDFusionModel: multi-modal RGB-D fusion
```python
from models.pose_estimator_RGBD import RGBDFusionModel

model = RGBDFusionModel(pretrained_rgb=True)
model.load_weights('checkpoints/pose/fusion_rgbd_512/best.pt')
model.eval()
# rgb: (B, 3, 224, 224), depth: (B, 1, 224, 224), meta: (B, 10)
pose = model(rgb, depth, meta)  # (B, 7) -> [qw, qx, qy, qz, tx, ty, tz]
```

## 6. Technical Notes and Conventions
- **Centralized configuration:** All models read default parameters from `config.Config` (device, dropout, learning rate, etc.).
- **PyTorch best practices:**
  - Use of `nn.Sequential` for backbone and regression heads
  - Quaternion normalization in output (unit norm)
  - Ability to freeze backbone for fine-tuning
- **YOLO compatibility:** The wrapper handles both pretrained and custom weights, and allows export to various formats (ONNX, TorchScript, etc.).
- **Model Comparison:**
  - *Baseline*: only rotation learned, translation computed geometrically (pinhole)
  - *End-to-End RGB*: both rotation and translation learned by the model
  - *RGB-D Fusion*: multi-modal architecture leveraging RGB, depth, and metadata for improved pose estimation
- **Examples and workflow:** See also the notebooks in the `notebooks/` directory and comments in the modules for complete pipelines.

---

For details on training, validation, and pipelines, refer to the documentation in individual files and example notebooks.
