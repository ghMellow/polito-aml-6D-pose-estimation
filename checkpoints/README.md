# Directory checkpoints

## 1. Overview

This directory contains saved checkpoints for the 6D pose estimation models used in this project. Checkpoints store model weight snapshots created during or after training, along with configuration files and training/validation logs. These files are useful for reproducing experiments, running inference, or resuming training.

## 2. Current subfolders and files

The `checkpoints/pose` directory currently contains the following subfolders and files:

- `fusion_rgbd_512/`
  - `best.pt`
- `pose_rgb_baseline/`
  - `args.yaml`
  - `validation_result.csv`
  - `weights/`
    - `best.pt`
    - `last.pt`
- `pose_rgb_endtoend/`
  - `args.yaml`
  - `training_result.csv`
  - `validation_result.csv`
  - `weights/`
    - `best.pt`
    - `last.pt`

Other checkpoint directories at the top level of `checkpoints/` (not under `pose`) include:

- `yolo/` (example runs):
  - `yolo_train10/` (args.yaml, weights/best.pt)
  - `yolo_train20/` (args.yaml, weights/best.pt, weights/last.pt)
- `pretrained/` (currently empty)

If you add new experiments, create one subfolder per experiment and include `args.yaml`, any result CSVs, and a `weights/` directory for saved `.pt` files.

## 3. Directory structure and common files

Each experiment folder typically contains:

- `args.yaml`: experiment configuration (hyperparameters, paths, training options).
- `training_result.csv`, `validation_result.csv`, or `results.csv`: logs with metrics recorded during training/validation.
- `weights/`: saved model states such as `best.pt` and `last.pt`.

Example layout:

```
pose_experiment_name/
  args.yaml
  training_result.csv
  validation_result.csv
  weights/
    best.pt
    last.pt
```

## 4. Components

- Configuration (`args.yaml`): holds hyperparameters and run settings; used by training scripts.
- Result logs (`*.csv`): record metrics like losses and accuracies during training and validation.
- `weights/` and `.pt` files: PyTorch checkpoint files that usually contain `model_state_dict`, optionally `optimizer_state_dict`, `epoch`, and metric summaries.

## 5. Usage examples

Loading a saved model for inference (PyTorch example):

```python
import torch
from models.pose_estimator_baseline import PoseEstimatorBaseline  # adapt to the actual model class

# Initialize the model
model = PoseEstimatorBaseline()

# Load checkpoint
ckpt = torch.load('checkpoints/pose/pose_rgb_baseline/weights/best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

Resuming training from a `last.pt` checkpoint:

```python
ckpt = torch.load('checkpoints/pose/pose_rgb_endtoend/weights/last.pt')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt.get('epoch', 0) + 1
```

Analyzing results using pandas:

```python
import pandas as pd
results = pd.read_csv('checkpoints/pose/pose_rgb_baseline/validation_result.csv')
print(results.head())
```

## 6. Technical notes

- Checkpoints are saved as PyTorch dictionaries (`.pt`) and commonly include `model_state_dict`, `optimizer_state_dict`, and `epoch`.
- Naming conventions: `best.pt` denotes the weights with the best validation performance; `last.pt` denotes the most recent saved weights.
- These files are intended to be loaded with PyTorch, but the general structure can be adapted to other frameworks if needed.
- For reproducibility, keep `args.yaml` and result CSVs with each experiment; avoid adding raw weight files to Git—store only configuration and small metadata.

---

If you want me to include the exact contents of any `args.yaml` or CSV (for richer documentation), tell me which experiment folder to read and I will add those details.
