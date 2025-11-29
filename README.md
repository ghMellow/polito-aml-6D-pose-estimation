# Project 6D pose estimation

Structured project template for Machine Learning/Deep Learning with PyTorch. Implements best practices for organization, reproducibility, and collaboration.

---

## 📁 Project Structure

```
polito-aml-6D_pose_estimation/
├── checkpoints/                  # 💾 MODEL CHECKPOINTS (created during training)
│   ├── .gitkeep                  # Keeps folder in git
│   ├── best_model.pth            # Best model saved automatically (gitignored)
│   └── checkpoint_epoch_N.pth    # Periodic checkpoints (gitignored)
│
├── data/                         # 📁 DATASET FILES (gitignored - download separately)
│   ├── .gitkeep
│   ├── training_set/             # Training images (cats/, dogs/)
│   └── test_set/                 # Test images (cats/, dogs/)
|
├── dataset/                      # 📦 DATASET MODULE
│   ├── __init__.py               # Exports: CustomImageDataset, create_annotations_csv
│   └── custom_dataset.py         # PyTorch Dataset class for data loading from CSV
│
├── models/                       # 🧠 MODELS MODULE
│   ├── __init__.py               # Exports: create_name_model
│   └── vgg_finetuning.py         # Model architectures
│
├── utils/                        # 🛠️ UTILITIES MODULE
│   ├── __init__.py               # Exports: transforms, visualization, metrics functions
│   ├── download_dataset.py       # DATASET DOWNLOADER (downloads dataset, e.g., from Kaggle)
│   ├── transforms.py             # Data augmentation and preprocessing (train/val/test)
│   ├── visualization.py          # Plotting and visualizations (denormalize, plot curves)
│   └── metrics.py                # Metrics computation and dataset statistics
│
├── train.py                      # 🚂 TRAINING SCRIPT (main training loop with CLI)
├── eval.py                       # 📊 EVALUATION SCRIPT (test set evaluation with CLI)
├── config.py                     # ⚙️ CONFIGURATION (hyperparameters and central settings)
│
├── colab_training.ipynb          # 📓 GOOGLE COLAB NOTEBOOK (training on Colab)
├── requirements.txt              # 📋 PYTHON DEPENDENCIES (pip install -r requirements.txt)
├── .gitignore                    # 🚫 GIT IGNORE (data/, checkpoints/*.pth, *.csv, wandb/)
│
└── README.md
```

---

## 🎯 Implemented Best Practices

✅ **Modularity**: Code split into reusable modules

✅ **CLI Interface**: Argparse for all scripts

* **What is it?** Command-Line Interface allows running scripts from the terminal by passing parameters as options (e.g., `--epochs 10 --lr 0.001`)
* **Benefits:** No need to modify the code for each experiment; all parameters are configurable from the command line
* **Implementation:** Uses Python’s `argparse` to define all available arguments (data_dir, epochs, batch_size, learning rate, etc.)

✅ **Reproducibility**: requirements.txt + config.py

* **config.py** defines default values and project constants
* **CLI arguments** allow overriding defaults without code changes
* The two approaches are complementary: config.py is the “control center,” the CLI provides flexibility for experiments

✅ **Checkpoint Management**: Auto-save best model

✅ **Logging**: Wandb integration

✅ **Documentation**: Docstrings + complete README

✅ **Git-friendly**: Proper .gitignore

✅ **Data Augmentation**: Only on train, not on val/test

✅ **Separation of Concerns**: train.py vs eval.py

---

## 🔍 For AI Assistants

**This project follows a standard modular structure:**

1. **Dataset Module** (`dataset/`): Data loading management
2. **Models Module** (`models/`): Architectures and model creation
3. **Utils Module** (`utils/`): Transforms, visualization, metrics
4. **Training Script** (`train.py`): Main training loop with CLI
5. **Eval Script** (`eval.py`): Test set evaluation
6. **Config** (`config.py`): Centralized configuration

**Key Points:**

* Each module has an `__init__.py` with explicit exports
* CLI scripts use argparse
* Training (train/validate/test functions)
* Automatic checkpoint management
* Optional but complete Wandb integration
* Transforms: AUGMENTATION only on train!

**When to suggest modifications:**

* Add new models → `models/new_model.py`
* New metrics → `utils/metrics.py`
* New datasets → `dataset/new_dataset.py`
* Training modifications → `train.py` (keep CLI style)

---

## 🤝 Contributing

To adapt this skeleton to your project:

1. **Dataset**: Modify `dataset/custom_dataset.py` for your format
2. **Model**: Add your architecture in `models/`
3. **Config**: Update `config.py` with your parameters
4. **Training**: Modify `train.py` if needed (keep CLI)
5. **Update README**: Document your changes

---

## 🚫 Git Ignore (`.gitignore`)

**What it ignores:**

* `data/` – Dataset (too large, downloaded separately)
* `checkpoints/*.pth` – Model checkpoints (too large)
* `*.csv` – Annotation files (generated automatically)
* `wandb/` – Wandb logs (synced to cloud)
* `__pycache__/` – Python cache
* `.DS_Store` – macOS files

**What it tracks:**

* Source code (`.py`)
* Configurations
* README and docs
* `.gitkeep` for empty folders

---

## 🔄 Typical Workflow

### 1. Initial Setup

```bash
git clone <repo-url>
cd polito-aml-6D_pose_estimation
pip install -r requirements.txt
python download_dataset.py
```

### 2. Training

```bash
# Feature extraction (base frozen)
python train.py --data_dir ./data --epochs 10 --freeze_base --use_wandb

# Full fine-tuning (everything trainable)
python train.py --data_dir ./data --epochs 10 --use_wandb
```

### 3. Evaluation

```bash
python eval.py --checkpoint ./checkpoints/best_model.pth --data_dir ./data
```

### 4. Experiments

```bash
# Experiment with different LR
python train.py --lr 0.001 --batch_size 64 --use_wandb

# All experiments tracked on Wandb!
```

---

## 📢 Release Information

**📅 Last update:** November 2025
**🏷️ Version:** v1.0.0 — First stable release

*For details on changes and fixes, see the changelog in the repository.*
