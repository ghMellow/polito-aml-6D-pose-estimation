import torch
from pathlib import Path
from typing import Any
import urllib.request
from config import Config

# GitHub repository configuration (for future use)
GITHUB_BASE_URL = "https://github.com/ghMellow/polito-aml-6D-pose-estimation/raw/refs/heads/main"

# Model registry: defines relative paths and how to load checkpoints
MODEL_REGISTRY = {
    "fusion_rgbd_512": {
        "path": Config.CHECKPOINT_DIR / "pose" / "fusion_rgbd_512" / "best.pt",
        "has_wrapper": False,
    },
    "pose_rgb_baseline": {
        "path": Config.CHECKPOINT_DIR / "pose" / "pose_rgb_baseline" / "weights" / "best.pt",
        "has_wrapper": False,  # Direct checkpoint (state_dict)
    },
    "pose_rgb_endtoend": {
        "path": Config.CHECKPOINT_DIR / "pose" / "pose_rgb_endtoend" / "weights" / "best.pt",
        "has_wrapper": False,  # Direct checkpoint (state_dict)
    },
    "yolo_train20": {
        "path": Config.CHECKPOINT_DIR / "yolo" / "yolo_train20" / "weights" / "best.pt",
        "has_wrapper": False,
    },
}

def download_checkpoint_from_github(model_name: str, dest_path: Path):
    """
    Download the checkpoint file from GitHub to the given destination path.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    # Build the relative path for the checkpoint file
    rel_path = MODEL_REGISTRY[model_name]["path"].relative_to(Config.CHECKPOINT_DIR)
    url = f"{GITHUB_BASE_URL}/checkpoints/{rel_path.as_posix()}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading checkpoint for {model_name} from {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded to {dest_path}")


def is_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer instead of the actual binary file."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read(200)
        return content.startswith(b'version https://git-lfs.github.com/spec/v1')
    except Exception:
        return False


def ensure_model_available(model_name: str, device: str = 'cpu', check_exists_only: bool = False) -> Path:
    """
    Ensure that a checkpoint is available locally.
    
    Args:
        model_name: Model name (must be in MODEL_REGISTRY)
        device: PyTorch device (not used, kept for compatibility)
        check_exists_only: If True, skip validation (used for YOLO)
    
    Returns:
        Path to the checkpoint
    
    Raises:
        FileNotFoundError: If the checkpoint does not exist or is a LFS pointer
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    checkpoint_path = MODEL_REGISTRY[model_name]["path"]
    
    # Check existence, try download if missing
    if not checkpoint_path.exists():
        try:
            download_checkpoint_from_github(model_name, checkpoint_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Checkpoint not found and download failed: {checkpoint_path}\n"
                f"Error: {e}\n"
                f"Download the checkpoints from GitHub or run training."
            )
    
    # Check if it is a LFS pointer
    if is_lfs_pointer(checkpoint_path):
        try:
            download_checkpoint_from_github(model_name, checkpoint_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Checkpoint not found and download failed: {checkpoint_path}\n"
                f"Error: {e}\n"
                f"Download the checkpoints from GitHub or run training."
            )
    
    print(f"\nCheckpoint found: {checkpoint_path}")
    return checkpoint_path


def load_model_checkpoint(model_name: str, model_instance: Any, device: str = 'cpu') -> Any:
    """
    Load a checkpoint into a model.
    
    Args:
        model_name: Model name (must be in MODEL_REGISTRY)
        model_instance: Model instance to load weights into
        device: PyTorch device ('cpu', 'cuda', etc.)
    
    Returns:
        Model with loaded weights
    
    Example:
        model = PoseEstimator(pretrained=True)
        model = load_model_checkpoint("pose_rgb_endtoend", model, device='cuda')
    """
    checkpoint_path = ensure_model_available(model_name, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load weights into model
    if MODEL_REGISTRY[model_name]["has_wrapper"]:
        # Checkpoint with wrapper (contains 'model_state_dict')
        model_instance.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct checkpoint (already a state_dict)
        model_instance.load_state_dict(checkpoint)
    
    print(f"Weights loaded successfully for {model_name}")
    return model_instance


if __name__ == "__main__":
    print("Model Loader - Registered Models")
    print("-" * 50)
    for model_name, config in MODEL_REGISTRY.items():
        exists = "OK" if config["path"].exists() else "MISSING"
        print(f"{exists} {model_name}: {config['path']}")

