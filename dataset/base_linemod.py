"""
Base LineMOD Dataset Class

Shared functionality for LineMOD dataset implementations.
Eliminates code duplication between CustomDataset and LinemodYOLODataset.
"""

import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from config import Config


class BaseLineMODDataset(Dataset):
    """
    Base class with shared LineMOD dataset logic.
    
    Provides common methods for:
    - Loading ground truth annotations (gt.yml)
    - Loading camera intrinsics (info.yml)
    - Collecting samples from dataset structure
    - Folder ID to class ID mapping
    
    Args:
        dataset_root (str or Path): Path to Linemod_preprocessed directory
        folder_to_class_mapping (dict, optional): Custom folder_id -> class_id mapping
    """
    
    def __init__(self, dataset_root, folder_to_class_mapping=None):
        self.dataset_root = Path(dataset_root)
        
        # Default mapping from Config if not provided
        if folder_to_class_mapping is None:
            self.folder_to_class = Config.FOLDER_ID_TO_CLASS_ID
        else:
            self.folder_to_class = folder_to_class_mapping
    
    def load_gt_yaml(self, folder_id):
        """
        Load ground truth YAML file for a specific object folder.
        
        Args:
            folder_id (int): Object folder ID (e.g., 1, 2, 4, ...)
            
        Returns:
            dict: Ground truth data with sample_id as keys
        """
        gt_path = self.dataset_root / 'data' / f'{folder_id:02d}' / 'gt.yml'
        
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        with open(gt_path, 'r') as f:
            gt_data = yaml.safe_load(f)
        
        return gt_data
    
    def load_camera_intrinsics(self, folder_id):
        """
        Load camera intrinsics from info.yml for a specific object folder.
        
        Args:
            folder_id (int): Object folder ID
            
        Returns:
            dict: Camera info data with sample_id as keys
        """
        info_path = self.dataset_root / 'data' / f'{folder_id:02d}' / 'info.yml'
        
        if not info_path.exists():
            raise FileNotFoundError(f"Camera info file not found: {info_path}")
        
        with open(info_path, 'r') as f:
            info_data = yaml.safe_load(f)
        
        return info_data
    
    def _collect_samples(self):
        """
        Collect all available samples from the dataset.
        
        Returns:
            list: List of (folder_id, sample_id) tuples
        """
        samples = []
        data_dir = self.dataset_root / 'data'
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Iterate through all object folders (01, 02, 04, ...)
        for folder in sorted(data_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            folder_id = int(folder.name)
            
            # Skip folders not in mapping (e.g., 03, 07 which are missing)
            if folder_id not in self.folder_to_class:
                continue
            
            rgb_dir = folder / 'rgb'
            if not rgb_dir.exists():
                continue
            
            # Get all RGB images
            rgb_files = sorted(rgb_dir.glob('*.png'))
            for rgb_file in rgb_files:
                sample_id = int(rgb_file.stem)
                samples.append((folder_id, sample_id))
        
        return samples
    
    def get_image_path(self, folder_id, sample_id):
        """
        Get path to RGB image.
        
        Args:
            folder_id (int): Object folder ID
            sample_id (int): Sample ID
            
        Returns:
            Path: Path to RGB image
        """
        return self.dataset_root / 'data' / f'{folder_id:02d}' / 'rgb' / f'{sample_id:04d}.png'
    
    def get_depth_path(self, folder_id, sample_id):
        """
        Get path to depth image.
        
        Args:
            folder_id (int): Object folder ID
            sample_id (int): Sample ID
            
        Returns:
            Path: Path to depth image
        """
        return self.dataset_root / 'data' / f'{folder_id:02d}' / 'depth' / f'{sample_id:04d}.png'
    
    def get_mask_path(self, folder_id, sample_id, obj_idx=0):
        """
        Get path to segmentation mask.
        
        Args:
            folder_id (int): Object folder ID
            sample_id (int): Sample ID
            obj_idx (int): Object index (for multiple objects, default: 0)
            
        Returns:
            Path: Path to mask image
        """
        return self.dataset_root / 'data' / f'{folder_id:02d}' / 'mask' / f'{sample_id:04d}_{obj_idx:06d}.png'
    
    def extract_gt_for_sample(self, gt_data, sample_id):
        """
        Extract ground truth annotations for a specific sample.
        
        Args:
            gt_data (dict): Full ground truth data from gt.yml
            sample_id (int): Sample ID to extract
            
        Returns:
            list: List of object annotations (bboxes, poses, etc.)
        """
        if sample_id not in gt_data:
            return []
        
        obj_list = gt_data[sample_id]
        
        # Ensure it's a list (some entries might be single dicts)
        if not isinstance(obj_list, list):
            obj_list = [obj_list]
        
        return obj_list
    
    def extract_camera_intrinsics(self, info_data, sample_id):
        """
        Extract camera intrinsics for a specific sample.
        
        Args:
            info_data (dict): Full camera info data from info.yml
            sample_id (int): Sample ID to extract
            
        Returns:
            dict: Camera intrinsics {'cam_K': [...], 'depth_scale': ...}
        """
        if sample_id not in info_data:
            return {}
        
        return info_data[sample_id]
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)
