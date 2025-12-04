"""
Custom Dataset for LineMOD 6D Pose Estimation

This module implements a PyTorch Dataset class for loading LineMOD dataset samples
including RGB images, depth maps, bounding boxes, masks, and 6D pose annotations.
"""

import os
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    """
    PyTorch Dataset for LineMOD 6D Pose Estimation.
    
    Loads RGB-D images along with:
    - Bounding boxes
    - Segmentation masks
    - 6D pose (rotation + translation)
    - Camera intrinsics
    
    Args:
        dataset_root (str): Path to dataset root directory
        split (str): 'train' or 'test'
        train_ratio (float): Ratio of training samples (default: 0.8)
        seed (int): Random seed for reproducibility (default: 42)
        transform: Optional transform to be applied on images
    """
    
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42, transform=None):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Define image transformations
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        # Collect all samples (folder_id, sample_id)
        self.samples = self._collect_samples()
        
        # Split into train/test
        self._split_dataset()
        
        print(f"✅ Dataset initialized: {len(self.samples)} {split} samples")
    
    def _collect_samples(self):
        """Collect all available samples from the dataset."""
        samples = []
        data_dir = self.dataset_root / 'data'
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Iterate through all object folders (01, 02, etc.)
        for folder in sorted(data_dir.iterdir()):
            if folder.is_dir():
                folder_id = int(folder.name)
                rgb_dir = folder / 'rgb'
                
                if rgb_dir.exists():
                    # Get all RGB images
                    rgb_files = sorted(rgb_dir.glob('*.png'))
                    for rgb_file in rgb_files:
                        sample_id = int(rgb_file.stem)
                        samples.append((folder_id, sample_id))
        
        return samples
    
    def _split_dataset(self):
        """Split dataset into train and test sets."""
        if len(self.samples) == 0:
            raise ValueError("No samples found in dataset!")
        
        # Stratified split by folder_id to ensure all objects in both splits
        folder_ids = [s[0] for s in self.samples]
        
        train_samples, test_samples = train_test_split(
            self.samples,
            train_size=self.train_ratio,
            random_state=self.seed,
            stratify=folder_ids
        )
        
        if self.split == 'train':
            self.samples = train_samples
        else:
            self.samples = test_samples
    
    def load_image(self, img_path):
        """Load an RGB image and convert to tensor."""
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
    
    def load_depth(self, depth_path):
        """Load depth map."""
        depth = Image.open(depth_path)
        depth = np.array(depth, dtype=np.float32)
        return torch.from_numpy(depth)
    
    def load_mask(self, mask_path):
        """Load segmentation mask."""
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.uint8)
        return torch.from_numpy(mask)
    
    def load_gt_data(self, gt_path, sample_id):
        """
        Load ground truth data from gt.yml file for a specific sample.
        
        Args:
            gt_path: Path to gt.yml file
            sample_id: Sample ID to extract from the YAML
        
        Returns:
            list: List of dicts, each containing rotation matrix, translation vector, bbox, and obj_id
        """
        with open(gt_path, 'r') as f:
            gt_data = yaml.safe_load(f)
        
        if not gt_data:
            return []
        
        # The YAML structure is: {sample_id_as_int: [list_of_objects]}
        # For example: {0: [{obj_data}], 1: [{obj_data}]}
        
        if isinstance(gt_data, dict):
            # Get the specific sample_id key
            if sample_id not in gt_data:
                print(f"⚠️ Warning: sample_id {sample_id} not found in gt.yml")
                return []
            
            obj_list = gt_data[sample_id]
            
            # obj_list should be a list of objects
            if not isinstance(obj_list, list):
                # Fallback: if not a list, wrap it
                obj_list = [obj_list]
        else:
            # If gt_data is a list directly
            obj_list = gt_data if isinstance(gt_data, list) else [gt_data]
        
        # Process ALL objects in the image
        all_objects = []
        for obj_data in obj_list:
            if not isinstance(obj_data, dict):
                continue
            
            # Extract rotation matrix (3x3)
            cam_R_m2c = np.array(obj_data['cam_R_m2c']).reshape(3, 3)
            
            # Extract translation vector (3,)
            cam_t_m2c = np.array(obj_data['cam_t_m2c'])
            
            # Extract bounding box [x, y, width, height]
            obj_bb = np.array(obj_data['obj_bb'])
            
            # Extract object ID
            obj_id = obj_data.get('obj_id', -1)
            
            all_objects.append({
                'rotation': torch.from_numpy(cam_R_m2c).float(),
                'translation': torch.from_numpy(cam_t_m2c).float(),
                'bbox': torch.from_numpy(obj_bb).float(),
                'obj_id': obj_id
            })
        
        return all_objects
    
    def load_camera_intrinsics(self, info_path, sample_id):
        """
        Load camera intrinsic parameters for a specific sample.
        
        Args:
            info_path: Path to info.yml file
            sample_id: Sample ID to extract from the YAML
        
        Returns:
            dict: Camera matrix and depth scale
        """
        with open(info_path, 'r') as f:
            info_data = yaml.safe_load(f)
        
        if not info_data:
            return None
        
        # Same parsing logic as load_gt_data
        if isinstance(info_data, dict):
            # Get the specific sample_id key
            if sample_id not in info_data:
                print(f"⚠️ Warning: sample_id {sample_id} not found in info.yml")
                return None
            
            obj_list = info_data[sample_id]
            
            # obj_list should be a list of objects
            if isinstance(obj_list, list) and len(obj_list) > 0:
                obj_info = obj_list[0]
            else:
                # Fallback: if not a list, use directly
                obj_info = obj_list
        else:
            # If info_data is a list directly
            obj_info = info_data[0] if isinstance(info_data, list) else info_data
        
        if not isinstance(obj_info, dict):
            return None
        
        # Extract camera matrix (3x3)
        cam_K = np.array(obj_info['cam_K']).reshape(3, 3)
        
        # Depth scale (if available)
        depth_scale = obj_info.get('depth_scale', 1.0)
        
        return {
            'cam_K': torch.from_numpy(cam_K).float(),
            'depth_scale': depth_scale
        }
    
    def __len__(self):
        """Return the total number of samples in the selected split."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load a dataset sample.
        
        Returns:
            dict: Contains all data for one sample:
                - rgb: RGB image tensor (3, H, W)
                - depth: Depth map tensor (H, W)
                - mask: Segmentation mask tensor (H, W)
                - rotation: Rotation matrix (3, 3)
                - translation: Translation vector (3,)
                - bbox: Bounding box [x, y, w, h] (4,)
                - cam_K: Camera intrinsic matrix (3, 3)
                - folder_id: Object class ID
                - sample_id: Sample ID
        """
        folder_id, sample_id = self.samples[idx]
        
        # Construct paths
        base_path = self.dataset_root / 'data' / f"{folder_id:02d}"
        
        img_path = base_path / 'rgb' / f"{sample_id:04d}.png"
        depth_path = base_path / 'depth' / f"{sample_id:04d}.png"
        mask_path = base_path / 'mask' / f"{sample_id:04d}.png"
        gt_path = base_path / 'gt.yml'
        info_path = base_path / 'info.yml'
        
        # Load data
        rgb = self.load_image(img_path)
        depth = self.load_depth(depth_path) if depth_path.exists() else None
        mask = self.load_mask(mask_path) if mask_path.exists() else None
        
        # Load ground truth (now returns list of all objects for this specific sample_id)
        gt_objects = self.load_gt_data(gt_path, sample_id) if gt_path.exists() else []
        
        # Load camera intrinsics for this specific sample_id
        cam_data = self.load_camera_intrinsics(info_path, sample_id) if info_path.exists() else {}
        
        # Build sample dictionary
        sample = {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "objects": gt_objects,  # List of all objects in the image
            "num_objects": len(gt_objects),
            "cam_K": cam_data.get('cam_K'),
            "depth_scale": cam_data.get('depth_scale', 1.0),
            "folder_id": folder_id,
            "sample_id": sample_id
        }
        
        return sample


def create_dataloaders(dataset_root, batch_size=8, num_workers=4, train_ratio=0.8, seed=42):
    """
    Create train and test DataLoaders.
    
    Args:
        dataset_root (str): Path to dataset root
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        train_ratio (float): Ratio of training samples
        seed (int): Random seed
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CustomDataset(dataset_root, split='train', train_ratio=train_ratio, seed=seed)
    test_dataset = CustomDataset(dataset_root, split='test', train_ratio=train_ratio, seed=seed)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n📊 DataLoaders created:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader
