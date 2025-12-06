"""
LineMOD Dataset for YOLO Training

Custom PyTorch Dataset that reads directly from LineMOD preprocessed structure
without duplicating images. Uses official train.txt/test.txt splits.
"""

import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch

from .base_linemod import BaseLineMODDataset
from utils.bbox_utils import convert_bbox_to_yolo_format


class LinemodYOLODataset(BaseLineMODDataset):
    """
    PyTorch Dataset for YOLO training on LineMOD.
    
    Reads directly from Linemod_preprocessed structure using official train/test splits.
    Returns full images with bounding boxes in YOLO format (normalized).
    
    Args:
        dataset_root (str): Path to Linemod_preprocessed directory
        split (str): 'train' or 'test' (uses official split files)
        folder_to_class_mapping (dict): Mapping from folder_id to class_id
    """
    
    def __init__(self, dataset_root, split='train', folder_to_class_mapping=None):
        # Initialize base class
        super().__init__(dataset_root, folder_to_class_mapping)
        
        self.split = split
        
        # Collect all samples from official split files
        self.samples = self._collect_samples_with_splits()
        
        print(f"✅ LinemodYOLODataset initialized: {len(self.samples)} {split} samples")
    
    def _collect_samples_with_splits(self):
        """
        Collect samples using official train.txt/test.txt split files.
        Each entry is (folder_id, sample_id, img_path, gt_data_for_sample).
        """
        samples = []
        data_dir = self.dataset_root / 'data'
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Iterate through all object folders
        for folder in sorted(data_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            folder_id = int(folder.name)
            
            # Skip folders not in mapping (e.g., 03, 07)
            if folder_id not in self.folder_to_class:
                continue
            
            # Get split file
            split_file = folder / f"{self.split}.txt"
            if not split_file.exists():
                print(f"⚠️  Warning: {split_file} not found, skipping folder {folder_id}")
                continue
            
            # Read sample IDs from split file
            with open(split_file, 'r') as f:
                sample_ids = [int(line.strip()) for line in f if line.strip()]
            
            # Load gt.yml once for this folder
            gt_path = folder / 'gt.yml'
            if not gt_path.exists():
                continue
            
            with open(gt_path, 'r') as f:
                gt_data = yaml.safe_load(f)
            
            # For each sample_id in the split
            for sample_id in sample_ids:
                img_path = folder / 'rgb' / f"{sample_id:04d}.png"
                
                if not img_path.exists():
                    continue
                
                # Get ground truth for this sample
                if sample_id not in gt_data:
                    # No objects in this image, skip or include with empty labels
                    continue
                
                obj_list = gt_data[sample_id]
                if not isinstance(obj_list, list):
                    obj_list = [obj_list]
                
                samples.append({
                    'folder_id': folder_id,
                    'sample_id': sample_id,
                    'img_path': img_path,
                    'objects': obj_list
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load a dataset sample.
        
        Returns:
            dict: Contains:
                - image: PIL Image (RGB)
                - bboxes: List of bounding boxes in YOLO format (normalized)
                - class_ids: List of class IDs corresponding to bboxes
                - img_path: Path to image (for debugging)
        """
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['img_path']).convert('RGB')
        img_width, img_height = img.size
        
        # Get class ID for this folder
        class_id = self.folder_to_class[sample['folder_id']]
        
        # Process all objects in this image
        bboxes = []
        class_ids = []
        
        for obj in sample['objects']:
            bbox = obj['obj_bb']  # [x, y, w, h]
            
            # Convert to YOLO format using shared utility
            yolo_bbox = convert_bbox_to_yolo_format(bbox, img_width, img_height)
            
            bboxes.append(yolo_bbox)
            class_ids.append(class_id)
        
        return {
            'image': img,
            'bboxes': bboxes,  # List of [x_center, y_center, w, h] normalized
            'class_ids': class_ids,  # List of class IDs
            'img_path': str(sample['img_path'])
        }


def create_yolo_dataloaders(dataset_root, batch_size=16, num_workers=2):
    """
    Create train and validation DataLoaders for YOLO training.
    
    Args:
        dataset_root (str): Path to Linemod_preprocessed directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Create datasets (test split used as validation)
    train_dataset = LinemodYOLODataset(dataset_root, split='train')
    val_dataset = LinemodYOLODataset(dataset_root, split='test')
    
    print(f"\n📊 YOLO Datasets created:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


if __name__ == '__main__':
    # Test dataset
    from config import Config
    
    print("Testing LinemodYOLODataset...\n")
    
    # Create dataset
    dataset = LinemodYOLODataset(Config.DATA_ROOT, split='train')
    
    # Get first sample
    sample = dataset[0]
    
    print(f"\n📦 Sample 0:")
    print(f"   Image path: {sample['img_path']}")
    print(f"   Image size: {sample['image'].size}")
    print(f"   Number of objects: {len(sample['bboxes'])}")
    print(f"   Class IDs: {sample['class_ids']}")
    print(f"   Bounding boxes (YOLO format):")
    for i, bbox in enumerate(sample['bboxes']):
        print(f"      Object {i}: class={sample['class_ids'][i]}, bbox={bbox}")
