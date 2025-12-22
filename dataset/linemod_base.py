"""
LineMODDatasetBase: Gestisce la logica comune per il dataset LineMOD.
"""

import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from config import Config

class LineMODDatasetBase(Dataset):
    def __init__(self, dataset_root, split='train', folder_to_class_mapping=None):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.folder_to_class = folder_to_class_mapping or getattr(Config, 'FOLDER_ID_TO_CLASS_ID', None)
        self._gt_cache = {}
        self._info_cache = {}
        self.samples = self._collect_samples()
        self._preload_metadata()

    def _collect_samples(self):
        data_dir = self.dataset_root / 'data'
        samples = []
        for folder in sorted(data_dir.iterdir()):
            if not folder.is_dir():
                continue
            folder_id = int(folder.name)
            split_file = folder / f"{self.split}.txt"
            if not split_file.exists():
                continue
            with open(split_file, 'r') as f:
                sample_ids = [int(line.strip()) for line in f if line.strip()]
            for sample_id in sample_ids:
                samples.append((folder_id, sample_id))
        return samples

    def _preload_metadata(self):
        folder_ids = set(folder_id for folder_id, _ in self.samples)
        for folder_id in folder_ids:
            folder_path = self.dataset_root / 'data' / f'{folder_id:02d}'
            gt_path = folder_path / 'gt.yml'
            if gt_path.exists() and folder_id not in self._gt_cache:
                with open(gt_path, 'r') as f:
                    self._gt_cache[folder_id] = yaml.safe_load(f)
            info_path = folder_path / 'info.yml'
            if info_path.exists() and folder_id not in self._info_cache:
                with open(info_path, 'r') as f:
                    self._info_cache[folder_id] = yaml.safe_load(f)

    def load_image(self, folder_id, sample_id):
        img_path = self.dataset_root / 'data' / f'{folder_id:02d}' / 'rgb' / f'{sample_id:04d}.png'
        return Image.open(img_path).convert('RGB')

    def load_depth(self, folder_id, sample_id):
        depth_path = self.dataset_root / 'data' / f'{folder_id:02d}' / 'depth' / f'{sample_id:04d}.png'
        return Image.open(depth_path)

    def load_gt(self, folder_id, sample_id):
        gt_data = self._gt_cache.get(folder_id, {})
        return gt_data.get(sample_id, [])

    def load_info(self, folder_id, sample_id):
        info_data = self._info_cache.get(folder_id, {})
        return info_data.get(sample_id, {})

    def __len__(self):
        return len(self.samples)
