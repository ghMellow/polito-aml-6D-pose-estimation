"""
LineMODPoseDataset: Estensione per task pose estimation (rotazione/traslazione/crop).
"""

from .linemod_base import LineMODDatasetBase
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from config import Config
from utils.bbox_utils import crop_and_pad
from utils.transforms import rotation_matrix_to_quaternion, get_pose_transforms

class LineMODPoseDataset(LineMODDatasetBase):
    def __init__(self, dataset_root, split='train', crop_margin=None, output_size=None, **kwargs):
        super().__init__(dataset_root, split, **kwargs)
        self.crop_margin = crop_margin if crop_margin is not None else Config.POSE_CROP_MARGIN
        output_size = output_size if output_size is not None else Config.POSE_IMAGE_SIZE
        self.output_size = (output_size, output_size)
        self.transform = get_pose_transforms(train=(split == 'train'))

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        img = self.load_image(folder_id, sample_id)
        gt_objs = self.load_gt(folder_id, sample_id)
        info_objs = self.load_info(folder_id, sample_id)
        results = []
        img_array = np.array(img)
        for obj in gt_objs:
            bbox = np.array(obj['obj_bb'])
            rotation_matrix = np.array(obj['cam_R_m2c']).reshape(3, 3)
            translation = np.array(obj['cam_t_m2c']) / 1000.0
            quaternion = rotation_matrix_to_quaternion(rotation_matrix)
            rgb_crop = crop_and_pad(img_array, bbox, self.output_size, margin=self.crop_margin)
            if self.transform:
                if isinstance(rgb_crop, np.ndarray):
                    rgb_crop = Image.fromarray(rgb_crop)
                rgb_crop = self.transform(rgb_crop)
            cam_K = np.array(info_objs['cam_K']).reshape(3, 3) if 'cam_K' in info_objs else None
            obj_id = obj.get('obj_id', None)
            results.append({
                'rgb_crop': rgb_crop,
                'quaternion': torch.from_numpy(quaternion).float(),
                'translation': torch.from_numpy(translation).float(),
                'cam_K': torch.from_numpy(cam_K).float() if cam_K is not None else None,
                'bbox': torch.from_numpy(bbox).float(),
                'folder_id': folder_id,
                'sample_id': sample_id,
                'obj_id': obj_id,
                'depth_path': str(self.dataset_root / 'data' / f'{folder_id:02d}' / 'depth' / f'{sample_id:04d}.png'),
                'info_path': str(self.dataset_root / 'data' / f'{folder_id:02d}' / 'info.yml')
            })
        # Se vuoi un solo oggetto per immagine, restituisci results[0]
        return results[0] if results else None

def create_pose_dataloaders(dataset_root, batch_size, crop_margin, output_size, num_workers=0, folder_to_class_mapping=None):
    """
    Helper per creare train/val/test dataloader LineMODPoseDataset.
    Se TRAIN_TEST_RATIO è definito in Config, effettua split random su train per ottenere anche validation.
    """
    # Dataset completo train (split ufficiale)
    full_train_dataset = LineMODPoseDataset(
        dataset_root=dataset_root,
        split='train',
        crop_margin=crop_margin,
        output_size=output_size,
        folder_to_class_mapping=folder_to_class_mapping
    )
    test_dataset = LineMODPoseDataset(
        dataset_root=dataset_root,
        split='test',
        crop_margin=crop_margin,
        output_size=output_size,
        folder_to_class_mapping=folder_to_class_mapping
    )
    # Split train/val
    train_ratio = getattr(Config, 'TRAIN_TEST_RATIO', 0.8)
    train_len = int(len(full_train_dataset) * train_ratio)
    val_len = len(full_train_dataset) - train_len
    
    generator = torch.Generator().manual_seed(getattr(Config, 'RANDOM_SEED', 42))
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader