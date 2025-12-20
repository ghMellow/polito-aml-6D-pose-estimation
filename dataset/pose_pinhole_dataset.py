"""
PosePinholeDataset: Estensione di PoseDataset per pipeline pinhole e deep
Restituisce anche il path della depth image e altri campi utili per pipeline geometriche e deep learning.
"""

from .custom_dataset import PoseDataset
from pathlib import Path
from config import Config

class PosePinholeDataset(PoseDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        folder_id = sample['folder_id']
        sample_id = sample['sample_id']
        # Usa i path centralizzati dal Config
        linemod_root = Path(Config.LINEMOD_ROOT)
        base_path = linemod_root / 'data' / f"{folder_id:02d}"
        depth_path = base_path / 'depth' / f"{sample_id:04d}.png"
        rgb_path = base_path / 'rgb' / f"{sample_id:04d}.png"
        sample['depth_path'] = str(depth_path)
        sample['rgb_path'] = str(rgb_path)
        return sample
