"""
LineMODYOLODataset: Classe di utilità per esplorazione/debug del dataset LineMOD.

Estende LineMODDatasetBase e aggiunge la conversione automatica delle annotazioni (bounding box e classi) in formato YOLO (bbox normalizzate, class_id), utile per:

Esplorare visivamente i dati e le annotazioni in formato YOLO
Debug e ispezione dei dati prima della conversione in file YOLO
Non viene usata direttamente nel training o inferenza con Ultralytics YOLO:

Il training/inferenza YOLO avviene su file e cartelle già in formato YOLO standard (images, labels, data.yaml)
Questa classe è pensata solo per analisi, visualizzazione e conversione dati
"""

from .linemod_base import LineMODDatasetBase
from utils.bbox_utils import convert_bbox_to_yolo_format

class LineMODYOLODataset(LineMODDatasetBase):
    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        img = self.load_image(folder_id, sample_id)
        gt_objs = self.load_gt(folder_id, sample_id)
        bboxes = []
        class_ids = []
        img_width, img_height = img.size
        class_id = self.folder_to_class[folder_id] if self.folder_to_class else folder_id
        for obj in gt_objs:
            bbox = obj['obj_bb']
            yolo_bbox = convert_bbox_to_yolo_format(bbox, img_width, img_height)
            bboxes.append(yolo_bbox)
            class_ids.append(class_id)
        return {
            'image': img,
            'bboxes': bboxes,
            'class_ids': class_ids,
            'folder_id': folder_id,
            'sample_id': sample_id,
            'num_objects': len(bboxes)
        }
