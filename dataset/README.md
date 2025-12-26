# Dataset Module – LineMOD Utilities

## 1. Panoramica
Questa cartella fornisce classi e utilità per la gestione, esplorazione e pre-processing del dataset [LineMOD](https://campar.in.tum.de/Chair/Projects/LineMOD), usato per task di 6D pose estimation e object detection. Qui sono implementate le logiche comuni di caricamento dati, parsing delle annotazioni, conversione in formati compatibili con pipeline di training (es. PyTorch, YOLO) e funzioni di supporto per la creazione di DataLoader.

## 2. Struttura della cartella
- `linemod_base.py` – Classe base astratta per la gestione del dataset LineMOD (caricamento immagini, annotazioni, info, ecc.)
- `linemod_pose.py` – Estende la base per task di pose estimation (rotazione, traslazione, cropping, DataLoader)
- `linemod_yolo.py` – Utility per conversione/esplorazione delle annotazioni in formato YOLO (bounding box normalizzate)
- `__init__.py` – Rende la cartella un modulo Python importabile
- `README.md` – Questo file

## 3. Componenti/Moduli principali

### `linemod_base.py`
- **Cosa fa:**
  - Definisce la classe `LineMODDatasetBase`, che eredita da `torch.utils.data.Dataset` e gestisce la logica comune di caricamento immagini, depth, annotazioni ground truth (`gt.yml`), info camera (`info.yml`) e raccolta dei sample.
- **Classi/Funzioni:**
  - `LineMODDatasetBase`: metodi per caricare immagini, depth, annotazioni, info, e per iterare sui sample.
- **Dipendenze chiave:**
  - `torch.utils.data.Dataset`, `PIL.Image`, `yaml`, `numpy`, `config.Config`

### `linemod_pose.py`
- **Cosa fa:**
  - Estende la base per task di pose estimation, aggiungendo cropping, normalizzazione, conversione rotazione/quaternion, e supporto a DataLoader PyTorch con split train/val/test.
- **Classi/Funzioni:**
  - `LineMODPoseDataset`: dataset per pose estimation, restituisce crop, pose, bbox, matrici camera, ecc.
  - `create_pose_dataloaders`: helper per creare DataLoader train/val/test con split random.
- **Dipendenze chiave:**
  - `LineMODDatasetBase`, `torch`, `PIL.Image`, `utils.bbox_utils`, `utils.transforms`, `config.Config`

### `linemod_yolo.py`
- **Cosa fa:**
  - Fornisce una classe per esplorare/convertire le annotazioni LineMOD in formato YOLO (bounding box normalizzate, class_id), utile per debug e visualizzazione.
- **Classi/Funzioni:**
  - `LineMODYOLODataset`: dataset che restituisce immagini, bbox normalizzate, classi, info sample.
- **Dipendenze chiave:**
  - `LineMODDatasetBase`, `utils.bbox_utils`

## 4. Utilizzo – Esempi pratici

### Caricamento dataset per pose estimation
```python
from dataset.linemod_pose import LineMODPoseDataset, create_pose_dataloaders

dataset_root = 'data/Linemod_preprocessed'  # o percorso custom
train_loader, val_loader, test_loader = create_pose_dataloaders(
    dataset_root=dataset_root,
    batch_size=16,
    crop_margin=30,
    output_size=128,
    num_workers=4
)
for batch in train_loader:
    rgb_crop = batch['rgb_crop']  # Tensor [B, C, H, W]
    quaternion = batch['quaternion']
    # ... training loop ...
```

### Esplorazione annotazioni in formato YOLO
```python
from dataset.linemod_yolo import LineMODYOLODataset

dataset = LineMODYOLODataset('data/Linemod_preprocessed')
sample = dataset[0]
print('Bboxes (YOLO):', sample['bboxes'])
print('Class IDs:', sample['class_ids'])
```

## 5. Note tecniche
- **Pattern:**
  - Tutte le classi dataset ereditano da `LineMODDatasetBase` per riuso e DRY.
  - Split train/val/test gestito via file `train.txt`/`test.txt` nelle sottocartelle oggetto.
  - Pre-caricamento metadati (`gt.yml`, `info.yml`) in cache per efficienza.
  - Cropping e normalizzazione immagini tramite utility dedicate (`crop_and_pad`, `get_pose_transforms`).
  - Conversione rotazione: matrice → quaternion (`rotation_matrix_to_quaternion`).
- **Convenzioni:**
  - I folder_id corrispondono agli oggetti LineMOD (01, 02, ...).
  - Le annotazioni sono lette e restituite come dict Python.
  - Le immagini sono restituite come PIL.Image o torch.Tensor a seconda del contesto.
- **Dettagli:**
  - Le classi non gestiscono direttamente la scrittura di file YOLO, ma forniscono dati già normalizzati.
  - I path alle immagini/depth sono centralizzati e compatibili con pipeline esterne.

---

Per dettagli sulle utility usate (es. `crop_and_pad`, `rotation_matrix_to_quaternion`), vedi la cartella `utils/`.
