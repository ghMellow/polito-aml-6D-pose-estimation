# Dataset Module

Questo modulo contiene le implementazioni dei dataset PyTorch per il caricamento dei dati LineMOD.

## File

### `__init__.py`
File di inizializzazione del modulo Python che rende la directory un package importabile.

### `custom_dataset.py`
Implementazione del dataset PyTorch per la stima della posa 6D degli oggetti LineMOD.

**Caratteristiche:**
- Carica immagini RGB-D con annotazioni complete (bounding box, maschere, pose 6D)
- Supporta split train/test automatico con stratificazione per oggetto
- Include trasformazioni di immagine configurabili
- Gestisce intrinseci della camera e rotazioni/traslazioni
- Utilizzato per il training del modello di pose estimation (PoseEstimator)

**Classe principale:** `CustomDataset(Dataset)`

### `linemod_yolo_dataset.py`
Dataset PyTorch ottimizzato per il training di YOLO su LineMOD.

**Caratteristiche:**
- Legge direttamente dalla struttura preprocessata LineMOD senza duplicazione
- Utilizza gli split ufficiali train.txt/test.txt
- Restituisce immagini complete con bounding box in formato YOLO normalizzato
- Più efficiente per il training del detector (nessuna copia di file)
- Utilizzato specificamente per il fine-tuning di yolo

**Classe principale:** `LinemodYOLODataset(Dataset)`

## Differenze tra i Dataset

| Caratteristica | `custom_dataset.py` | `linemod_yolo_dataset.py` |
|---------------|---------------------|---------------------------|
| **Scopo** | Training pose estimation (6D pose) | Training object detection (YOLO) |
| **Output** | RGB-D + mask + quaternione + traslazione | RGB + bounding box normalizzati |
| **Split** | Automatico con train_ratio | Usa train.txt/test.txt ufficiali |
| **Formato bbox** | Pixel assoluti | YOLO normalizzato (0-1) |
| **Modello target** | PoseEstimator (ResNet-50) | yolo detector |
| **Gestione depth** | ✅ Include depth map | ❌ Solo RGB |
| **Efficienza** | Caricamento standard | Ottimizzato, no duplicazione |

## Uso

```python
# Per pose estimation
from dataset.custom_dataset import CustomDataset
train_dataset = CustomDataset(dataset_root='data/Linemod_preprocessed', split='train')

# Per YOLO detection
from dataset.linemod_yolo_dataset import LinemodYOLODataset
yolo_dataset = LinemodYOLODataset(dataset_root='data/Linemod_preprocessed', split='train')
```

## Campi restituiti da `__getitem__`

- **CustomDataset:** struttura (tuple o dict) contenente tipicamente:
	- `rgb`: PIL Image o `Tensor` (H,W,3) — immagine RGB
	- `depth`: `numpy` array o `Tensor` (H,W) — mappa di profondità
	- `mask`: `numpy` array o `Tensor` (H,W) — maschera binaria dell'oggetto
	- `bbox`: list o array `[x_min, y_min, x_max, y_max]` (pixel) — bounding box in pixel
	- `pose`: dict con `rotation` (quaternion o matrice 3x3) e `translation` (3,) — posa 6D
	- `cam_K`: lista o array (3x3) — matrice delle intrinseche
	- `meta`: dict — campi addizionali (`folder_id`, `sample_id`, `obj_idx`, ecc.)

- **PoseDataset:** versione focalizzata su singolo oggetto / crop (usata per la rete di pose). Tipicamente restituisce:
	- `crop`: PIL Image o `Tensor` — immagine ritagliata in ingresso alla rete
	- `mask_crop`: `Tensor` — maschera corrispondente al crop
	- `bbox_crop`: bbox nel sistema di coordinate del crop
	- `rotation`, `translation`, `cam_K`, `meta`

- **LinemodYOLODataset:** ottimizzato per training di YOLO, ritorna:
	- `image`: PIL Image — immagine completa
	- `bboxes`: lista di `[class_id, x_center, y_center, width, height]` (valori normalizzati 0-1)
	- `class_ids`: lista di interi (ridondante se i `class_id` sono inclusi in `bboxes`)
	- `meta`: dict con `folder_id`, `sample_id`

Le tipologie sopra sono indicative: per il formato esatto controllare l'implementazione di `__getitem__` nelle rispettive classi.

