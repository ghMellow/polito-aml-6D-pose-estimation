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
- Utilizzato specificamente per il fine-tuning di yolo8

**Classe principale:** `LinemodYOLODataset(Dataset)`

## Differenze tra i Dataset

| Caratteristica | `custom_dataset.py` | `linemod_yolo_dataset.py` |
|---------------|---------------------|---------------------------|
| **Scopo** | Training pose estimation (6D pose) | Training object detection (YOLO) |
| **Output** | RGB-D + mask + quaternione + traslazione | RGB + bounding box normalizzati |
| **Split** | Automatico con train_ratio | Usa train.txt/test.txt ufficiali |
| **Formato bbox** | Pixel assoluti | YOLO normalizzato (0-1) |
| **Modello target** | PoseEstimator (ResNet-50) | yolo8 detector |
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
