# Models Module

Questo modulo contiene le implementazioni dei modelli neurali per object detection e pose estimation.

## File

### `__init__.py`
File di inizializzazione del modulo Python che rende la directory un package importabile.

### `pose_estimator.py`
Modello di deep learning per la stima della posa 6D degli oggetti.

**Architettura:**
- Backbone: ResNet-50 (pretrained su ImageNet)
- Output: 7 valori (4 quaternione + 3 traslazione)
- Regressione diretta della posa da immagine croppata

**Caratteristiche:**
- Normalizzazione automatica del quaternione (unit length)
- Opzione per congelare il backbone (training più veloce)
- Dropout configurabile per regolarizzazione
- Predizione separata di rotazione e traslazione

**Classe principale:** `PoseEstimator(nn.Module)`

**Metodi:**
- `forward()`: Predizione durante training
- `predict()`: Predizione con output formattato per inference

### `yolo_detector.py`
Wrapper per yolo (Ultralytics) adattato per LineMOD.

**Caratteristiche:**
- Interfaccia per modelli yolo (n/s/m/l/x variants)
- Gestione automatica del download dei pesi pretrained
- Supporto per fine-tuning con freeze del backbone
- Salvataggio dei pesi in directory personalizzata (checkpoints/)
- Compatibile con 13 classi LineMOD

**Classe principale:** `YOLODetector`

**Metodi:**
- `train()`: Fine-tuning su dataset custom
- `predict()`: Detection con confidence filtering
- `freeze_backbone()`: Congela layer per transfer learning
- `save()` / `load()`: Gestione checkpoints

## Differenze tra i Modelli

| Caratteristica | `pose_estimator.py` | `yolo_detector.py` |
|---------------|---------------------|-------------------|
| **Task** | 6D Pose Estimation | Object Detection |
| **Input** | Immagine croppata (224×224) | Immagine completa (416-640) |
| **Output** | Quaternione (4D) + Traslazione (3D) | Bounding boxes + Class IDs |
| **Backbone** | ResNet-50 (classificazione) | yolo (detection) |
| **Training** | Custom PyTorch training loop | Ultralytics API |
| **Loss** | PoseLoss (geodesic + L1) | YOLO Loss (multi-task) |
| **Metrica** | ADD/ADD-S | mAP@0.5, Precision, Recall |
| **Uso tipico** | Dopo detection, per posa precisa | Prima fase: localizzazione oggetti |

## Pipeline Completa

```
Immagine RGB
    ↓
[YOLODetector] → Bounding Boxes + Class IDs
    ↓
Crop oggetti rilevati
    ↓
[PoseEstimator] → Pose 6D (R, t) per ogni oggetto
    ↓
Posa finale
```

## Uso

```python
# Object Detection
from models.yolo_detector import YOLODetector
detector = YOLODetector(model_name='yolo11n', pretrained=True, num_classes=13)
boxes = detector.predict(images)

# Pose Estimation
from models.pose_estimator import PoseEstimator
pose_model = PoseEstimator(pretrained=True, dropout=0.3)
quaternion, translation = pose_model(cropped_images)
```
