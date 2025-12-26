# Modulo `models/` – 6D Pose Estimation

## 1. Panoramica
Questa cartella contiene tutte le implementazioni dei modelli principali per la pipeline di stima della posa 6D. Qui sono definiti i modelli di **object detection** (YOLO) e di **6D pose estimation** (sia baseline che end-to-end), utilizzati per rilevare oggetti e stimarne posizione e orientamento nello spazio. I modelli sono progettati per essere modulari, facilmente estendibili e integrabili nella pipeline del progetto.

## 2. Struttura della cartella
- `yolo_detector.py` — Wrapper per YOLOv11 (Ultralytics) per object detection
- `pose_estimator_baseline.py` — Modello baseline: stima solo la rotazione (quaternion) con ResNet-50, traslazione calcolata geometricamente
- `pose_estimator_endtoend.py` — Modello end-to-end: stima sia rotazione (quaternion) che traslazione (vettore 3D) con ResNet-50
- `__init__.py` — Rende la cartella un modulo Python
- `README.md` — Questo file

## 3. Componenti/Moduli principali

### `yolo_detector.py`
- **Cosa fa:**
  - Fornisce una classe `YOLODetector` che incapsula la logica di caricamento, training, validazione e inferenza di modelli YOLOv11 tramite Ultralytics.
  - Permette di personalizzare il numero di classi, congelare il backbone, esportare il modello e visualizzare le predizioni.
- **Classi/Funzioni principali:**
  - `YOLODetector`: wrapper completo per YOLOv11 (inizializzazione, train, predict, validate, export, freeze_backbone, ecc.)
  - `visualize_detections`: funzione per visualizzare le predizioni su immagini
- **Dipendenze chiave:**
  - `ultralytics` (YOLO), `torch`, `numpy`, `config.Config`

### `pose_estimator_baseline.py`
- **Cosa fa:**
  - Implementa il modello baseline richiesto: la rotazione viene stimata da una ResNet-50, la traslazione è calcolata con il modello di camera pinhole (non appresa).
- **Classi/Funzioni principali:**
  - `PoseEstimatorBaseline`: modulo PyTorch che predice solo la rotazione (quaternion)
  - `create_pose_estimator_baseline`: factory function per istanziare e configurare il modello
- **Dipendenze chiave:**
  - `torch`, `torchvision`, `config.Config`, `utils.pinhole` (per la traslazione)

### `pose_estimator_endtoend.py`
- **Cosa fa:**
  - Implementa un modello end-to-end per la stima della posa 6D: sia rotazione (quaternion) che traslazione sono apprese da una ResNet-50.
- **Classi/Funzioni principali:**
  - `PoseEstimator`: modulo PyTorch che predice rotazione e traslazione
  - `create_pose_estimator`: factory function per istanziare e configurare il modello
- **Dipendenze chiave:**
  - `torch`, `torchvision`, `config.Config`

## 4. Utilizzo: Esempi pratici

### YOLODetector: rilevamento oggetti
```python
from models.yolo_detector import YOLODetector

yolo = YOLODetector(model_name='yolo11n', pretrained=True, num_classes=13)
results = yolo.predict('path/to/image.jpg')
# results: lista di oggetti rilevati
```

### PoseEstimatorBaseline: stima rotazione (baseline)
```python
from models.pose_estimator_baseline import create_pose_estimator_baseline

model = create_pose_estimator_baseline(pretrained=True, freeze_backbone=False)
model.eval()
# x = batch di immagini torch (B, 3, H, W)
quaternion = model(x)  # (B, 4)
```

### PoseEstimator (end-to-end): stima rotazione e traslazione
```python
from models.pose_estimator_endtoend import create_pose_estimator

model = create_pose_estimator(pretrained=True, freeze_backbone=False)
model.eval()
# x = batch di immagini torch (B, 3, H, W)
pred = model.predict(x)
# pred['quaternion']: (B, 4), pred['translation']: (B, 3)
```

## 5. Note tecniche e convenzioni
- **Configurazione centralizzata:** Tutti i modelli leggono i parametri di default da `config.Config` (device, dropout, learning rate, ecc.).
- **PyTorch best practices:**
  - Uso di `nn.Sequential` per il backbone e le teste di regressione
  - Normalizzazione dei quaternioni in output (norma unitaria)
  - Possibilità di congelare il backbone per il fine-tuning
- **Compatibilità YOLO:** Il wrapper gestisce sia pesi pre-addestrati che custom, e permette l’esportazione in vari formati (ONNX, TorchScript, ecc.).
- **Baseline vs End-to-End:**
  - *Baseline*: solo rotazione appresa, traslazione calcolata geometricamente (pinhole)
  - *End-to-End*: sia rotazione che traslazione apprese dal modello
- **Esempi e workflow:** Vedi anche le notebook nella cartella `notebooks/` e i commenti nei moduli per pipeline complete.

---

Per dettagli su training, validazione e pipeline, consultare la documentazione nei singoli file e le notebook di esempio.
