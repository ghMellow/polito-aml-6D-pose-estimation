# Cartella `notebooks/` – Esempi, Pipeline e Sperimentazione

## 1. Panoramica
Questa cartella raccoglie tutti i notebook Jupyter di esempio, validazione e sperimentazione per la pipeline di stima della posa 6D. I notebook coprono l’intero workflow: dal fine-tuning di YOLO, al training dei modelli di pose, fino alla valutazione e visualizzazione dei risultati. Sono pensati sia per la riproducibilità degli esperimenti che per l’esplorazione interattiva del progetto.

## 2. Struttura della cartella
- `pipeline_baseline.ipynb` — Pipeline completa baseline: YOLO → Pinhole → ResNet (rotazione)
- `pipeline_endtoend.ipynb` — Pipeline completa end-to-end: YOLO → ResNet (rotazione+traslazione)
- `finetuning_Yolo.ipynb` — Fine-tuning e validazione YOLO su LineMOD
- `finetuning_ResNet50_rotation_only.ipynb` — Training e validazione modello baseline (solo rotazione)
- `finetuning_ResNet50_endtoend.ipynb` — Training e validazione modello end-to-end (rotazione+traslazione)
- `Enhancing_6DPose_Estimation.ipynb` — Notebook di esplorazione e idee avanzate
- `colab_training.ipynb` — Esempio di training su Google Colab

## 3. Componenti/Notebook principali

### `pipeline_baseline.ipynb`
- **Cosa fa:** Esegue la pipeline classica: rilevamento oggetti con YOLO, stima traslazione con modello pinhole, stima rotazione con ResNet-50.
- **Step principali:**
  1. Caricamento modelli pre-addestrati
  2. Estrazione batch di test (con bbox GT o YOLO)
  3. Calcolo e visualizzazione di rotazione e traslazione
  4. Valutazione metrica ADD e visualizzazione risultati per classe
- **Dipendenze chiave:** `models/`, `utils/`, `dataset/`, `config.py`

### `pipeline_endtoend.ipynb`
- **Cosa fa:** Esegue la pipeline end-to-end: YOLO per detection, ResNet-50 per stima congiunta di rotazione e traslazione.
- **Step principali:**
  1. Caricamento modelli pre-addestrati
  2. Estrazione batch di test (con bbox GT o YOLO)
  3. Calcolo e visualizzazione di rotazione e traslazione
  4. Valutazione metrica ADD e visualizzazione risultati per classe
- **Dipendenze chiave:** `models/`, `utils/`, `dataset/`, `config.py`

### `finetuning_Yolo.ipynb`
- **Cosa fa:** Mostra come effettuare il fine-tuning di YOLO su LineMOD, organizzare i risultati e valutare le performance.
- **Step principali:**
  1. Caricamento modello YOLO pre-addestrato
  2. Preparazione dataset e data.yaml
  3. Fine-tuning della detection head
  4. Valutazione su test set e visualizzazione predizioni
- **Dipendenze chiave:** `models.yolo_detector`, `utils.prepare_yolo_symlinks`, `config.py`

### `finetuning_ResNet50_rotation_only.ipynb`
- **Cosa fa:** Allena e valuta il modello baseline (solo rotazione) su LineMOD.
- **Step principali:**
  1. Caricamento e visualizzazione dataset
  2. Inizializzazione modello `PoseEstimatorBaseline`
  3. Training e salvataggio checkpoint
  4. Valutazione su test set (ADD rotazione)
  5. Visualizzazione metriche e grafici
- **Dipendenze chiave:** `models.pose_estimator_baseline`, `utils/`, `config.py`

### `finetuning_ResNet50_endtoend.ipynb`
- **Cosa fa:** Allena e valuta il modello end-to-end (rotazione+traslazione) su LineMOD.
- **Step principali:**
  1. Caricamento e visualizzazione dataset
  2. Inizializzazione modello `PoseEstimator`
  3. Training e salvataggio checkpoint
  4. Valutazione su test set (ADD full pose)
  5. Visualizzazione metriche e grafici
- **Dipendenze chiave:** `models.pose_estimator_endtoend`, `utils/`, `config.py`

## 4. Utilizzo: Esempi pratici

### Eseguire la pipeline baseline
```python
# In pipeline_baseline.ipynb
from models.yolo_detector import YOLODetector
from models.pose_estimator_baseline import PoseEstimatorBaseline
from utils.pinhole import compute_translation_pinhole

yolo = YOLODetector(model_name='yolo11n.pt', num_classes=13)
model = PoseEstimatorBaseline(pretrained=True)
# ...carica batch, esegui detection, stima rotazione e traslazione...
```

### Fine-tuning YOLO
```python
# In finetuning_Yolo.ipynb
detector = YOLODetector(model_name='yolo11n', pretrained=True, num_classes=13)
detector.freeze_backbone(freeze_until_layer=10)
results = detector.train(data_yaml='path/to/data.yaml', epochs=20)
```

### Training modello baseline (rotazione)
```python
# In finetuning_ResNet50_rotation_only.ipynb
from models.pose_estimator_baseline import PoseEstimatorBaseline
model = PoseEstimatorBaseline(pretrained=True)
# ...setup optimizer, loss, train loop...
```

## 5. Note tecniche e convenzioni
- **Struttura modulare:** Ogni notebook si concentra su una fase specifica (fine-tuning, training, pipeline, valutazione).
- **Riproducibilità:** I notebook salvano checkpoint, risultati intermedi e grafici per facilitare la ripetizione degli esperimenti.
- **Configurazione centralizzata:** Tutti i parametri chiave sono gestiti tramite `config.py`.
- **Visualizzazione:** Ampio uso di grafici, tabelle e immagini per interpretare i risultati.
- **Best practice:** Uso di tqdm per progress bar, salvataggio automatico di metriche e risultati, commenti dettagliati.

---

Per dettagli su pipeline, training e valutazione, consultare i singoli notebook e la documentazione nei moduli Python.
