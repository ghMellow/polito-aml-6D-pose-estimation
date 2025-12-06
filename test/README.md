# Test Notebooks

Questa cartella contiene Jupyter Notebooks per il testing e validazione dei componenti del progetto.

## File

### `test_yolo.ipynb`
Notebook per testare il detector yolo8.

**Test inclusi:**
- Caricamento modello YOLO (pretrained/fine-tuned)
- Inferenza su singole immagini
- Validazione bounding box predictions
- Visualizzazione detection results
- Test confidence thresholds
- Performance testing (FPS, latency)

**Verifica:**
- ✅ YOLO carica correttamente pesi
- ✅ Detection funzionano su immagini LineMOD
- ✅ Bounding box sono accurati
- ✅ Classi corrette (13 oggetti LineMOD)

**Uso tipico:**
```python
from models.yolo_detector import YOLODetector
detector = YOLODetector('yolo11n.pt')
results = detector.predict(test_images)
# Visualizza results
```

### `test_pose_estimation.ipynb`
Notebook per testare il modello di pose estimation.

**Test inclusi:**
- Caricamento PoseEstimator con checkpoint
- Forward pass con immagini test
- Validazione output (quaternioni normalizzati)
- Conversione quaternioni → matrici rotazione
- Calcolo metrica ADD
- Visualizzazione pose 3D predette
- Confronto con ground truth

**Verifica:**
- ✅ Modello produce output validi (7 valori)
- ✅ Quaternioni normalizzati (||q|| = 1)
- ✅ Traslazioni in range realistico
- ✅ ADD score ragionevole
- ✅ Gradient flow (per debugging training)

**Uso tipico:**
```python
from models.pose_estimator import PoseEstimator
model = PoseEstimator(pretrained=True)
quaternion, translation = model(cropped_images)
R = quaternion_to_rotation_matrix(quaternion)
```

### `test_local_dataset.ipynb`
Notebook per validare i dataset loaders.

**Test inclusi:**
- Verifica struttura dataset LineMOD
- Test `CustomDataset` loading
- Test `LinemodYOLODataset` loading
- Validazione annotazioni (bbox, pose, mask)
- Confronto formato dati train vs test
- Statistiche dataset (dimensioni, distribuzione classi)
- Visualizzazione samples con annotazioni

**Verifica:**
- ✅ Dataset path corretti
- ✅ Train/test split consistenti
- ✅ Annotazioni valide (no NaN/Inf)
- ✅ Immagini caricano correttamente
- ✅ Formato bbox/pose corretto
- ✅ Camera intrinsics disponibili

**Uso tipico:**
```python
from dataset.custom_dataset import CustomDataset
dataset = CustomDataset('data/Linemod_preprocessed', split='train')
sample = dataset[0]
# Visualizza rgb, depth, bbox, pose
```

## Differenze tra Test Notebooks

| Notebook | Componente Testato | Input | Output Atteso | Metriche |
|----------|-------------------|-------|---------------|----------|
| `test_yolo.ipynb` | YOLODetector | Immagini RGB | Bounding boxes | mAP, Precision |
| `test_pose_estimation.ipynb` | PoseEstimator | Crop RGB | Quaternione + t | ADD, ADD-S |
| `test_local_dataset.ipynb` | Dataset loaders | File path | Batch dati | Validità dati |

## Workflow Testing Tipico

```
1. test_local_dataset.ipynb
   → Verifica dataset è OK
   ↓
2. test_yolo.ipynb
   → Testa detection pipeline
   ↓
3. test_pose_estimation.ipynb
   → Testa pose estimation pipeline
```

## Quando Usare Questi Notebook

### Prima del Training
- ✅ `test_local_dataset.ipynb`: Valida dataset
- ✅ Verifica che tutti i sample caricano

### Durante il Training
- ✅ `test_yolo.ipynb`: Monitora performance detection
- ✅ `test_pose_estimation.ipynb`: Monitora convergenza loss

### Dopo il Training
- ✅ Tutti e tre: Valuta modelli finali
- ✅ Confronta con baseline
- ✅ Identifica failure cases

### Debugging
- ✅ Quando training non converge
- ✅ Quando inferenza dà errori
- ✅ Per visualizzare predizioni vs ground truth

## Best Practices

1. **Ordine esecuzione:** Dataset → YOLO → Pose (dipendenze)
2. **Samples piccoli:** Usa pochi sample per test veloci (es. 10-50 immagini)
3. **Visualizzazioni:** Plot sempre risultati per debugging visuale
4. **Assertions:** Aggiungi `assert` per validare assunzioni
5. **Cleanup:** Libera memoria GPU dopo test (`torch.cuda.empty_cache()`)

## Uso

```bash
# Avvia Jupyter Lab
jupyter lab

# Oppure singolo notebook
jupyter notebook test/test_local_dataset.ipynb

# Esegui tutti i test in sequenza (consigliato prima training)
# 1. test_local_dataset.ipynb
# 2. test_yolo.ipynb
# 3. test_pose_estimation.ipynb
```

## Note

- Test notebook devono essere veloci (< 5 min ciascuno)
- Per test completi, usa `scripts/eval.py` (quando implementato)
- I notebook testano componenti isolati, non end-to-end pipeline
- Ideali per debugging e sviluppo iterativo
