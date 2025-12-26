# Experimental Notebooks – Panoramica e Guida

## 1. Panoramica
Questa cartella raccoglie notebook Jupyter di esplorazione, test e confronto relativi al progetto di 6D pose estimation. I notebook sono pensati per:
- Analizzare dati e pipeline
- Sperimentare modelli e strategie di training
- Eseguire confronti quantitativi e qualitativi tra soluzioni
- Documentare risultati intermedi e finali

## 2. Struttura della cartella
- `test1_explore_linemod.ipynb` – Esplorazione e visualizzazione del dataset LineMOD
- `test2_yolo1_pretrained.ipynb` – Test e analisi YOLO pre-addestrato
- `test3_yolo2_finetuning.ipynb` – Fine-tuning YOLO e valutazione
- `test4_ResNet50_pose_rotation_only.ipynb` – Training/valutazione ResNet50 solo rotazione
- `test4_RestNet50_pose_finetuning.ipynb` – Fine-tuning ResNet50 per pose
- `test5_baseline_pinhole_pipeline.ipynb` – Pipeline baseline: detection + pinhole + ResNet
- `test6_extension_endtoend_pipeline.ipynb` – Pipeline estesa: end-to-end pose estimation
- `test7_baseline_vs_extension_comparison.ipynb` – Confronto quantitativo/qualitativo tra pipeline baseline ed estesa
- `README.md` – Questo file

## 3. Componenti/Moduli principali

### `test1_explore_linemod.ipynb`
- **Cosa fa:**
  - Esplora la struttura del dataset LineMOD, visualizza immagini, annotazioni, bounding box.
- **Dipendenze chiave:**
  - `matplotlib`, `PIL`, `numpy`, moduli dataset custom

### `test5_baseline_pinhole_pipeline.ipynb`
- **Cosa fa:**
  - Implementa la pipeline baseline: detection (YOLO), stima traslazione con pinhole, stima rotazione con ResNet.
- **Dipendenze chiave:**
  - `torch`, `models.pose_estimator_baseline`, `models.yolo_detector`, `utils.pinhole`, `utils.transforms`

### `test6_extension_endtoend_pipeline.ipynb`
- **Cosa fa:**
  - Implementa la pipeline estesa: modello end-to-end che stima rotazione e traslazione direttamente.
- **Dipendenze chiave:**
  - `torch`, `models.pose_estimator_endtoend`, `models.yolo_detector`, `utils.transforms`

### `test7_baseline_vs_extension_comparison.ipynb`
- **Cosa fa:**
  - Esegue un confronto quantitativo e qualitativo tra pipeline baseline e pipeline estesa, con metriche, grafici e visualizzazioni side-by-side.
- **Dipendenze chiave:**
  - `torch`, `matplotlib`, `pandas`, `models`, `utils`, `scipy.spatial.transform`, `PIL`, `yaml`

## 4. Utilizzo – Esempi pratici

### Esecuzione di un notebook (da terminale):
```bash
jupyter notebook test7_baseline_vs_extension_comparison.ipynb
```

### Esempio di import e uso di modelli nei notebook:
```python
from models.pose_estimator_baseline import PoseEstimatorBaseline
from models.yolo_detector import YOLODetector
from utils.pinhole import compute_translation_pinhole

# Carica modello
model = PoseEstimatorBaseline(pretrained=True)
# ...
```

### Esempio di visualizzazione risultati:
```python
import matplotlib.pyplot as plt
plt.imshow(image)
plt.title('Prediction vs Ground Truth')
plt.show()
```

## 5. Note tecniche
- **Pattern:**
  - Ogni notebook è pensato come esperimento indipendente e documentato.
  - Uso esteso di visualizzazioni (grafici, immagini, tabelle) per analisi qualitativa e quantitativa.
  - Modularità: i notebook importano moduli dal progetto (models, utils, dataset) per evitare duplicazione di codice.
- **Convenzioni:**
  - I notebook seguono una numerazione progressiva e titoli descrittivi.
  - Le pipeline sono suddivise in step chiari: setup, caricamento modelli, test, analisi risultati.
- **Dettagli:**
  - I notebook di confronto (es. test7) includono sia metriche aggregate che visualizzazioni per singola immagine.
  - Le dipendenze sono coerenti con il resto del progetto e richiedono l'ambiente Python configurato.

---

Per dettagli su modelli e utility, vedi le cartelle `models/`, `utils/`, `dataset/` del progetto.
