# 6D Pose Estimation on LineMOD

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-red.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Stima della posa 6D di oggetti su dataset LineMOD tramite pipeline modulari: detection, stima rotazione/traslazione, training e valutazione modelli deep learning.

---

## 📑 Indice
- [Panoramica](#panoramica)
- [Features](#features)
- [Architettura](#architettura)
- [Getting Started](#getting-started)
- [Utilizzo](#utilizzo)
- [Struttura del Progetto](#struttura-del-progetto)
- [Documentazione](#documentazione)
- [Contributing](#contributing)
- [License](#license)
- [Contatti/Autori](#contattiautori)

---

## 🧐 Panoramica

Questo progetto fornisce una pipeline completa per la stima della posa 6D di oggetti su LineMOD, combinando detection (YOLO), stima della rotazione (ResNet-50) e traslazione (modello pinhole o end-to-end). È pensato per:
- Ricercatori e studenti in visione artificiale e robotica
- Sviluppatori interessati a pipeline modulari per 6D pose
- Chi vuole riprodurre, estendere o confrontare soluzioni su LineMOD

**Problema risolto:** stima accurata di posizione e orientamento 3D di oggetti noti in immagini RGB, con pipeline facilmente adattabile e riproducibile.

---

## 🚀 Features
- Pipeline modulari: baseline (YOLO + pinhole + ResNet) ed end-to-end (YOLO + ResNet)
- Training e fine-tuning: YOLOv11, ResNet-50 (rotazione e traslazione)
- Dataset handler: caricamento, parsing e split LineMOD, conversione in formato YOLO
- Metriche e visualizzazione: ADD/ADD-S, grafici, overlay predizioni
- Checkpoints e riproducibilità: salvataggio pesi, log, configurazioni YAML
- Notebook di esempio: training, validazione, pipeline, confronto modelli
- Configurazione centralizzata: parametri in `config.py`
- Supporto GPU/CPU/MPS: auto-detect device

---

## 🏗️ Architettura

Il progetto è suddiviso in moduli principali, ciascuno documentato da README specifici:

- **config.py**: Configurazione centralizzata (path, hyperparametri, mapping oggetti)
- **checkpoints/** ([README](checkpoints/README.md)): Checkpoint modelli, pesi, log, configurazioni
- **data/** ([README](data/README.md)): Dataset LineMOD pre-processato, annotazioni, modelli 3D, symlink YOLO
- **dataset/** ([README](dataset/README.md)): Loader, parser, conversioni e DataLoader per LineMOD
- **models/** ([README](models/README.md)): Implementazioni YOLO, ResNet-50 baseline, end-to-end
- **utils/** ([README](utils/README.md)): Utility per bbox, loss, metriche, pinhole, training, visualizzazione
- **notebooks/** ([README](notebooks/readme.md)): Notebook Jupyter per training, pipeline, validazione, analisi
- **experimental_notebooks/** ([README](experimental_notebooks/README.md)): Notebook di esplorazione e confronto

---

## ⚡ Getting Started

### Prerequisiti
- Python 3.8+
- PyTorch 1.10+
- Ultralytics YOLO (v11)
- Altri: numpy, pandas, matplotlib, pyyaml, tqdm, PIL, opencv-python, gdown

### Installazione
1. **Clona il repository:**
	```bash
	git clone https://github.com/[user]/[repo].git
	cd polito-aml-6D_pose_estimation
	```
2. **Crea un ambiente virtuale (opzionale ma consigliato):**
	```bash
	python -m venv .venv
	source .venv/bin/activate
	```
3. **Installa le dipendenze:**
	```bash
	pip install -r requirements.txt
	# oppure usa pyproject.toml/poetry se preferito
	```
4. **Scarica il dataset LineMOD pre-processato:**
	```bash
	python utils/download_dataset.py
	# oppure segui le istruzioni in data/README.md
	```

### Configurazione
- Tutti i path e parametri sono gestiti in `config.py`.
- Modifica i parametri secondo le tue esigenze (es. device, batch size, learning rate).

### Primo avvio
- Esegui uno dei notebook in `notebooks/` per pipeline, training o validazione.
- Oppure lancia uno script custom usando i moduli `models/`, `utils/`, `dataset/`.

---

## 🛠️ Utilizzo – Esempi pratici

### Pipeline baseline (YOLO → Pinhole → ResNet)
```python
from models.yolo_detector import YOLODetector
from models.pose_estimator_baseline import PoseEstimatorBaseline
from utils.pinhole import compute_translation_pinhole

yolo = YOLODetector(model_name='yolo11n.pt', num_classes=13)
model = PoseEstimatorBaseline(pretrained=True)
# ...carica batch, esegui detection, stima rotazione e traslazione...
```

### Fine-tuning YOLO
```python
from models.yolo_detector import YOLODetector
detector = YOLODetector(model_name='yolo11n', pretrained=True, num_classes=13)
detector.freeze_backbone(freeze_until_layer=10)
results = detector.train(data_yaml='path/to/data.yaml', epochs=20)
```

### Training modello baseline (rotazione)
```python
from models.pose_estimator_baseline import PoseEstimatorBaseline
model = PoseEstimatorBaseline(pretrained=True)
# ...setup optimizer, loss, train loop...
```

Altri esempi dettagliati nei notebook in [notebooks/](notebooks/readme.md).

---

## 📂 Struttura del Progetto

```
polito-aml-6D_pose_estimation/
├── config.py
├── checkpoints/         # [README](checkpoints/README.md)
├── data/                # [README](data/README.md)
├── dataset/             # [README](dataset/README.md)
├── models/              # [README](models/README.md)
├── utils/               # [README](utils/README.md)
├── notebooks/           # [README](notebooks/readme.md)
├── experimental_notebooks/ # [README](experimental_notebooks/README.md)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 📚 Documentazione
- [checkpoints/README.md](checkpoints/README.md): Checkpoint, pesi, log
- [data/README.md](data/README.md): Dataset LineMOD pre-processato
- [dataset/README.md](dataset/README.md): Loader e parser dataset
- [models/README.md](models/README.md): Modelli YOLO, ResNet-50, end-to-end
- [utils/README.md](utils/README.md): Utility, metriche, pinhole, training
- [notebooks/readme.md](notebooks/readme.md): Notebook pipeline, training, validazione
- [experimental_notebooks/README.md](experimental_notebooks/README.md): Notebook di esplorazione e confronto

---

## 🤝 Contributing

Contributi, segnalazioni di bug e proposte di miglioramento sono benvenuti! Apri una issue o una pull request seguendo le best practice di GitHub.

---

## 📝 License

Questo progetto è distribuito sotto licenza MIT. Vedi il file [LICENSE](LICENSE) per dettagli.

---

## 👤 Contatti/Autori

- [Tuo Nome] – [tuo.email@esempio.com]
- [Altri autori/contributor]

Per domande, suggerimenti o collaborazioni, non esitare a contattarci!
