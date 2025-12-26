# Utils – Utility Functions for 6D Pose Estimation

## 1. Panoramica

Questa cartella raccoglie tutte le utility condivise e i moduli di supporto per il progetto di 6D Pose Estimation su LineMOD. Qui sono centralizzate funzioni comuni, trasformazioni, metriche, visualizzazione, gestione dei dati e script di supporto per training, validazione e preparazione dataset. L'obiettivo è evitare duplicazione di codice e fornire strumenti riutilizzabili per pipeline di training, validazione e analisi.

## 2. Struttura della Cartella

- `bbox_utils.py` – Utility per bounding box (crop, conversioni formato, padding)
- `download_dataset.py` – Script per scaricare e scompattare il dataset LineMOD pre-processato
- `losses.py` – Funzioni di loss per la stima della posa (rotazione, traslazione, combinata)
- `metrics.py` – Metriche ADD/ADD-S per valutazione modelli di pose
- `organize_yolo_results.py` – Organizzazione automatica dei risultati YOLO in sottocartelle
- `pinhole.py` – Modello pinhole per calcolo traslazione 3D da bbox e depth
- `prepare_yolo_symlinks.py` – Preparazione dataset YOLO tramite symlink (o copia)
- `training.py` – Loop di training centralizzati per modelli di pose
- `transforms.py` – Trasformazioni su immagini, quaternioni, augmentation
- `validation.py` – Pipeline di validazione e funzioni di analisi risultati
- `visualization.py` – Utility di visualizzazione per training, metriche, sample

## 3. Componenti/Moduli Principali

### `bbox_utils.py`
- **Cosa fa**: Gestisce operazioni comuni sui bounding box: crop centrato, padding, conversioni formato (es. YOLO).
- **Funzioni principali**:
	- `crop_and_pad(img, bbox, output_size, margin)` – Crop quadrato centrato sul bbox con padding opzionale.
	- `convert_bbox_to_yolo_format(bbox, img_width, img_height)` – Conversione bbox in formato YOLO normalizzato.
	- `crop_bbox_optimized(img, bbox_xyxy, margin, output_size)` – Crop ottimizzato da bbox in formato xyxy.
- **Dipendenze**: `numpy`, `cv2`

### `download_dataset.py`
- **Cosa fa**: Scarica e scompatta il dataset LineMOD pre-processato da Google Drive.
- **Funzioni principali**:
	- `download_linemod_dataset(output_dir)` – Scarica ed estrae il dataset nella cartella desiderata.
- **Dipendenze**: `gdown`, `zipfile`, `os`, `pathlib`, `config.py`
- **Esempio d'uso**:
	```bash
	python utils/download_dataset.py --output_dir ./data
	```

### `losses.py`
- **Cosa fa**: Implementa le funzioni di loss per la stima della posa (rotazione, traslazione, combinata).
- **Classi principali**:
	- `PoseLoss` – Loss combinata (rotazione + traslazione) con pesi configurabili.
	- `PoseLossBaseline` – Loss solo per la rotazione (baseline, traslazione non appresa).
- **Dipendenze**: `torch`, `config.py`
- **Pattern**: Uso di Smooth L1 per la traslazione, distanza geodetica su quaternioni per la rotazione.

### `metrics.py`
- **Cosa fa**: Calcola metriche di valutazione ADD/ADD-S per la 6D pose (sia batch che per sample), carica modelli 3D e info oggetti.
- **Funzioni principali**:
	- `compute_add_batch_rotation_only`, `compute_add_batch_full_pose`, `compute_add_batch`, `compute_add_batch_gpu` – Calcolo ADD per diversi scenari.
	- `load_models_info`, `load_3d_model`, `load_all_models` – Caricamento info e mesh 3D.
- **Dipendenze**: `numpy`, `torch`, `config.py`

### `organize_yolo_results.py`
- **Cosa fa**: Organizza i risultati YOLO (grafici, sample, pesi) in sottocartelle ordinate.
- **Funzioni principali**:
	- `organize_yolo_output(run_dir, destination_dir)` – Sposta e ordina i file di output.
	- `print_organization_summary(project_dir, stats)` – Stampa riepilogo organizzazione.
- **Dipendenze**: `pathlib`, `shutil`, `config.py`

### `pinhole.py`
- **Cosa fa**: Implementa il modello pinhole per calcolare la traslazione 3D da bbox, depth map e intrinseci camera.
- **Funzioni principali**:
	- `load_camera_intrinsics(gt_yml_path)` – Carica parametri intrinseci da info.yml.
	- `compute_translation_pinhole(bbox, depth_path, camera_intrinsics, ...)` – Calcola la traslazione 3D.
	- `compute_translation_pinhole_batch(...)` – Versione batch.
- **Dipendenze**: `numpy`, `PIL`, `yaml`, `config.py`

### `prepare_yolo_symlinks.py`
- **Cosa fa**: Prepara la struttura del dataset in formato YOLO usando symlink (o copia file se non supportato).
- **Funzioni principali**:
	- `prepare_yolo_dataset_symlinks(dataset_root, output_root, use_symlinks)` – Crea symlink/copie e label YOLO.
	- `create_data_yaml(output_root, dataset_root)` – Genera file data.yaml per YOLO.
- **Dipendenze**: `pathlib`, `tqdm`, `PIL`, `config.py`, `bbox_utils.py`

### `training.py`
- **Cosa fa**: Centralizza i loop di training per modelli di pose (baseline e end-to-end).
- **Funzioni principali**:
	- `train_pose_baseline(...)` – Training per modelli che predicono solo la rotazione.
	- `train_pose_full(...)` – Training per modelli che predicono rotazione e traslazione.
- **Dipendenze**: `torch`, `tqdm`, `config.py`

### `transforms.py`
- **Cosa fa**: Trasformazioni su immagini e pose, conversioni tra rappresentazioni di rotazione, augmentation.
- **Funzioni principali**:
	- `rotation_matrix_to_quaternion`, `quaternion_to_rotation_matrix_batch` – Conversioni tra rotazione e quaternione.
	- `crop_image_from_bbox` – Crop immagini da bbox.
	- `get_pose_transforms` – Compose di trasformazioni per training/val.
- **Dipendenze**: `torch`, `numpy`, `PIL`, `config.py`

### `validation.py`
- **Cosa fa**: Pipeline di validazione per modelli di pose e pipeline complete (YOLO+ResNet), calcolo metriche e salvataggio risultati.
- **Funzioni principali**:
	- `run_pinhole_deep_pipeline`, `run_deep_pose_pipeline`, `run_yolo_baseline_pipeline`, `run_yolo_endtoend_pipeline` – Pipeline di validazione.
	- `save_validation_results`, `load_validation_results` – Gestione risultati.
	- `calc_add_accuracy_per_class`, `calc_pinhole_error_per_class` – Analisi metriche per classe.
- **Dipendenze**: `torch`, `config.py`, moduli utils

### `visualization.py`
- **Cosa fa**: Utility per visualizzazione di training, metriche, sample, curve di loss e ADD.
- **Funzioni principali**:
	- `plot_training_curves`, `plot_training_validation_loss_from_csv` – Plot delle curve di training.
	- `show_per_class_table`, `plot_add_per_class`, `plot_pinhole_error_per_class` – Visualizzazione metriche per classe.
	- `show_pose_samples`, `show_pose_samples_with_add` – Visualizzazione immagini e predizioni.
- **Dipendenze**: `matplotlib`, `pandas`, `torch`, `config.py`

## 4. Utilizzo – Esempi Pratici

### Esempio: Crop e conversione bbox
```python
from utils.bbox_utils import crop_and_pad, convert_bbox_to_yolo_format
img_crop = crop_and_pad(img, bbox=[x, y, w, h], output_size=(224, 224), margin=0.1)
yolo_bbox = convert_bbox_to_yolo_format([x, y, w, h], img_width=640, img_height=480)
```

### Esempio: Calcolo traslazione con pinhole
```python
from utils.pinhole import compute_translation_pinhole, load_camera_intrinsics
intrinsics = load_camera_intrinsics('data/01/gt.yml')
t = compute_translation_pinhole([x1, y1, x2, y2], 'data/01/depth/0000.png', intrinsics)
```

### Esempio: Training loop
```python
from utils.training import train_pose_baseline
history, best_loss, best_epoch = train_pose_baseline(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50)
```

### Esempio: Calcolo metrica ADD
```python
from utils.metrics import compute_add_batch_rotation_only
results = compute_add_batch_rotation_only(pred_R_batch, gt_R_batch, obj_ids, models_dict, models_info)
```

### Esempio: Visualizzazione
```python
from utils.visualization import plot_training_curves
plot_training_curves(history)
```

## 5. Note Tecniche e Convenzioni

- **Pattern**: Tutti i moduli sono pensati per essere importabili e riutilizzabili in pipeline diverse.
- **Config**: Molti moduli dipendono da `config.py` per parametri globali (path, pesi, mapping classi, device, ecc.).
- **Batch e GPU**: Le funzioni di metriche e loss sono ottimizzate per batch e supportano GPU.
- **Formati**: Le funzioni accettano sia `numpy` che `torch` dove possibile, e convertono internamente.
- **Visualizzazione**: Le funzioni di plotting usano `matplotlib` e `pandas` per tabelle e grafici.
- **Dataset**: La preparazione del dataset YOLO usa symlink per efficienza e risparmio spazio.
- **Documentazione**: Ogni file e funzione principale è documentata con docstring dettagliate.

---

Per dettagli aggiuntivi, consultare i docstring nei singoli file o la documentazione principale del progetto.
