# Scripts Module

Questo modulo contiene gli script eseguibili per training, evaluation e preparazione dati.

## File

### `train_yolo.py`
Script per il fine-tuning di yolo8 sul dataset LineMOD.

**Funzionalità:**
- Fine-tuning di yolo8 (n/s/m/l/x) con pesi COCO pretrained
- Supporto per freeze del backbone (transfer learning)
- Creazione automatica di data.yaml temporaneo
- Integrazione con `LinemodYOLODataset` (no duplicazione dati)
- Training con AMP (Automatic Mixed Precision) per velocità
- Caching immagini in RAM per performance
- Early stopping e checkpoint automatici

**Parametri principali:**
- `--model`: Variante YOLO (yolo11n/s/m/l/x)
- `--epochs`: Numero di epoche
- `--batch_size`: Dimensione batch
- `--freeze_backbone`: Congela backbone per training veloce
- `--lr`: Learning rate iniziale
- `--project`: Directory output (default: `checkpoints/yolo/`)

**Output:** Modello YOLO fine-tuned in `checkpoints/yolo/`

### `train_pose.py`
Script per il training del modello di pose estimation (PoseEstimator).

**Funzionalità:**
- Training completo di PoseEstimator con backbone ResNet-50
- Ottimizzatore AdamW con scheduler CosineAnnealingLR
- Gradient accumulation per batch size effettivo più grande
- Mixed precision training (FP16) per efficienza
- Validazione con metrica ADD durante training
- Logging con Weights & Biases (opzionale)
- Salvataggio checkpoint con best model tracking

**Parametri principali:**
- `--backbone`: Architettura backbone (default: resnet50)
- `--epochs`: Numero di epoche
- `--lr`: Learning rate
- `--freeze_backbone`: Congela backbone (solo head)
- `--lambda_rot`: Peso loss rotazione
- `--lambda_trans`: Peso loss traslazione

**Output:** Modello PoseEstimator in `checkpoints/`

### `eval.py`
⚠️ **File vuoto** - Placeholder per script di evaluation futuro.

**Uso previsto:** Valutazione completa su test set con metriche ADD/ADD-S.

### `prepare_yolo_symlinks.py`
Script di utilità per preparare dataset YOLO con symbolic links.

**Funzionalità:**
- Crea struttura YOLO-format (images/labels) usando symlink
- **Veloce**: ~2 secondi vs 5+ minuti (nessuna copia)
- **Efficiente**: Risparmia ~4.5GB di spazio disco
- Converte bounding box da formato LineMOD a YOLO normalizzato
- Genera data.yaml con configurazione dataset
- Usa split ufficiali train.txt/test.txt

**Vantaggi:**
- No duplicazione file
- Modifica sul posto (modifiche visibili ovunque)
- Ideale per esperimenti rapidi

**Output:** Dataset YOLO in `data/Linemod_preprocessed/yolo_symlinks/`

## Differenze tra Script di Training

| Caratteristica | `train_yolo.py` | `train_pose.py` |
|---------------|-----------------|-----------------|
| **Modello** | yolo8 Detector | PoseEstimator (ResNet-50) |
| **Task** | Object Detection | 6D Pose Estimation |
| **Dataset** | `LinemodYOLODataset` | `CustomDataset` |
| **Framework** | Ultralytics API | PyTorch custom loop |
| **Loss** | YOLO multi-task | PoseLoss (geodesic + L1) |
| **Metrica** | mAP, Precision, Recall | ADD, ADD-S |
| **Durata training** | ~30 min (freeze backbone) | ~2-4 ore (full training) |
| **Logging** | Ultralytics built-in | Weights & Biases |
| **Output** | `checkpoints/yolo/` | `checkpoints/*.pth` |

## Workflow Tipico

```bash
# 1. Prepara dataset YOLO (opzionale, più veloce)
python scripts/prepare_yolo_symlinks.py

# 2. Fine-tune YOLO per detection
python scripts/train_yolo.py --model yolo11n --epochs 50 --freeze_backbone

# 3. Train pose estimator
python scripts/train_pose.py --epochs 100 --lr 0.001 --use_wandb

# 4. Evalua modelli (quando eval.py sarà implementato)
# python scripts/eval.py --yolo_weights checkpoints/yolo/best.pt --pose_weights checkpoints/best.pth
```

## Uso

```bash
# Training YOLO veloce (backbone congelato)
python scripts/train_yolo.py --epochs 50 --freeze_backbone --batch_size 32

# Training YOLO completo
python scripts/train_yolo.py --epochs 100 --lr 0.01

# Training Pose Estimation
python scripts/train_pose.py --epochs 100 --batch_size 16 --use_wandb

# Preparazione dataset YOLO
python scripts/prepare_yolo_symlinks.py
```
