# Utils Module

Questo modulo contiene funzioni di utilità per loss, metriche, trasformazioni e download dati.

## File

### `download_dataset.py`
Script per il download automatico del dataset LineMOD preprocessato.

**Funzionalità:**
- Download da Google Drive usando `gdown`
- Estrazione automatica dello zip
- Pulizia file temporanei
- Validazione integrità download

**Uso:**
```python
from utils.download_dataset import download_linemod_dataset
download_linemod_dataset(output_dir='./data')
```

**Output:** Dataset completo in `data/Linemod_preprocessed/`

### `losses.py`
Implementazione delle funzioni di loss per pose estimation.

**Componenti principali:**
- **`PoseLoss`**: Loss combinata per training pose estimation
  - **Translation Loss**: Smooth L1 (robusta a outlier)
  - **Rotation Loss**: Distanza geodetica su quaternioni
  - Pesi configurabili (`lambda_trans`, `lambda_rot`)

**Formula rotation loss:**
```
d = arccos(|q_pred · q_gt|)
```
Dove il dot product misura l'angolo tra quaternioni.

**Vantaggi:**
- Gestisce ambiguità quaternione (q ≡ -q)
- Metrica geometricamente corretta per rotazioni
- Numericamente stabile (clamping per arccos)

### `metrics.py`
Metriche di evaluation per pose estimation 6D.

**Metriche implementate:**
- **ADD** (Average Distance of Model Points): Distanza media dei punti 3D dopo trasformazione
- **ADD-S** (Symmetric ADD): Variante per oggetti simmetrici
- **Accuracy@threshold**: Percentuale predizioni sotto soglia

**Funzioni principali:**
- `compute_add()`: Calcola ADD per singola predizione
- `compute_add_batch()`: Batch processing per efficienza
- `load_3d_model()`: Carica mesh 3D da file PLY
- `load_models_info()`: Carica informazioni oggetti (diametro, simmetria)

**Soglia tipica:** 10% del diametro dell'oggetto (ADD < 0.1 * diameter)

### `transforms.py`
Trasformazioni e conversioni per pose estimation.

**Funzioni principali:**

**Conversioni rotazione:**
- `rotation_matrix_to_quaternion()`: Matrice 3×3 → Quaternione
- `quaternion_to_rotation_matrix()`: Quaternione → Matrice 3×3
- `axis_angle_to_rotation_matrix()`: Axis-angle → Matrice
- `rotation_matrix_to_axis_angle()`: Matrice → Axis-angle

**Image preprocessing:**
- `crop_and_resize()`: Crop da bbox + resize per CNN
- `normalize_image()`: Normalizzazione ImageNet
- `apply_affine_transform()`: Trasformazioni geometriche

**Augmentation:**
- Random crop, flip, rotation
- Color jitter (HSV)
- Gaussian noise

**Supporto:** NumPy e PyTorch tensors

## Differenze tra File

| File | Scopo Principale | Input/Output | Uso Tipico |
|------|-----------------|--------------|------------|
| `download_dataset.py` | Setup iniziale | URL → Dataset locale | Una tantum all'inizio |
| `losses.py` | Training | (pred, gt) → Loss scalare | Ogni batch di training |
| `metrics.py` | Evaluation | (pred, gt, model) → ADD score | Dopo ogni epoca/test |
| `transforms.py` | Preprocessing | Immagini/Rotazioni → Formato NN | Data loading + inference |

## Pipeline Tipica

```
1. Download dataset (download_dataset.py)
    ↓
2. Carica e preprocess (transforms.py)
    ↓
3. Training con loss (losses.py)
    ↓
4. Valutazione con metriche (metrics.py)
```

## Uso

```python
# Download dataset
from utils.download_dataset import download_linemod_dataset
download_linemod_dataset()

# Loss durante training
from utils.losses import PoseLoss
criterion = PoseLoss(lambda_trans=1.0, lambda_rot=10.0)
loss_dict = criterion(pred_q, pred_t, gt_q, gt_t)

# Evaluation
from utils.metrics import compute_add, load_3d_model
model_points = load_3d_model('data/Linemod_preprocessed/models/obj_01.ply')
add_score = compute_add(pred_R, pred_t, gt_R, gt_t, model_points, diameter)

# Conversioni rotazione
from utils.transforms import quaternion_to_rotation_matrix
R = quaternion_to_rotation_matrix(quaternion)
```

## Note

- **Loss vs Metrics:** Le loss sono differenziabili (per backprop), le metriche no
- **ADD threshold:** Tipicamente 0.1 (10% diametro), ma configurabile
- **Quaternioni:** Sempre normalizzati (unit length) per validità
- **Simmetria:** ADD-S necessaria per oggetti simmetrici (eggbox, glue)
