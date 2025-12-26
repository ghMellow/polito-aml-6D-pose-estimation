# Cartella `checkpoints/pose`

## 1. Panoramica

Questa cartella contiene i checkpoint dei modelli di stima della posa 6D addestrati e testati all'interno del progetto. I checkpoint rappresentano gli stati salvati dei pesi dei modelli durante o al termine del training, insieme ai file di configurazione e ai risultati delle sessioni di addestramento/validazione. La cartella è fondamentale per riprodurre esperimenti, effettuare inferenze o riprendere l'addestramento da uno stato salvato.

## 2. Struttura

La struttura della cartella è organizzata per esperimenti e versioni di training. Ogni sottocartella rappresenta un esperimento o una variante di training e contiene:

- `args.yaml`: Parametri di configurazione usati per l'esperimento.
- `results.csv`, `training_result.csv`, `validation_result.csv`: Log dei risultati di training e validazione.
- `weights/`: Directory con i file dei pesi salvati (`best.pt`, `last.pt`, ecc.).

Esempio di struttura:

```
pose_baseline_train50/
    weights/
        best.pt
        best1812.pt
pose_stable_train100/
    args.yaml
    results.csv
    weights/
        best.pt
        last.pt
test_endtoend_pose_1/
    args.yaml
    training_result.csv
    validation_result.csv
    weights/
        best.pt
        last.pt
...
```

## 3. Componenti/Moduli

### a. File di configurazione (`args.yaml`)

- **Cosa fa**: Contiene i parametri di configurazione (hyperparametri, percorsi, opzioni di training) usati per l'esperimento.
- **Dipendenze**: Viene letto dagli script di training per impostare l'esecuzione.

### b. File dei risultati (`results.csv`, `training_result.csv`, `validation_result.csv`)

- **Cosa fanno**: Salvano metriche di performance (es. loss, accuracy, ecc.) durante il training e la validazione.
- **Dipendenze**: Utili per analisi post-training e per il monitoraggio degli esperimenti.

### c. Directory `weights/` e file dei pesi (`best.pt`, `last.pt`, ecc.)

- **Cosa fanno**: Contengono i pesi del modello salvati in diversi momenti:
  - `best.pt`: Pesi del modello con la miglior performance su validation.
  - `last.pt`: Pesi del modello all'ultima epoca di training.
  - Altri file (`best1812.pt`, ecc.) possono rappresentare salvataggi intermedi o versioni specifiche.
- **Dipendenze**: Caricati dagli script di inferenza o per riprendere il training.

## 4. Utilizzo

### a. Caricamento di un modello per inferenza

Esempio in PyTorch:

```python
import torch
from models.pose_estimator_baseline import PoseEstimatorBaseline  # esempio, adattare al modello usato

# Carica la configurazione (se necessario)
# with open('checkpoints/pose/pose_stable_train100/args.yaml') as f:
#     config = yaml.safe_load(f)

# Inizializza il modello
model = PoseEstimatorBaseline()  # o il modello appropriato

# Carica i pesi
checkpoint = torch.load('checkpoints/pose/pose_stable_train100/weights/best.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### b. Ripresa del training

```python
# ...inizializzazione modello e ottimizzatore...
checkpoint = torch.load('checkpoints/pose/pose_stable_train100/weights/last.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### c. Analisi dei risultati

```python
import pandas as pd

results = pd.read_csv('checkpoints/pose/pose_stable_train100/results.csv')
print(results.head())
```

## 5. Note tecniche

- **Pattern di salvataggio**: I checkpoint sono salvati come dizionari PyTorch (`.pt`) che includono tipicamente `model_state_dict`, `optimizer_state_dict`, `epoch`, e metriche.
- **Convenzioni di naming**: 
  - Le sottocartelle seguono la convenzione `<tipo>_<descrizione>_<parametri>`, facilitando la tracciabilità degli esperimenti.
  - I file dei pesi sono denominati in base al criterio di salvataggio (`best`, `last`, ecc.).
- **Compatibilità**: I file sono pensati per essere caricati con PyTorch, ma la struttura può essere adattata ad altri framework se necessario.
- **Riproducibilità**: La presenza di `args.yaml` e dei log CSV garantisce la riproducibilità degli esperimenti.
- **Gestione versioni**: È buona pratica non versionare i file di pesi su Git, ma solo le configurazioni e i log.

---

Se vuoi aggiungere dettagli specifici sui modelli o sulle metriche salvate, fornisci i contenuti di `args.yaml` o dei CSV e posso integrare la documentazione.
