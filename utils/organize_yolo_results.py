"""
Utility per organizzare i risultati YOLO in sottocartelle.
Mantiene pulita la directory principale spostando grafici e samples.
"""

from pathlib import Path
import shutil
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config

def organize_yolo_output(run_dir: Path, destination_dir: Path) -> dict:
    """
    Organizza i file di output YOLO in sottocartelle strutturate.
    
    Struttura creata:
    - plots/: Grafici di performance (F1, PR, confusion matrix, ecc.)
    - training_samples/: Batch samples dal training
    - validation_samples/: Batch samples dalla validation
    - weights/: Pesi del modello (già esistente)
    - args.yaml: Config del training
    - results.csv: Metriche per epoca
    
    Args:
        run_dir: Path alla directory YOLO temporanea (es. runs/detect/val)
        destination_dir: Path alla directory di destinazione YOLO (es. checkpoints/yolo/run1/)
    
    Returns:
        dict: Statistiche dei file spostati
    """    
    if not destination_dir.exists():
        raise FileNotFoundError(f"Directory non trovata: {destination_dir}")
    
    # Crea sottocartelle
    plots_dir = destination_dir / 'plots'
    training_dir = destination_dir / 'training_samples'
    validation_dir = destination_dir / 'validation_samples'
    
    plots_dir.mkdir(exist_ok=True)
    training_dir.mkdir(exist_ok=True)
    validation_dir.mkdir(exist_ok=True)
    
    stats = {
        'plots': 0,
        'training_samples': 0,
        'validation_samples': 0,
        'skipped': 0
    }
    
    # Pattern per i diversi tipi di file
    plot_patterns = [
        'BoxF1_curve.png',
        'BoxPR_curve.png', 
        'BoxP_curve.png',
        'BoxR_curve.png',
        'results.png',
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'F1_curve.png',
        'PR_curve.png',
        'P_curve.png',
        'R_curve.png'
    ]
    
    training_patterns = ['train_batch*.jpg', 'labels.jpg']
    validation_patterns = ['val_batch*.jpg']
    
    # Sposta grafici
    for pattern in plot_patterns:
        for file in run_dir.glob(pattern):
            if file.is_file():
                dest = plots_dir / file.name
                try:
                    if dest.exists():
                        dest.unlink()  # sovrascrivi se già presente
                    shutil.move(str(file), str(dest))
                    stats['plots'] += 1
                except Exception as e:
                    print(f"⚠️  Errore spostando {file.name}: {e}")
                    stats['skipped'] += 1

    # Sposta training samples
    for pattern in training_patterns:
        for file in run_dir.glob(pattern):
            if file.is_file():
                dest = training_dir / file.name
                try:
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(file), str(dest))
                    stats['training_samples'] += 1
                except Exception as e:
                    print(f"⚠️  Errore spostando {file.name}: {e}")
                    stats['skipped'] += 1

    # Sposta validation samples
    for pattern in validation_patterns:
        for file in run_dir.glob(pattern):
            if file.is_file():
                dest = validation_dir / file.name
                try:
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(file), str(dest))
                    stats['validation_samples'] += 1
                except Exception as e:
                    print(f"⚠️  Errore spostando {file.name}: {e}")
                    stats['skipped'] += 1

    # --- SPOSTA FILE DA run_dir (ex runs/detect/val) ---
    if run_dir.exists() and run_dir.is_dir():
        for file in run_dir.iterdir():
            if file.is_file():
                # Destinazione: immagini predette vanno in validation_samples, grafici in plots
                if file.name.endswith('.jpg'):
                    dest = validation_dir / file.name
                elif file.suffix == '.png':
                    dest = plots_dir / file.name
                else:
                    dest = validation_dir / file.name  # fallback
                try:
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(file), str(dest))
                    if file.name.endswith('.jpg'):
                        stats['validation_samples'] += 1
                    elif file.suffix == '.png':
                        stats['plots'] += 1
                except Exception as e:
                    print(f"⚠️  Errore spostando {file.name} da {run_dir}: {e}")
                    stats['skipped'] += 1
        # Elimina SEMPRE la cartella run_dir (anche se non vuota)
        try:
            shutil.rmtree(run_dir)
            print(f"🧹 Cartella eliminata: {run_dir}")
            # Elimina anche le cartelle padre se sono vuote (runs/detect e runs)
            parent = run_dir.parent
            while parent != parent.parent and parent.name in {"detect", "runs"}:
                try:
                    parent.rmdir()
                    print(f"🧹 Cartella eliminata: {parent}")
                except OSError:
                    # Cartella non vuota, interrompi
                    break
                parent = parent.parent
        except Exception as e:
            print(f"⚠️  Impossibile eliminare {run_dir}: {e}")

    return stats


def print_organization_summary(project_dir: Path, stats: dict):
    """Stampa riepilogo dell'organizzazione."""
    print(f"\n📁 Risultati organizzati in: {project_dir.name}/")
    print(f"   📊 Grafici: {stats['plots']} file → plots/")
    print(f"   🎓 Training samples: {stats['training_samples']} file → training_samples/")
    print(f"   ✅ Validation samples: {stats['validation_samples']} file → validation_samples/")
    print(f"   📦 Weights: weights/")
    print(f"   📄 Config: args.yaml")
    print(f"   📈 Metriche: results.csv")
    
    if stats['skipped'] > 0:
        print(f"   ⚠️  File saltati: {stats['skipped']}")


