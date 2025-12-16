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

def organize_yolo_output(project_dir: Path) -> dict:
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
        project_dir: Path alla directory del progetto YOLO (es. checkpoints/yolo/run1/)
    
    Returns:
        dict: Statistiche dei file spostati
    """
    project_dir = Path(project_dir)
    
    if not project_dir.exists():
        raise FileNotFoundError(f"Directory non trovata: {project_dir}")
    
    # Crea sottocartelle
    plots_dir = project_dir / 'plots'
    training_dir = project_dir / 'training_samples'
    validation_dir = project_dir / 'validation_samples'
    
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
        for file in project_dir.glob(pattern):
            if file.is_file():
                try:
                    shutil.move(str(file), str(plots_dir / file.name))
                    stats['plots'] += 1
                except Exception as e:
                    print(f"⚠️  Errore spostando {file.name}: {e}")
                    stats['skipped'] += 1
    
    # Sposta training samples
    for pattern in training_patterns:
        for file in project_dir.glob(pattern):
            if file.is_file():
                try:
                    shutil.move(str(file), str(training_dir / file.name))
                    stats['training_samples'] += 1
                except Exception as e:
                    print(f"⚠️  Errore spostando {file.name}: {e}")
                    stats['skipped'] += 1
    
    # Sposta validation samples
    for pattern in validation_patterns:
        for file in project_dir.glob(pattern):
            if file.is_file():
                try:
                    shutil.move(str(file), str(validation_dir / file.name))
                    stats['validation_samples'] += 1
                except Exception as e:
                    print(f"⚠️  Errore spostando {file.name}: {e}")
                    stats['skipped'] += 1
    
    # Clean old folder
    #clean_old_directory()

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

def clean_old_directory():
    # ----------------------
    # Pulizia cartelle temporanee create da YOLO (runs/.../val)
    # ----------------------
    import shutil
    from pathlib import Path
    
    cleanup_paths = [
        Path.cwd() / 'runs' / 'detect' / 'val',
        Path.cwd() / 'runs' / 'val',
        Config.PROJECT_ROOT / 'runs' / 'detect' / 'val',
        Config.PROJECT_ROOT / 'runs' / 'val',
    ]
    
    removed = []
    for p in cleanup_paths:
        try:
            if p.exists():
                shutil.rmtree(p)
                removed.append(str(p))
        except Exception as e:
            print(f"⚠️ Errore rimuovendo {p}: {e}")
    
    if removed:
        print(f"🧹 Rimosse cartelle temporanee: {', '.join(removed)}")
    else:
        print("🧹 Nessuna cartella temporanea trovata da rimuovere")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python organize_yolo_results.py <project_dir>")
        sys.exit(1)
    
    project_dir = Path(sys.argv[1])
    stats = organize_yolo_output(project_dir)
    print_organization_summary(project_dir, stats)
