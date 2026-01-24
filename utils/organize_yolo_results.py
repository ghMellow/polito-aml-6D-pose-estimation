"""
Utility per organizzare i risultati YOLO in sottocartelle.
Mantiene pulita la directory principale spostando grafici e samples.
"""

from pathlib import Path
import shutil

def _move_files(run_dir: Path, patterns: list[str], destination: Path, stats: dict, key: str) -> None:
    for pattern in patterns:
        for file in run_dir.glob(pattern):
            if not file.is_file():
                continue
            dest = destination / file.name
            try:
                if dest.exists():
                    dest.unlink()
                shutil.move(str(file), str(dest))
                stats[key] += 1
            except Exception:
                stats["skipped"] += 1


def _move_remaining(run_dir: Path, plots_dir: Path, validation_dir: Path, stats: dict) -> None:
    for file in run_dir.iterdir():
        if not file.is_file():
            continue
        if file.suffix == ".png":
            dest = plots_dir / file.name
            stats_key = "plots"
        else:
            dest = validation_dir / file.name
            stats_key = "validation_samples"
        try:
            if dest.exists():
                dest.unlink()
            shutil.move(str(file), str(dest))
            stats[stats_key] += 1
        except Exception:
            stats["skipped"] += 1


def _cleanup_run_dir(run_dir: Path) -> None:
    try:
        shutil.rmtree(run_dir)
    except Exception:
        return
    parent = run_dir.parent
    while parent != parent.parent and parent.name in {"detect", "runs"}:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def organize_yolo_output(run_dir: Path, destination_dir: Path) -> dict:
    """Organizza gli output YOLO in sottocartelle strutturate."""
    if not destination_dir.exists():
        raise FileNotFoundError(f"Directory non trovata: {destination_dir}")

    plots_dir = destination_dir / "plots"
    training_dir = destination_dir / "training_samples"
    validation_dir = destination_dir / "validation_samples"

    plots_dir.mkdir(exist_ok=True)
    training_dir.mkdir(exist_ok=True)
    validation_dir.mkdir(exist_ok=True)

    stats = {
        "plots": 0,
        "training_samples": 0,
        "validation_samples": 0,
        "skipped": 0,
    }

    plot_patterns = [
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
    ]

    _move_files(run_dir, plot_patterns, plots_dir, stats, "plots")
    _move_files(run_dir, ["train_batch*.jpg", "labels.jpg"], training_dir, stats, "training_samples")
    _move_files(run_dir, ["val_batch*.jpg"], validation_dir, stats, "validation_samples")

    if run_dir.exists() and run_dir.is_dir():
        _move_remaining(run_dir, plots_dir, validation_dir, stats)
        _cleanup_run_dir(run_dir)

    return stats


def print_organization_summary(project_dir: Path, stats: dict):
    """Stampa riepilogo dell'organizzazione."""
    print(f"\nRisultati organizzati in: {project_dir.name}/")
    print(f"   Grafici: {stats['plots']} file → plots/")
    print(f"   Training samples: {stats['training_samples']} file → training_samples/")
    print(f"   Validation samples: {stats['validation_samples']} file → validation_samples/")
    print("   Weights: weights/")
    print("   Config: args.yaml")
    print("   Metriche: results.csv")
    if stats['skipped'] > 0:
        print(f"   File saltati: {stats['skipped']}")


