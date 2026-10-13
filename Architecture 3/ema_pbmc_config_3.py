
"""
ema_pbmc_config_3.py

Shared paths and settings for Architecture_3 on the normal PBMC dataset.
"""

from pathlib import Path
import os


def guess_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
        script_dir.parent.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]

    for candidate in candidates:
        candidate = candidate.resolve()

        if (candidate / "PBMC").exists():
            return candidate

    return script_dir.resolve()


PROJECT_ROOT = Path(
    os.environ.get("VQVAE_PROJECT_ROOT", guess_project_root())
).resolve()


def project_path(relative_path: str) -> str:
    path = Path(relative_path)

    if path.is_absolute():
        return str(path)

    return str(PROJECT_ROOT / path)


DATASET_NAME = "PBMC"

H5AD_PATH = project_path(
    "PBMC/PBMC_68k_filtered_normalized.h5ad"
)

ANNOTATION_PATH = project_path(
    "PBMC/68k_pbmc_barcodes_annotation.tsv"
)

ANNOTATION_SEP = "\t"
ANNOTATION_HAS_HEADER = True
ANNOTATION_BARCODE_COL = "barcodes"
ANNOTATION_LABEL_COL = "celltype"

PIPELINE_ROOT = project_path(
    "PBMC/vqvae_runs/architecture_3"
)

OPTUNA_OUTPUT_DIR = project_path(
    "PBMC/vqvae_runs/architecture_3/optuna"
)

TRAIN_OUTPUT_DIR = project_path(
    "PBMC/vqvae_runs/architecture_3/train_val_test"
)

CHECKPOINT_PATH = project_path(
    "PBMC/vqvae_runs/architecture_3/train_val_test/"
    "best_vqvae_ema_checkpoint.pth"
)

ASSIGNMENTS_CSV = project_path(
    "PBMC/vqvae_runs/architecture_3/train_val_test/"
    "barcode_cluster_assignments_architecture_3.csv"
)

COMPARISON_OUTPUT_DIR = project_path(
    "PBMC/vqvae_runs/architecture_3/train_val_test/"
    "cluster_annotation_comparison"
)

VISUALIZATION_OUTPUT_DIR = project_path(
    "PBMC/vqvae_runs/architecture_3/train_val_test/"
    "cluster_visualization"
)

RANDOM_SEED = 42

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

ASSIGNMENT_BATCH_SIZE = 256
INFERENCE_BATCH_SIZE = 512

UNKNOWN_LABEL = "Unknown"
EXCLUDE_UNKNOWN = True
