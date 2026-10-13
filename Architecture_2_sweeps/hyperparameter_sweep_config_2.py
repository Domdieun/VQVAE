
"""
hyperparameter_sweep_config_2.py

Configuration for a full 5 x 5 Architecture_2 Negative-Binomial VQ-VAE sweep.

Every codebook size is paired with every training batch size:

    num_embeddings = [10, 11, 12, 13, 14]
    batch_size      = [32, 64, 128, 256, 512]

This gives exactly 25 unique training configurations.

Only num_embeddings and batch_size vary. All remaining PBMC settings are
copied from train_model_2.py.
"""

from pathlib import Path
import os


def guess_project_root() -> Path:
    """Find the VQVAE directory containing PBMC/ and Ath/."""
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
        if (candidate / "PBMC").exists() or (candidate / "Ath").exists():
            return candidate

    return script_dir.resolve()


PROJECT_ROOT = Path(
    os.environ.get("VQVAE_PROJECT_ROOT", guess_project_root())
).resolve()


def project_path(relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


DATASET_NAME = "PBMC"

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
}

ANNOTATION_CONFIGS = {
    "PBMC": {
        "path": project_path("PBMC/68k_pbmc_barcodes_annotation.tsv"),
        "has_header": True,
        "sep": "\t",
        "names": None,
        "barcode_col": "barcodes",
        "label_col": "celltype",
    },
    "Ath": {
        "path": project_path("Ath/Shahan_annotation.csv"),
        "has_header": True,
        "sep": ",",
        "names": None,
        "barcode_col": "cell_barcode_id",
        "label_col": "celltype.anno",
    },
}

# Fixed settings copied from the PBMC section of train_model_2.py.
# The grid changes only num_embeddings and batch_size.
BASE_HYPERPARAMETERS = {
    "PBMC": {
        "hidden_dim": 768,
        "latent_dim": 128,
        "commitment_cost": 1.012229629179095,
        "learning_rate": 0.0003042077888794354,
        "weight_decay": 3.3950954367462734e-09,
    },
    "Ath": {
        "hidden_dim": 256,
        "latent_dim": 96,
        "commitment_cost": 0.7049961189783545,
        "learning_rate": 0.0044749326652717926,
        "weight_decay": 7.153267862744501e-08,
    },
}

CODEBOOK_SIZES = [10, 11, 12, 13, 14]
BATCH_SIZES = [32, 64, 128, 256, 512]

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

NUM_EPOCHS = 30
PATIENCE = 8
LR_SCHEDULER_PATIENCE = 5

SKIP_COMPLETED_RUNS = True
ASSIGNMENT_BATCH_SIZE = 256
INFERENCE_BATCH_SIZE = 512

EXCLUDE_UNKNOWN = True
UNKNOWN_LABEL = "Unknown"

N_NEIGHBORS = 15
UMAP_RANDOM_STATE = 42

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
ANNOTATION_CONFIG = ANNOTATION_CONFIGS[DATASET_NAME]

RUN_ROOT = project_path(
    f"{DATASET_NAME}/vqvae_runs/"
    "architecture_2_full_factorial_sweeps_unfiltered"
)


ATH_SAMPLE_MAP = {
    "SRX8574728_cellbender_filter": "sc_1",
    "SRX8574729_cellbender_filter": "sc_9_at",
    "SRX8574730_cellbender_filter": "sc_10_at",
    "SRX8574731_cellbender_filter": "sc_11",
    "SRX8574732_cellbender_filter": "sc_12",
    "SRX8574736_cellbender_filter": "sc_30",
    "SRX8574737_cellbender_filter": "sc_31",
    "SRX8574739_cellbender_filter": "sc_37",
    "SRX8574740_cellbender_filter": "sc_40",
    "SRX8574741_cellbender_filter": "sc_51",
    "SRX8574744_cellbender_filter": "col0",
    "SRX8574745_cellbender_filter": "tnw1",
    "SRX8574746_cellbender_filter": "tnw2",
}


def build_experiments():
    experiments = []

    for num_embeddings in CODEBOOK_SIZES:
        for batch_size in BATCH_SIZES:
            run_name = f"codebook_{num_embeddings}_batch_{batch_size}"
            experiments.append({
                "run_name": run_name,
                "num_embeddings": num_embeddings,
                "batch_size": batch_size,
                "run_dir": RUN_ROOT / run_name,
            })

    return experiments
