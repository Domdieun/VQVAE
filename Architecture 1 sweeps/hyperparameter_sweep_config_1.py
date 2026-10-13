"""
hyperparameter_sweep_config_1.py

Shared configuration for a full 5 x 5 Architecture_1 factorial experiment.

Every codebook size is tested with every training batch size:

    num_embeddings = [10, 11, 12, 13, 14]
    batch_size      = [32, 64, 128, 256, 512]

This creates exactly 25 unique training runs. Existing completed run folders
are reused when SKIP_COMPLETED_RUNS=True, so converting the earlier nine-run
experiment to this grid requires only the 16 missing combinations to train.
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

# Change only this value when switching datasets.
DATASET_NAME = "PBMC"  # "Ath" or "PBMC"

# Scripts are expected to be placed in:
#     VQVAE/Pipeline1_MSE/2HL_sweeps/
# This resolves the main VQVAE project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

H5AD_PATHS = {
    "PBMC": PROJECT_ROOT / "PBMC" / "PBMC_68k_filtered_normalized.h5ad",
    "Ath": PROJECT_ROOT / "Ath" / "combined_cellbender_filter_matrix_main_normalized.h5ad",
}

ANNOTATION_CONFIGS = {
    "PBMC": {
        "path": PROJECT_ROOT / "PBMC" / "68k_pbmc_barcodes_annotation.tsv",
        "has_header": True,
        "sep": "\t",
        "names": None,
        "barcode_col": "barcodes",
        "label_col": "celltype",
    },
    "Ath": {
        "path": PROJECT_ROOT / "Ath" / "Shahan_annotation.csv",
        "has_header": True,
        "sep": ",",
        "names": None,
        "barcode_col": "cell_barcode_id",
        "label_col": "celltype.anno",
    },
}

# Fixed hyperparameters inherited from train_model_1.py.
# Only num_embeddings and batch_size change across the 25-run grid.
BASE_HYPERPARAMETERS = {
    "PBMC": {
        "hidden_dim": 128,
        "latent_dim": 8,
        "commitment_cost": 1.9396462944591644,
        "learning_rate": 0.00011995141620829452,
        "weight_decay": 1.6229179584481823e-07,
    },
    "Ath": {
        "hidden_dim": 128,
        "latent_dim": 128,
        "commitment_cost": 0.7754711260109922,
        "learning_rate": 0.0029985564396042726,
        "weight_decay": 8.543860846911426e-08,
    },
}

CODEBOOK_SIZES = [10, 11, 12, 13, 14]
BATCH_SIZES = [32, 64, 128, 256, 512]

RANDOM_SEED = 42
NUM_EPOCHS = 30
PATIENCE = 8
LR_SCHEDULER_PATIENCE = 5

# When True, a run is skipped only when both its checkpoint and run_result.json
# already exist. This lets the previous nine-run experiment be extended to the
# full grid without retraining those nine completed combinations.
SKIP_COMPLETED_RUNS = True

# Inference batch size does not change the learned model.
ASSIGNMENT_BATCH_SIZE = 256
INFERENCE_BATCH_SIZE = 512

# UMAP settings.
N_PCS = 30
N_NEIGHBORS = 15
UMAP_RANDOM_STATE = 42

EXCLUDE_UNKNOWN = True
UNKNOWN_LABEL = "Unknown"

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
ANNOTATION_CONFIG = ANNOTATION_CONFIGS[DATASET_NAME]

# Keep the same root as the earlier sweep so completed subset runs are reused.
RUN_ROOT = (
    PROJECT_ROOT
    / DATASET_NAME
    / "vqvae_runs"
    / "architecture_1_hyperparameter_sweeps"
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
    """Return all 25 codebook-size x batch-size combinations."""
    experiments = []

    for num_embeddings in CODEBOOK_SIZES:
        for batch_size in BATCH_SIZES:
            run_name = f"codebook_{num_embeddings}_batch_{batch_size}"
            experiments.append({
                "run_name": run_name,
                "num_embeddings": num_embeddings,
                "batch_size": batch_size,
                "sweep_membership": ["full_factorial_grid"],
                "run_dir": RUN_ROOT / run_name,
            })

    return experiments


def expanded_sweep_entries():
    """Return one comparison entry for each of the 25 unique runs."""
    return [
        {
            "sweep_type": "full_factorial_grid",
            "num_embeddings": experiment["num_embeddings"],
            "batch_size": experiment["batch_size"],
            "run_name": experiment["run_name"],
        }
        for experiment in build_experiments()
    ]
