"""
visualize_cluster_assignments_2.py

Create UMAP visualizations for a trained Architecture_2 VQ-VAE model.

Works for both:
- PBMC
- Arabidopsis / Ath

Change only DATASET_NAME in the CONFIG section.

Important:
Architecture_2 uses Negative Binomial reconstruction loss.
Therefore, this script should use the same raw-count h5ad file used during
training and cluster assignment extraction, not normalized/log1p data.

This script produces:
1. Expression UMAP colored by VQ-VAE cluster
2. Latent UMAP colored by VQ-VAE cluster
3. Latent UMAP colored by biological annotation, if annotations are available
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from torch.utils.data import DataLoader, TensorDataset
from Architecture_2 import VQVAE
from load_data_split_2 import project_path, PROJECT_ROOT, get_count_matrix



# CONFIG

DATASET_NAME = "PBMC"   # change to "PBMC" when needed

DATASET_CONFIGS = {
    "PBMC": {
        "h5ad_path": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
        "checkpoint_path": project_path("PBMC/vqvae_runs/train_val_test_2/best_vqvae_2_checkpoint.pth"),
        "assignments_csv": project_path("PBMC/vqvae_runs/train_val_test_2/barcode_cluster_assignments_final_optuna_architecture_2.csv"),
        "annotation_path": project_path("PBMC/68k_pbmc_barcodes_annotation.tsv"),
        "annotation_has_header": True,
        "annotation_sep": "\t",
        "annotation_names": None,
        "annotation_barcode_col": "barcodes",
        "annotation_cell_type_col": "celltype",
        "annotation_label": "PBMC cell type annotation",
        "output_dir": project_path("PBMC/vqvae_runs/train_val_test_2/cluster_visualization"),
    },
    "Ath": {
        "h5ad_path": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
        "checkpoint_path": project_path("Ath/vqvae_runs/train_val_test_2/best_vqvae_2_checkpoint.pth"),
        "assignments_csv": project_path("Ath/vqvae_runs/train_val_test_2/barcode_cluster_assignments_final_optuna_architecture_2.csv"),
        "annotation_path": project_path("Ath/Shahan_annotation.csv"),
        "annotation_has_header": True,
        "annotation_sep": ",",
        "annotation_names": None,
        "annotation_barcode_col": "cell_barcode_id",
        "annotation_cell_type_col": "celltype.anno",
        "annotation_label": "Shahan cell type annotation",
        "output_dir": project_path("Ath/vqvae_runs/train_val_test_2/cluster_visualization"),
    },
}

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

CONFIG = DATASET_CONFIGS[DATASET_NAME]

H5AD_PATH = CONFIG["h5ad_path"]
CHECKPOINT_PATH = CONFIG["checkpoint_path"]
ASSIGNMENTS_CSV = CONFIG["assignments_csv"]

ANNOTATION_PATH = CONFIG["annotation_path"]
ANNOTATION_HAS_HEADER = CONFIG["annotation_has_header"]
ANNOTATION_SEP = CONFIG["annotation_sep"]
ANNOTATION_NAMES = CONFIG["annotation_names"]
ANNOTATION_BARCODE_COL = CONFIG["annotation_barcode_col"]
ANNOTATION_CELL_TYPE_COL = CONFIG["annotation_cell_type_col"]
ANNOTATION_LABEL = CONFIG["annotation_label"]

OUTPUT_DIR = CONFIG["output_dir"]

BARCODE_COL = "barcode"
CLUSTER_COL = "cluster_index"

N_PCS = 30
N_NEIGHBORS = 15
RANDOM_STATE = 42
BATCH_SIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# BASIC HELPERS

def check_required_columns(df, required_cols, df_name):
    """
    Check that a DataFrame contains the required columns.
    """

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def get_dense_matrix(adata):
    """
    Convert the model's selected raw count matrix to a dense float32 NumPy array.

    AnnData matrices are often sparse. PyTorch needs a dense tensor here.
    """

    X = get_count_matrix(adata)

    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X, dtype=np.float32)


def check_count_matrix(X_sample):
    """
    Print a small diagnostic to check whether the matrix looks like raw counts.

    Architecture_2 was trained with Negative Binomial loss, so input values
    should look count-like, not log-normalized.
    """

    min_value = float(np.min(X_sample))
    max_value = float(np.max(X_sample))
    integer_like = np.allclose(X_sample, np.round(X_sample))

    print("\nCount-matrix diagnostic:")
    print(f"Sample min value: {min_value}")
    print(f"Sample max value: {max_value}")
    print(f"Sample values look integer-like: {integer_like}")

    if not integer_like:
        print(
            "\nWARNING:"
            "\nThe values do not look integer-like."
            "\nArchitecture_2 uses Negative Binomial loss, so this script should use"
            "\nraw counts, not normalized/log1p values."
        )



# CLUSTER ASSIGNMENTS


def load_assignments(assignments_csv):
    """
    Load the VQ-VAE barcode-to-cluster assignment table.
    """

    assignments_df = pd.read_csv(assignments_csv)

    check_required_columns(
        assignments_df,
        [BARCODE_COL, CLUSTER_COL],
        "Assignments CSV",
    )

    assignments_df[BARCODE_COL] = assignments_df[BARCODE_COL].astype(str).str.strip()
    assignments_df[CLUSTER_COL] = assignments_df[CLUSTER_COL].astype(str)

    return assignments_df


def attach_cluster_assignments(adata, assignments_df):
    """
    Attach VQ-VAE cluster labels to adata.obs using barcode matching.
    """

    adata = adata.copy()

    adata.obs[BARCODE_COL] = adata.obs_names.astype(str)

    cluster_map = assignments_df.set_index(BARCODE_COL)[CLUSTER_COL]
    adata.obs[CLUSTER_COL] = adata.obs[BARCODE_COL].map(cluster_map)

    missing_assignments = adata.obs[CLUSTER_COL].isna().sum()

    if missing_assignments > 0:
        raise ValueError(
            f"{missing_assignments} cells are missing cluster assignments after merge. "
            "Check whether barcodes match between the .h5ad file and the assignments CSV."
        )

    adata.obs[CLUSTER_COL] = adata.obs[CLUSTER_COL].astype("category")

    return adata


# ANNOTATION LOADING


def load_annotation_table():
    """
    Load either the PBMC annotation table or the Ath/Shahan annotation table.
    """

    if ANNOTATION_PATH is None or not os.path.exists(ANNOTATION_PATH):
        return None

    if ANNOTATION_HAS_HEADER:
        annotation_df = pd.read_csv(
            ANNOTATION_PATH,
            sep=ANNOTATION_SEP,
            low_memory=False,
        )
    else:
        annotation_df = pd.read_csv(
            ANNOTATION_PATH,
            sep=ANNOTATION_SEP,
            header=None,
            names=ANNOTATION_NAMES,
            low_memory=False,
        )

    check_required_columns(
        annotation_df,
        [ANNOTATION_BARCODE_COL, ANNOTATION_CELL_TYPE_COL],
        "Annotation file",
    )

    return annotation_df


def get_ath_barcode_core_from_adata(obs_name):
    """
    Extract the core barcode from Ath adata.obs_names.

    Example:
        AAACCCAAGAAACACT-1_SRX8574728_cellbender_filter
    becomes:
        AAACCCAAGAAACACT
    """

    return str(obs_name).split("-1_")[0]


def get_ath_barcode_core_from_annotation(barcode):
    """
    Extract the core barcode from the Shahan annotation barcode.

    Example:
        AAACCCAAGAAACACT_sc_1
    becomes:
        AAACCCAAGAAACACT
    """

    return str(barcode).rsplit("_", 1)[0]


def get_pbmc_barcode(barcode):
    """
    Clean PBMC barcode string.
    """

    return str(barcode).strip()


def attach_cell_type_annotations(adata):
    """
    Attach biological annotations to adata.obs.

    For PBMC:
        matches annotation barcode directly to adata.obs_names.

    For Ath:
        uses sample-aware matching:
            barcode_core + "__" + sample_key

        adata sample names are mapped using ATH_SAMPLE_MAP to match
        annotation orig.ident values.
    """

    annotation_df = load_annotation_table()

    if annotation_df is None:
        print(f"Annotation file not found or not set: {ANNOTATION_PATH}")
        print("Skipping annotation-colored latent UMAP.")
        return adata, None

    print(f"Loading annotation file: {ANNOTATION_PATH}")
    print(f"Using barcode column: {ANNOTATION_BARCODE_COL}")
    print(f"Using label column: {ANNOTATION_CELL_TYPE_COL}")

    adata = adata.copy()

    if DATASET_NAME == "Ath":
        check_required_columns(
            adata.obs,
            ["sample"],
            "AnnData obs",
        )

        check_required_columns(
            annotation_df,
            ["orig.ident"],
            "Ath annotation file",
        )

        annotation_df["barcode_core"] = (
            annotation_df[ANNOTATION_BARCODE_COL]
            .astype(str)
            .map(get_ath_barcode_core_from_annotation)
        )

        annotation_df["sample_key"] = annotation_df["orig.ident"].astype(str)

        annotation_df["merge_key"] = (
            annotation_df["barcode_core"] +
            "__" +
            annotation_df["sample_key"]
        )

        adata.obs["barcode_core"] = (
            adata.obs_names.astype(str)
            .map(get_ath_barcode_core_from_adata)
        )

        adata.obs["sample_key"] = (
            adata.obs["sample"]
            .astype(str)
            .map(ATH_SAMPLE_MAP)
        )

        missing_sample_map = adata.obs["sample_key"].isna().sum()

        if missing_sample_map > 0:
            print(
                f"Warning: {missing_sample_map} cells have samples not found in ATH_SAMPLE_MAP."
            )

        adata.obs["merge_key"] = (
            adata.obs["barcode_core"] +
            "__" +
            adata.obs["sample_key"].astype(str)
        )

        annotation_map = annotation_df.set_index("merge_key")[ANNOTATION_CELL_TYPE_COL]

        adata.obs["cell_type_annotation"] = (
            adata.obs["merge_key"].map(annotation_map)
        )

    else:
        annotation_df[ANNOTATION_BARCODE_COL] = (
            annotation_df[ANNOTATION_BARCODE_COL]
            .astype(str)
            .map(get_pbmc_barcode)
        )

        annotation_map = annotation_df.set_index(ANNOTATION_BARCODE_COL)[
            ANNOTATION_CELL_TYPE_COL
        ]

        adata.obs["barcode_for_annotation"] = (
            adata.obs_names.astype(str)
            .map(get_pbmc_barcode)
        )

        adata.obs["cell_type_annotation"] = (
            adata.obs["barcode_for_annotation"].map(annotation_map)
        )

    missing_annotations = adata.obs["cell_type_annotation"].isna().sum()
    matched_annotations = adata.n_obs - missing_annotations

    print(f"Matched annotations: {matched_annotations}/{adata.n_obs}")
    print(f"Missing annotations: {missing_annotations}")

    if matched_annotations == 0:
        print("No annotation barcodes matched. Skipping annotation-colored UMAP.")
        return adata, None

    adata.obs["cell_type_annotation"] = (
        adata.obs["cell_type_annotation"]
        .fillna("Unknown")
        .astype("category")
    )

    return adata, "cell_type_annotation"


# MODEL LOADING FOR ARCHITECTURE_2


def load_model_from_checkpoint():
    """
    Rebuild Architecture_2 VQ-VAE from checkpoint hyperparameters and load weights.

    train_model_2.py saves:
        checkpoint["hyperparameters"]
        checkpoint["model_state_dict"]
    """

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if "hyperparameters" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'hyperparameters'. "
            "This script expects a checkpoint saved by train_model_2.py."
        )

    hparams = checkpoint["hyperparameters"]

    required_keys = [
        "input_dim",
        "hidden_dim",
        "latent_dim",
        "num_embeddings",
        "commitment_cost",
    ]

    missing = [key for key in required_keys if key not in hparams]

    if missing:
        raise KeyError(f"Missing hyperparameters in checkpoint: {missing}")

    print("\nLoaded checkpoint hyperparameters:")
    for key in required_keys:
        print(f"{key}: {hparams[key]}")

    model = VQVAE(
        input_dim=hparams["input_dim"],
        hidden_dim=hparams["hidden_dim"],
        latent_dim=hparams["latent_dim"],
        num_embeddings=hparams["num_embeddings"],
        commitment_cost=hparams["commitment_cost"],
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model



# LATENT EXTRACTION


def extract_encoder_latent_vectors(model, adata):
    """
    Extract continuous encoder latent vectors, z_e, from Architecture_2.

    Architecture_2 forward returns:
        mu, theta, vq_loss, encoding_indices

    But for latent UMAP, we only need:
        z_e = model.encoder(x)
    """

    print("Extracting Architecture_2 encoder latent vectors...")

    X = get_dense_matrix(adata)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    latent_batches = []

    with torch.no_grad():
        for batch_idx, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(DEVICE)

            z_e = model.encoder(batch_x)

            if isinstance(z_e, tuple) or isinstance(z_e, list):
                z_e = z_e[0]

            if z_e.dim() > 2:
                z_e = z_e.view(z_e.size(0), -1)

            latent_batches.append(z_e.cpu().numpy())

            if batch_idx % 100 == 0:
                processed = min((batch_idx + 1) * BATCH_SIZE, adata.n_obs)
                print(f"Processed {processed}/{adata.n_obs} cells")

    latent_vectors = np.vstack(latent_batches)

    print(f"Latent vector shape: {latent_vectors.shape}")

    return latent_vectors


# UMAP FUNCTIONS


def compute_expression_umap(adata):
    """
    Compute expression-space UMAP from adata.X.

    Note:
    For Architecture_2, adata.X is raw counts. This is consistent with the
    model input, but the expression-space UMAP may be strongly affected by
    library size. The latent-space UMAP is usually the more relevant plot for
    checking the learned VQ-VAE representation.
    """

    print("Computing expression-space PCA...")
    sc.tl.pca(
        adata,
        svd_solver="arpack",
        random_state=RANDOM_STATE,
    )

    print("Computing expression-space neighbor graph...")
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        random_state=RANDOM_STATE,
    )

    print("Computing expression-space UMAP...")
    sc.tl.umap(
        adata,
        random_state=RANDOM_STATE,
    )

    return adata


def compute_latent_umap(latent_vectors, obs):
    """
    Compute UMAP directly from Architecture_2 encoder latent vectors.
    """

    latent_adata = sc.AnnData(
        X=latent_vectors,
        obs=obs.copy(),
    )

    print("Computing latent-space neighbor graph...")
    sc.pp.neighbors(
        latent_adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X",
        random_state=RANDOM_STATE,
    )

    print("Computing latent-space UMAP...")
    sc.tl.umap(
        latent_adata,
        random_state=RANDOM_STATE,
    )

    return latent_adata


def save_umap(adata, color_col, title, output_path):
    """
    Save a UMAP plot.
    """

    print(f"Saving: {output_path}")

    fig = sc.pl.umap(
        adata,
        color=color_col,
        title=title,
        show=False,
        return_fig=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"H5AD path: {H5AD_PATH}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Assignments CSV: {ASSIGNMENTS_CSV}")
    print(f"Annotation path: {ANNOTATION_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")

    if not os.path.exists(ASSIGNMENTS_CSV):
        raise FileNotFoundError(f"Could not find assignments CSV: {ASSIGNMENTS_CSV}")

    run_tag = "final_optuna_architecture_2"

    print(f"Using assignments file: {ASSIGNMENTS_CSV}")
    print(f"Run tag: {run_tag}")

    # Load raw-count AnnData

    print(f"Loading raw-count {DATASET_NAME} AnnData...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"AnnData shape: {adata.shape}")

    X_sample = get_dense_matrix(adata[: min(1000, adata.n_obs)])
    check_count_matrix(X_sample)


    # Attach VQ-VAE cluster assignments

    assignments_df = load_assignments(ASSIGNMENTS_CSV)
    adata = attach_cluster_assignments(adata, assignments_df)


    # Attach biological annotations

    adata, cell_type_col = attach_cell_type_annotations(adata)


    # 1. Expression UMAP by VQ-VAE cluster
    expression_adata = adata.copy()
    expression_adata = compute_expression_umap(expression_adata)

    expression_vq_output = os.path.join(
        OUTPUT_DIR,
        f"expression_umap_by_vq_cluster_{run_tag}.png",
    )

    save_umap(
        expression_adata,
        color_col=CLUSTER_COL,
        title=f"{DATASET_NAME} expression-space UMAP colored by Architecture_2 VQ-VAE cluster",
        output_path=expression_vq_output,
    )

    # 2. Latent UMAP by VQ-VAE cluster
    print("Loading trained Architecture_2 checkpoint...")
    model = load_model_from_checkpoint()

    latent_vectors = extract_encoder_latent_vectors(model, adata)

    latent_adata = compute_latent_umap(
        latent_vectors=latent_vectors,
        obs=adata.obs,
    )

    latent_vq_output = os.path.join(
        OUTPUT_DIR,
        f"latent_umap_by_vq_cluster_{run_tag}.png",
    )

    save_umap(
        latent_adata,
        color_col=CLUSTER_COL,
        title=f"{DATASET_NAME} Architecture_2 latent-space UMAP colored by VQ cluster",
        output_path=latent_vq_output,
    )

    # 3. Latent UMAP by biological annotation

    if cell_type_col is not None:
        latent_celltype_output = os.path.join(
            OUTPUT_DIR,
            f"latent_umap_by_annotation_{run_tag}.png",
        )

        # Keep Unknown in adata.obs, but remove it only for annotation-colored UMAP.
        known_latent_adata = latent_adata[
            latent_adata.obs[cell_type_col] != "Unknown"
        ].copy()

        save_umap(
            known_latent_adata,
            color_col=cell_type_col,
            title=f"Architecture_2 latent-space UMAP colored by {ANNOTATION_LABEL}",
            output_path=latent_celltype_output,
        )


    # Summary
    print("\nSaved visualizations to:")
    print(f"1. {expression_vq_output}")
    print(f"2. {latent_vq_output}")

    if cell_type_col is not None:
        print(f"3. {latent_celltype_output}")
    else:
        print("3. Annotation-colored latent UMAP skipped.")

    print("\nVQ cluster counts:")
    print(adata.obs[CLUSTER_COL].value_counts().sort_index())

    if cell_type_col is not None:
        print(f"\n{ANNOTATION_LABEL} counts:")
        print(adata.obs[cell_type_col].value_counts())


if __name__ == "__main__":
    main()