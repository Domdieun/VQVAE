"""
visualize_cluster_assignments_1.py

Create UMAP visualizations for a trained VQ-VAE model.

Works for both:
- PBMC
- Arabidopsis / Ath

Change only DATASET_NAME in the CONFIG section.
"""

import os
from load_data_split_1 import guess_project_root as get_project_root
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from torch.utils.data import DataLoader, TensorDataset
from Architecture_1 import VQVAE


# CONFIG

# Scripts are expected to live in VQVAE/Pipeline1_MSE/2HL.
# This points back to the main VQVAE directory so files are read from/written to VQVAE/Ath and VQVAE/PBMC.
PROJECT_ROOT = Path(os.environ.get("VQVAE_PROJECT_ROOT", get_project_root())).resolve()

DATASET_NAME = os.environ.get("VQVAE_DATASET", "PBMC")

DATASET_CONFIGS = {
    "PBMC": {
        "h5ad_path": str(PROJECT_ROOT / "PBMC" / "PBMC_68k_filtered_normalized.h5ad"),
        "checkpoint_path": str(PROJECT_ROOT / "PBMC" / "vqvae_runs" / "train_val_test" / "best_vqvae_checkpoint.pth"),
        "assignments_csv": str(PROJECT_ROOT / "PBMC" / "vqvae_runs" / "train_val_test" / "barcode_cluster_assignments_final_optuna.csv"),
        "annotation_path": str(PROJECT_ROOT / "PBMC" / "68k_pbmc_barcodes_annotation.tsv"),
        "annotation_has_header": True,
        "annotation_sep": "\t",
        "annotation_names": None,
        "annotation_barcode_col": "barcodes",
        "annotation_cell_type_col": "celltype",
        "annotation_label": "PBMC cell type annotation",
        "output_dir": str(PROJECT_ROOT / "PBMC" / "vqvae_runs" / "train_val_test" / "cluster_visualization"),
    },

    "Ath": {
        "h5ad_path": str(PROJECT_ROOT / "Ath" / "combined_cellbender_filter_matrix_main_normalized.h5ad"),
        "checkpoint_path": str(PROJECT_ROOT / "Ath" / "vqvae_runs" / "train_val_test" / "best_vqvae_checkpoint.pth"),
        "assignments_csv": str(PROJECT_ROOT / "Ath" / "vqvae_runs" / "train_val_test" / "barcode_cluster_assignments_final_optuna.csv"),
        "annotation_path": str(PROJECT_ROOT / "Ath" / "Shahan_annotation.csv"),
        "annotation_has_header": True,
        "annotation_sep": ",",
        "annotation_names": None,
        "annotation_barcode_col": "cell_barcode_id",
        "annotation_cell_type_col": "celltype.anno",
        "annotation_label": "Shahan cell type annotation",
        "output_dir": str(PROJECT_ROOT / "Ath" / "vqvae_runs" / "train_val_test" / "cluster_visualization"),
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


def check_required_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def load_assignments(assignments_csv):
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


def load_annotation_table():
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
    return str(obs_name).split("-1_")[0]


def get_ath_barcode_core_from_annotation(barcode):
    return str(barcode).rsplit("_", 1)[0]


def get_pbmc_barcode(barcode):
    return str(barcode).strip()


def attach_cell_type_annotations(adata):
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
            print(f"Warning: {missing_sample_map} cells have samples not found in ATH_SAMPLE_MAP.")

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

        annotation_map = annotation_df.set_index(ANNOTATION_BARCODE_COL)[ANNOTATION_CELL_TYPE_COL]

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


def load_best_params():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "hyperparameters" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'hyperparameters'.")

    best_params = checkpoint["hyperparameters"]
    print("Loaded trained parameters:")
    print(best_params)
    return best_params


def build_model(input_dim, best_params):
    return VQVAE(
        input_dim=input_dim,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=best_params["latent_dim"],
        num_embeddings=best_params["num_embeddings"],
        commitment_cost=best_params["commitment_cost"],
    )


def load_checkpoint(model):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def get_dense_matrix(adata):
    X = adata.X

    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X, dtype=np.float32)


def extract_encoder_latent_vectors(model, adata):
    print("Extracting VQ-VAE encoder latent vectors...")

    X = get_dense_matrix(adata)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    latent_batches = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)

            z_e = model.encoder(batch_x)

            if isinstance(z_e, tuple) or isinstance(z_e, list):
                z_e = z_e[0]

            if z_e.dim() > 2:
                z_e = z_e.view(z_e.size(0), -1)

            latent_batches.append(z_e.cpu().numpy())

    latent_vectors = np.vstack(latent_batches)

    print(f"Latent vector shape: {latent_vectors.shape}")

    return latent_vectors


def compute_expression_umap(adata):
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")

    if not os.path.exists(ASSIGNMENTS_CSV):
        raise FileNotFoundError(f"Could not find assignments CSV: {ASSIGNMENTS_CSV}")

    run_tag = "final_optuna"

    print(f"Using assignments file: {ASSIGNMENTS_CSV}")
    print(f"Run tag: {run_tag}")

    print(f"Loading normalized {DATASET_NAME} AnnData...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"AnnData shape: {adata.shape}")

    assignments_df = load_assignments(ASSIGNMENTS_CSV)
    adata = attach_cluster_assignments(adata, assignments_df)

    adata, cell_type_col = attach_cell_type_annotations(adata)

    expression_adata = adata.copy()
    expression_adata = compute_expression_umap(expression_adata)

    expression_vq_output = os.path.join(
        OUTPUT_DIR,
        f"expression_umap_by_vq_cluster_{run_tag}.png",
    )

    save_umap(
        expression_adata,
        color_col=CLUSTER_COL,
        title=f"{DATASET_NAME} expression-space UMAP colored by VQ-VAE cluster",
        output_path=expression_vq_output,
    )

    best_params = load_best_params()

    input_dim = adata.n_vars

    print("Building VQ-VAE model...")
    model = build_model(input_dim=input_dim, best_params=best_params)

    print("Loading trained checkpoint...")
    model = load_checkpoint(model)

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
        title=f"{DATASET_NAME} VQ-VAE latent-space UMAP colored by VQ cluster",
        output_path=latent_vq_output,
    )

    if cell_type_col is not None:
        latent_celltype_output = os.path.join(
            OUTPUT_DIR,
            f"latent_umap_by_annotation_{run_tag}.png",
        )

        known_latent_adata = latent_adata[
            latent_adata.obs[cell_type_col] != "Unknown"
        ].copy()

        save_umap(
            known_latent_adata,
            color_col=cell_type_col,
            title=f"VQ-VAE latent-space UMAP colored by {ANNOTATION_LABEL}",
            output_path=latent_celltype_output,
        )

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