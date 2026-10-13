"""
visualize_cluster_assignments_4.py

Create UMAP visualizations for a trained orthogonal-codebook VQ-VAE model.

THIS VERSION USES THE MSE ORTHOGONAL MODEL.
Use the same normalized/log1p h5ad file used during training and extraction.

Outputs:
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

from load_data_split_4 import project_path, PROJECT_ROOT
from Architecture_4 import VQVAE


# CONFIG

DATASET_NAME = "PBMC"  # change to "PBMC" or "Ath"

# Clear single input path per dataset for this MSE model.
H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_filtered_normalized.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main_normalized.h5ad"),
}

DATASET_CONFIGS = {
    "PBMC": {
        "h5ad_path": H5AD_PATHS["PBMC"],
        "checkpoint_path": project_path(
            "PBMC/vqvae_runs/train_val_test_orthogonal_mse/best_vqvae_orthogonal_mse_checkpoint.pth"),
        "assignments_csv": project_path(
            "PBMC/vqvae_runs/train_val_test_orthogonal_mse/barcode_cluster_assignments_final_optuna_orthogonal_mse.csv"),
        "annotation_path": project_path("PBMC/68k_pbmc_barcodes_annotation.tsv"),
        "annotation_has_header": True,
        "annotation_sep": "\t",
        "annotation_names": None,
        "annotation_barcode_col": "barcodes",
        "annotation_cell_type_col": "celltype",
        "annotation_label": "PBMC cell type annotation",
        "output_dir": project_path("PBMC/vqvae_runs/train_val_test_orthogonal_mse/cluster_visualization"),
    },
    "Ath": {
        "h5ad_path": H5AD_PATHS["Ath"],
        "checkpoint_path": project_path(
            "Ath/vqvae_runs/train_val_test_orthogonal_mse/best_vqvae_orthogonal_mse_checkpoint.pth"),
        "assignments_csv": project_path(
            "Ath/vqvae_runs/train_val_test_orthogonal_mse/barcode_cluster_assignments_final_optuna_orthogonal_mse.csv"),
        "annotation_path": project_path("Ath/Shahan_annotation.csv"),
        "annotation_has_header": True,
        "annotation_sep": ",",
        "annotation_names": None,
        "annotation_barcode_col": "cell_barcode_id",
        "annotation_cell_type_col": "celltype.anno",
        "annotation_label": "Shahan cell type annotation",
        "output_dir": project_path("Ath/vqvae_runs/train_val_test_orthogonal_mse/cluster_visualization"),
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


def check_required_columns(df, columns, df_name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def to_dense_matrix(x):
    if isinstance(x, np.ndarray):
        return x
    return x.toarray()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> VQVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint["hyperparameters"]

    model = VQVAE(
        input_dim=hparams["input_dim"],
        hidden_dim=hparams["hidden_dim"],
        latent_dim=hparams["latent_dim"],
        num_embeddings=hparams["num_embeddings"],
        commitment_cost=hparams["commitment_cost"],
        orthogonal_reg_weight=hparams.get("orthogonal_reg_weight", 0.0),
        orthogonal_reg_max_codes=hparams.get("orthogonal_reg_max_codes", None),
        orthogonal_reg_active_codes_only=hparams.get("orthogonal_reg_active_codes_only", False),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_assignments(assignments_csv):
    assignments_df = pd.read_csv(assignments_csv)
    check_required_columns(assignments_df, [BARCODE_COL, CLUSTER_COL], "Assignments CSV")
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
            "Check whether barcodes match between the h5ad file and assignments CSV."
        )

    adata.obs[CLUSTER_COL] = adata.obs[CLUSTER_COL].astype("category")
    return adata


def load_annotation_table():
    if ANNOTATION_PATH is None or not os.path.exists(ANNOTATION_PATH):
        return None

    if ANNOTATION_HAS_HEADER:
        annotation_df = pd.read_csv(ANNOTATION_PATH, sep=ANNOTATION_SEP, low_memory=False)
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
        check_required_columns(adata.obs, ["sample"], "AnnData obs")
        check_required_columns(annotation_df, ["orig.ident"], "Ath annotation file")

        annotation_df["barcode_core"] = annotation_df[ANNOTATION_BARCODE_COL].map(get_ath_barcode_core_from_annotation)
        annotation_df["sample_key"] = annotation_df["orig.ident"].astype(str)
        annotation_df["merge_key"] = annotation_df["barcode_core"] + "__" + annotation_df["sample_key"]

        adata.obs["barcode_core"] = adata.obs_names.astype(str).map(get_ath_barcode_core_from_adata)
        adata.obs["sample_key"] = adata.obs["sample"].astype(str).map(ATH_SAMPLE_MAP)

        missing_sample_map = adata.obs["sample_key"].isna().sum()
        if missing_sample_map > 0:
            print(f"Warning: {missing_sample_map} cells have samples not found in ATH_SAMPLE_MAP.")

        adata.obs["merge_key"] = adata.obs["barcode_core"] + "__" + adata.obs["sample_key"].astype(str)
        annotation_map = annotation_df.set_index("merge_key")[ANNOTATION_CELL_TYPE_COL]
        adata.obs["cell_type_annotation"] = adata.obs["merge_key"].map(annotation_map)

    else:
        annotation_df[ANNOTATION_BARCODE_COL] = annotation_df[ANNOTATION_BARCODE_COL].map(get_pbmc_barcode)
        annotation_df[ANNOTATION_CELL_TYPE_COL] = annotation_df[ANNOTATION_CELL_TYPE_COL].astype(str)

        annotation_map = annotation_df.set_index(ANNOTATION_BARCODE_COL)[ANNOTATION_CELL_TYPE_COL]
        adata.obs["barcode_for_annotation"] = adata.obs_names.astype(str).map(get_pbmc_barcode)
        adata.obs["cell_type_annotation"] = adata.obs["barcode_for_annotation"].map(annotation_map)

    missing_annotations = adata.obs["cell_type_annotation"].isna().sum()
    matched_annotations = adata.n_obs - missing_annotations
    print(f"Matched annotations: {matched_annotations}/{adata.n_obs}")
    print(f"Missing annotations: {missing_annotations}")

    if matched_annotations == 0:
        print("No annotation barcodes matched. Skipping annotation-colored UMAP.")
        return adata, None

    adata.obs["cell_type_annotation"] = adata.obs["cell_type_annotation"].fillna("Unknown").astype("category")
    return adata, "cell_type_annotation"


def compute_expression_umap(adata):
    print("Computing expression-space PCA...")
    sc.tl.pca(adata, svd_solver="arpack", random_state=RANDOM_STATE)

    print("Computing expression-space neighbor graph...")
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        random_state=RANDOM_STATE,
    )

    print("Computing expression-space UMAP...")
    sc.tl.umap(adata, random_state=RANDOM_STATE)
    return adata


def compute_latent_vectors(model, adata):
    print("Computing continuous encoder latent vectors...")

    X_np = to_dense_matrix(adata.X).astype(np.float32)
    tensor_data = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    latent_batches = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE)
            z_e = model.encode_latent(x_batch)
            latent_batches.append(z_e.cpu().numpy())

    return np.concatenate(latent_batches, axis=0)


def compute_latent_umap(latent_vectors, obs):
    latent_adata = sc.AnnData(X=latent_vectors, obs=obs.copy())

    print("Computing latent-space neighbor graph...")
    sc.pp.neighbors(
        latent_adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X",
        random_state=RANDOM_STATE,
    )

    print("Computing latent-space UMAP...")
    sc.tl.umap(latent_adata, random_state=RANDOM_STATE)
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"H5AD path: {H5AD_PATH}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Assignments CSV: {ASSIGNMENTS_CSV}")
    print(f"Annotation path: {ANNOTATION_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Model/loss type: Orthogonal VQ-VAE with MSE reconstruction")
    print(f"Device: {DEVICE}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(ASSIGNMENTS_CSV):
        raise FileNotFoundError(f"Could not find assignments CSV: {ASSIGNMENTS_CSV}")

    run_tag = "orthogonal_mse"

    print(f"Loading normalized/log1p {DATASET_NAME} AnnData...")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"AnnData shape: {adata.shape}")

    assignments_df = load_assignments(ASSIGNMENTS_CSV)
    adata = attach_cluster_assignments(adata, assignments_df)
    adata, cell_type_col = attach_cell_type_annotations(adata)

    expression_adata = compute_expression_umap(adata.copy())
    save_umap(
        expression_adata,
        CLUSTER_COL,
        f"{DATASET_NAME} expression UMAP by orthogonal VQ-VAE cluster",
        os.path.join(OUTPUT_DIR, f"expression_umap_by_vq_cluster_{run_tag}.png"),
    )

    model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
    latent_vectors = compute_latent_vectors(model, adata)
    latent_adata = compute_latent_umap(latent_vectors, adata.obs)

    save_umap(
        latent_adata,
        CLUSTER_COL,
        f"{DATASET_NAME} latent UMAP by orthogonal VQ-VAE cluster",
        os.path.join(OUTPUT_DIR, f"latent_umap_by_vq_cluster_{run_tag}.png"),
    )

    if cell_type_col is not None:
        save_umap(
            latent_adata,
            cell_type_col,
            f"{DATASET_NAME} latent UMAP by {ANNOTATION_LABEL}",
            os.path.join(OUTPUT_DIR, f"latent_umap_by_cell_type_annotation_{run_tag}.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
