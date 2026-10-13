"""
visualize_cluster_assignments_sweeps_1.py

Create comparable encoder-latent (z_e) UMAPs for every completed
Architecture_1 sweep run. Each model is reconstructed directly from its own
checkpoint, preventing hyperparameter/checkpoint mismatches. z_q is not sent
through UMAP because it contains only K unique codebook vectors repeated across
cells, which can create visually misleading clouds of duplicate points.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from Architecture_1 import VQVAE
from hyperparameter_sweep_config_1 import (
    ANNOTATION_CONFIG,
    ATH_SAMPLE_MAP,
    DATASET_NAME,
    H5AD_PATH,
    INFERENCE_BATCH_SIZE,
    N_NEIGHBORS,
    RUN_ROOT,
    UMAP_RANDOM_STATE,
    build_experiments,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BARCODE_COL = "barcode"
CLUSTER_COL = "cluster_index"


def dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    hparams = checkpoint["hyperparameters"]
    model = VQVAE(
        input_dim=hparams["input_dim"],
        hidden_dim=hparams["hidden_dim"],
        latent_dim=hparams["latent_dim"],
        num_embeddings=hparams["num_embeddings"],
        commitment_cost=hparams["commitment_cost"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def attach_assignments(adata, assignments_path):
    assignments_df = pd.read_csv(assignments_path)
    cluster_map = assignments_df.set_index(BARCODE_COL)[CLUSTER_COL]

    output = adata.copy()
    output.obs[BARCODE_COL] = output.obs_names.astype(str)
    output.obs[CLUSTER_COL] = output.obs[BARCODE_COL].map(cluster_map)

    missing = output.obs[CLUSTER_COL].isna().sum()
    if missing:
        raise ValueError(f"{missing} cells are missing cluster assignments.")

    output.obs[CLUSTER_COL] = output.obs[CLUSTER_COL].astype(str).astype("category")
    return output


def load_annotation_table():
    path = Path(ANNOTATION_CONFIG["path"])
    if not path.exists():
        print(f"Annotation file not found: {path}")
        return None

    if ANNOTATION_CONFIG["has_header"]:
        return pd.read_csv(path, sep=ANNOTATION_CONFIG["sep"], low_memory=False)

    return pd.read_csv(
        path,
        sep=ANNOTATION_CONFIG["sep"],
        header=None,
        names=ANNOTATION_CONFIG["names"],
        low_memory=False,
    )


def attach_annotations(adata, annotation_df):
    if annotation_df is None:
        return adata, None

    output = adata.copy()
    barcode_col = ANNOTATION_CONFIG["barcode_col"]
    label_col = ANNOTATION_CONFIG["label_col"]

    if DATASET_NAME == "PBMC":
        annotation_df = annotation_df.copy()
        annotation_df["merge_key"] = annotation_df[barcode_col].astype(str).str.strip()
        annotation_map = annotation_df.set_index("merge_key")[label_col]
        output.obs["merge_key"] = output.obs_names.astype(str).str.strip()
        output.obs["cell_type_annotation"] = output.obs["merge_key"].map(annotation_map)
    else:
        annotation_df = annotation_df.copy()
        annotation_df["barcode_core"] = (
            annotation_df[barcode_col].astype(str).map(lambda x: x.rsplit("_", 1)[0])
        )
        annotation_df["sample_key"] = annotation_df["orig.ident"].astype(str)
        annotation_df["merge_key"] = (
            annotation_df["barcode_core"] + "__" + annotation_df["sample_key"]
        )
        annotation_map = annotation_df.set_index("merge_key")[label_col]

        output.obs["barcode_core"] = (
            output.obs_names.astype(str).map(lambda x: x.split("-1_")[0])
        )
        output.obs["sample_key"] = output.obs["sample"].astype(str).map(ATH_SAMPLE_MAP)
        output.obs["merge_key"] = (
            output.obs["barcode_core"] + "__" + output.obs["sample_key"].astype(str)
        )
        output.obs["cell_type_annotation"] = output.obs["merge_key"].map(annotation_map)

    matched = output.obs["cell_type_annotation"].notna().sum()
    if matched == 0:
        print("No annotation barcodes matched; annotation UMAP will be skipped.")
        return output, None

    output.obs["cell_type_annotation"] = (
        output.obs["cell_type_annotation"].fillna("Unknown").astype("category")
    )
    return output, "cell_type_annotation"


def extract_encoder_latents(model, expression_matrix):
    """Extract z_e; this preserves cell-level variation for meaningful UMAPs."""
    z_e_batches = []
    n_cells = expression_matrix.shape[0]

    with torch.no_grad():
        for start in range(0, n_cells, INFERENCE_BATCH_SIZE):
            end = min(start + INFERENCE_BATCH_SIZE, n_cells)
            x_batch = torch.from_numpy(dense_float32(expression_matrix[start:end])).to(DEVICE)
            z_e = model.encoder(x_batch)
            z_e_batches.append(z_e.cpu().numpy())

    return np.vstack(z_e_batches)


def compute_umap(vectors, obs):
    latent_adata = sc.AnnData(X=vectors, obs=obs.copy())
    sc.pp.neighbors(
        latent_adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X",
        random_state=UMAP_RANDOM_STATE,
    )
    sc.tl.umap(latent_adata, random_state=UMAP_RANDOM_STATE)
    return latent_adata


def save_umap(adata, color, title, output_path):
    fig = sc.pl.umap(
        adata,
        color=color,
        title=title,
        show=False,
        return_fig=True,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if not Path(H5AD_PATH).exists():
        raise FileNotFoundError(f"Could not find AnnData file: {H5AD_PATH}")

    print(f"Loading AnnData once: {H5AD_PATH}")
    base_adata = sc.read_h5ad(H5AD_PATH)
    annotation_df = load_annotation_table()

    completed = 0

    for experiment in build_experiments():
        run_dir = Path(experiment["run_dir"])
        checkpoint_path = run_dir / "best_vqvae_checkpoint.pth"
        assignments_path = run_dir / "barcode_cluster_assignments.csv"
        output_dir = run_dir / "visualizations"

        if not checkpoint_path.exists() or not assignments_path.exists():
            print(f"Skipping incomplete run: {experiment['run_name']}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nVisualizing {experiment['run_name']}")

        adata = attach_assignments(base_adata, assignments_path)
        adata, annotation_col = attach_annotations(adata, annotation_df)

        model = load_model(checkpoint_path)
        z_e = extract_encoder_latents(model, adata.X)
        ze_adata = compute_umap(z_e, adata.obs)

        title_suffix = (
            f"K={experiment['num_embeddings']}, batch={experiment['batch_size']}"
        )

        save_umap(
            ze_adata,
            CLUSTER_COL,
            f"z_e UMAP by VQ cluster ({title_suffix})",
            output_dir / "ze_umap_by_vq_cluster.png",
        )
        if annotation_col is not None:
            known_ze = ze_adata[ze_adata.obs[annotation_col] != "Unknown"].copy()
            save_umap(
                known_ze,
                annotation_col,
                f"z_e UMAP by annotation ({title_suffix})",
                output_dir / "ze_umap_by_annotation.png",
            )
        np.save(output_dir / "z_e.npy", z_e)
        completed += 1

        del model, z_e, ze_adata, adata
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if completed == 0:
        raise RuntimeError("No runs were visualized. Train and extract assignments first.")

    print(f"\nVisualized {completed} unique runs under: {RUN_ROOT}")


if __name__ == "__main__":
    main()
