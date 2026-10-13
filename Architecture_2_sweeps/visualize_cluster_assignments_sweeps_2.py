
"""
visualize_cluster_assignments_sweeps_2.py

Create latent-space UMAPs for every completed Architecture_2 sweep run.

This stage is optional and expensive for 25 models. The runner supports
--skip-visualization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from Architecture_2 import VQVAE
from load_data_split_2 import get_count_matrix
from hyperparameter_sweep_config_2 import (
    ANNOTATION_CONFIG,
    ATH_SAMPLE_MAP,
    DATASET_NAME,
    H5AD_PATH,
    INFERENCE_BATCH_SIZE,
    N_NEIGHBORS,
    UMAP_RANDOM_STATE,
    build_experiments,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dense_batch(matrix, start, end):
    batch = matrix[start:end]
    if hasattr(batch, "toarray"):
        batch = batch.toarray()
    return np.asarray(batch, dtype=np.float32)


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


def attach_annotations(obs):
    config = ANNOTATION_CONFIG
    annotations = pd.read_csv(
        config["path"],
        sep=config["sep"],
        low_memory=False,
    )
    result = obs.copy()
    if DATASET_NAME == "PBMC":
        annotations["merge_key"] = annotations[config["barcode_col"]].astype(str).str.strip()
        mapping = annotations.set_index("merge_key")[config["label_col"]]
        keys = result.index.astype(str).str.strip()
    else:
        if "sample" not in result or "orig.ident" not in annotations:
            raise KeyError("Ath annotation matching requires sample and orig.ident.")
        barcode_core = annotations[config["barcode_col"]].astype(str).str.rsplit("_", n=1).str[0]
        annotations["merge_key"] = barcode_core + "__" + annotations["orig.ident"].astype(str)
        mapping = annotations.set_index("merge_key")[config["label_col"]]
        obs_core = result.index.astype(str).str.split("-1_").str[0]
        sample_keys = result["sample"].astype(str).map(ATH_SAMPLE_MAP)
        keys = obs_core + "__" + sample_keys.astype(str)
    result["cell_type_annotation"] = keys.map(mapping).fillna("Unknown")
    result["cell_type_annotation"] = result["cell_type_annotation"].astype(str).str.strip().replace(
        {"": "Unknown", "nan": "Unknown", "None": "Unknown"}
    ).astype("category")
    return result


def save_umap(adata, color, title, output_path):
    figure = sc.pl.umap(
        adata,
        color=color,
        title=title,
        show=False,
        return_fig=True,
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    adata = sc.read_h5ad(H5AD_PATH)
    matrix = get_count_matrix(adata)
    obs = attach_annotations(adata.obs)

    for experiment in build_experiments():
        run_dir = Path(experiment["run_dir"])
        checkpoint_path = run_dir / "best_vqvae_checkpoint.pth"
        assignments_path = run_dir / "barcode_cluster_assignments.csv"

        if not checkpoint_path.exists() or not assignments_path.exists():
            print(f"Skipping incomplete run: {experiment['run_name']}")
            continue

        print(f"Visualizing: {experiment['run_name']}")
        model = load_model(checkpoint_path)
        if matrix.shape[1] != model.encoder.net[0].in_features:
            raise ValueError(
                f"Gene count mismatch for {experiment['run_name']}: "
                f"matrix contains {matrix.shape[1]} genes but the checkpoint expects "
                f"{model.encoder.net[0].in_features}."
            )
        latent_batches = []

        with torch.no_grad():
            for start in range(0, adata.n_obs, INFERENCE_BATCH_SIZE):
                end = min(start + INFERENCE_BATCH_SIZE, adata.n_obs)
                x_batch = torch.tensor(
                    dense_batch(matrix, start, end),
                    dtype=torch.float32,
                    device=DEVICE,
                )
                latent_batches.append(model.encoder(x_batch).cpu().numpy())

        latent = np.vstack(latent_batches)
        assignments = pd.read_csv(assignments_path)
        cluster_map = assignments.set_index("barcode")["cluster_index"]

        latent_obs = obs.copy()
        assigned = latent_obs.index.astype(str).map(cluster_map)
        if assigned.isna().any():
            raise ValueError(
                f"{assigned.isna().sum()} cells have no cluster assignment for "
                f"{experiment['run_name']}."
            )
        latent_obs["cluster_index"] = assigned.astype(str).astype("category")

        latent_adata = sc.AnnData(X=latent, obs=latent_obs)
        sc.pp.neighbors(
            latent_adata,
            n_neighbors=N_NEIGHBORS,
            use_rep="X",
            random_state=UMAP_RANDOM_STATE,
        )
        sc.tl.umap(latent_adata, random_state=UMAP_RANDOM_STATE)

        output_dir = run_dir / "cluster_visualization"
        output_dir.mkdir(parents=True, exist_ok=True)

        save_umap(
            latent_adata,
            "cluster_index",
            (
                f"Architecture_2 latent UMAP: K={experiment['num_embeddings']}, "
                f"batch={experiment['batch_size']}"
            ),
            output_dir / "latent_umap_by_vq_cluster.png",
        )

        known = latent_adata[
            latent_adata.obs["cell_type_annotation"] != "Unknown"
        ].copy()

        if known.n_obs > 0:
            save_umap(
                known,
                "cell_type_annotation",
                f"{DATASET_NAME} latent UMAP by annotation",
                output_dir / "latent_umap_by_annotation.png",
            )


if __name__ == "__main__":
    main()
