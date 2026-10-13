
"""
extract_cluster_assignments_sweeps_2.py

Extract one codebook assignment per PBMC cell for all 25 checkpoints.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from Architecture_2 import VQVAE
from load_data_split_2 import get_count_matrix
from hyperparameter_sweep_config_2 import (
    H5AD_PATH,
    ASSIGNMENT_BATCH_SIZE,
    RUN_ROOT,
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


def main():
    print(f"Loading AnnData: {H5AD_PATH}")
    adata = sc.read_h5ad(H5AD_PATH)
    matrix = get_count_matrix(adata)
    barcodes = adata.obs_names.astype(str).tolist()

    for experiment in build_experiments():
        run_dir = Path(experiment["run_dir"])
        checkpoint_path = run_dir / "best_vqvae_checkpoint.pth"
        output_path = run_dir / "barcode_cluster_assignments.csv"

        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {checkpoint_path}")
            continue

        print(f"\nExtracting: {experiment['run_name']}")
        model = load_model(checkpoint_path)
        if matrix.shape[1] != model.encoder.net[0].in_features:
            raise ValueError(
                f"Gene count mismatch for {experiment['run_name']}: "
                f"matrix contains {matrix.shape[1]} genes but the checkpoint expects "
                f"{model.encoder.net[0].in_features}."
            )
        all_indices = []

        with torch.no_grad():
            for start in range(0, adata.n_obs, ASSIGNMENT_BATCH_SIZE):
                end = min(start + ASSIGNMENT_BATCH_SIZE, adata.n_obs)
                x_batch = torch.tensor(
                    dense_batch(matrix, start, end),
                    dtype=torch.float32,
                    device=DEVICE,
                )
                indices = model.encode(x_batch)
                if isinstance(indices, tuple):
                    indices = indices[-1]
                all_indices.extend(indices.cpu().numpy().tolist())

        if len(all_indices) != len(barcodes):
            raise ValueError(
                f"Expected {len(barcodes)} cluster assignments, got {len(all_indices)}."
            )
        assignments = pd.DataFrame({
            "barcode": barcodes,
            "cluster_index": all_indices,
        })
        assignments.to_csv(output_path, index=False)

        counts = (
            assignments["cluster_index"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster_index")
            .reset_index(name="cell_count")
        )
        counts.to_csv(run_dir / "cluster_counts.csv", index=False)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
