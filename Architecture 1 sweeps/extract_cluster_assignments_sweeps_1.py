"""
extract_cluster_assignments_sweeps_1.py

Load every completed hyperparameter-sweep checkpoint and save one codebook
assignment per cell for each model run.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from Architecture_1 import VQVAE
from hyperparameter_sweep_config_1 import (
    DATASET_NAME,
    H5AD_PATH,
    ASSIGNMENT_BATCH_SIZE,
    RUN_ROOT,
    build_experiments,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def load_model(checkpoint_path: Path):
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
    return model, checkpoint


def assign_cells(model, expression_matrix):
    all_indices = []
    n_cells = expression_matrix.shape[0]

    with torch.no_grad():
        for start in range(0, n_cells, ASSIGNMENT_BATCH_SIZE):
            end = min(start + ASSIGNMENT_BATCH_SIZE, n_cells)
            batch_np = dense_float32(expression_matrix[start:end])
            x_batch = torch.from_numpy(batch_np).to(DEVICE)
            indices = model.encode(x_batch)
            all_indices.extend(indices.cpu().tolist())

    return all_indices


def main():
    if not Path(H5AD_PATH).exists():
        raise FileNotFoundError(f"Could not find AnnData file: {H5AD_PATH}")

    print(f"Loading {DATASET_NAME} AnnData: {H5AD_PATH}")
    adata = sc.read_h5ad(H5AD_PATH)
    barcodes = adata.obs_names.astype(str).tolist()

    summary_rows = []

    for experiment in build_experiments():
        run_dir = Path(experiment["run_dir"])
        checkpoint_path = run_dir / "best_vqvae_checkpoint.pth"
        output_csv = run_dir / "barcode_cluster_assignments.csv"
        counts_csv = run_dir / "cluster_counts.csv"

        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {checkpoint_path}")
            continue

        print(f"\nExtracting: {experiment['run_name']}")
        model, checkpoint = load_model(checkpoint_path)
        indices = assign_cells(model, adata.X)

        if len(indices) != adata.n_obs:
            raise RuntimeError(
                f"{experiment['run_name']}: predicted {len(indices)} assignments "
                f"for {adata.n_obs} cells."
            )

        assignment_df = pd.DataFrame({
            "barcode": barcodes,
            "cluster_index": indices,
            "run_name": experiment["run_name"],
            "num_embeddings": experiment["num_embeddings"],
            "training_batch_size": experiment["batch_size"],
        })
        assignment_df.to_csv(output_csv, index=False)

        counts = (
            assignment_df["cluster_index"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster_index")
            .reset_index(name="cell_count")
        )
        counts["cell_percent"] = 100.0 * counts["cell_count"] / adata.n_obs
        counts.to_csv(counts_csv, index=False)

        summary_rows.append({
            "run_name": experiment["run_name"],
            "num_embeddings": experiment["num_embeddings"],
            "batch_size": experiment["batch_size"],
            "sweep_membership": ",".join(experiment["sweep_membership"]),
            "n_cells": adata.n_obs,
            "unique_codes_all_cells": int(counts.shape[0]),
            "assignment_csv": str(output_csv),
        })

        print(f"Saved: {output_csv}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not summary_rows:
        raise RuntimeError("No assignments were extracted. Train the models first.")

    with (RUN_ROOT / "assignment_extraction_summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=4)

    pd.DataFrame(summary_rows).to_csv(
        RUN_ROOT / "assignment_extraction_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
