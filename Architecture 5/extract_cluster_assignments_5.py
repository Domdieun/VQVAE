"""
extract_cluster_assignments_5.py

Load a trained Architecture_5 checkpoint, assign every cell to a discrete
VQ codebook entry, and save the assignments.

This script uses the same raw-count h5ad file used for training.
It is sparse-friendly and does not densify the full matrix.
"""

import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from Architecture_5 import VQVAE
from load_data_split_5 import (
    project_path,
    PROJECT_ROOT,
    make_full_dataloader_from_adata,
    compute_library_sizes,
    ensure_csr_or_ndarray,
    get_count_matrix,
    read_count_adata,
)


# CONFIG

DATASET_NAME = "PBMC"  # change to "PBMC" or "Ath"

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
}

CHECKPOINT_PATHS = {
    "PBMC": project_path("PBMC/vqvae_runs/train_val_test_5/best_vqvae_5_checkpoint.pth"),
    "Ath": project_path("Ath/vqvae_runs/train_val_test_5/best_vqvae_5_checkpoint.pth"),
}

OUTPUT_CSVS = {
    "PBMC": project_path("PBMC/vqvae_runs/train_val_test_5/barcode_cluster_assignments_final_optuna_architecture_5.csv"),
    "Ath": project_path("Ath/vqvae_runs/train_val_test_5/barcode_cluster_assignments_final_optuna_architecture_5.csv"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
CHECKPOINT_PATH = CHECKPOINT_PATHS[DATASET_NAME]
OUTPUT_CSV = OUTPUT_CSVS[DATASET_NAME]

BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> VQVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "hyperparameters" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'hyperparameters'.")

    hparams = checkpoint["hyperparameters"]

    required_keys = [
        "input_dim",
        "hidden_dim",
        "latent_dim",
        "num_embeddings",
        "commitment_cost",
    ]

    missing = [k for k in required_keys if k not in hparams]
    if missing:
        raise KeyError(f"Missing hyperparameters in checkpoint: {missing}")

    model = VQVAE(
        input_dim=hparams["input_dim"],
        hidden_dim=hparams["hidden_dim"],
        latent_dim=hparams["latent_dim"],
        num_embeddings=hparams["num_embeddings"],
        commitment_cost=hparams["commitment_cost"],
        dropout_rate=hparams.get("dropout_rate", 0.1),
        gene_likelihood=hparams.get("gene_likelihood", "nb"),
    ).to(device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    print(f"Loading AnnData: {H5AD_PATH}")
    adata = read_count_adata(H5AD_PATH, dataset_name=DATASET_NAME)
    print(f"AnnData shape: {adata.shape}")

    barcodes = list(adata.obs_names)

    X_counts = ensure_csr_or_ndarray(get_count_matrix(adata))
    library_sizes = compute_library_sizes(X_counts)

    print(f"Library size min/median/max: {library_sizes.min():.2f} / {np.median(library_sizes):.2f} / {library_sizes.max():.2f}")

    loader, input_dim = make_full_dataloader_from_adata(
        adata=adata,
        batch_size=BATCH_SIZE,
    )

    print(f"Input dim from data: {input_dim}")

    print(f"\nLoading trained Architecture_5 checkpoint: {CHECKPOINT_PATH}")
    model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    print("Extracting codebook assignments...")
    all_indices = []

    with torch.no_grad():
        processed = 0

        for x_batch, _library_batch in loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            indices = model.encode(x_batch)

            all_indices.extend(indices.cpu().numpy().tolist())

            processed += x_batch.shape[0]
            if processed % (BATCH_SIZE * 100) == 0 or processed == adata.n_obs:
                print(f"Processed {processed}/{adata.n_obs} cells")

    if len(all_indices) != len(barcodes):
        raise ValueError(
            f"Number of predicted indices ({len(all_indices)}) does not match "
            f"number of barcodes ({len(barcodes)})."
        )

    results_df = pd.DataFrame({
        "barcode": barcodes,
        "cluster_index": all_indices,
        "library_size": library_sizes,
    })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    counts = results_df["cluster_index"].value_counts().sort_index()
    code_count = model.vq.num_embeddings
    summary = {
        "total_codes": int(code_count),
        "active_codes": int(len(counts)),
        "usage_percent": float(100 * len(counts) / code_count),
        "cluster_counts": {str(int(k)): int(v) for k, v in counts.items()},
    }
    usage_path = os.path.join(os.path.dirname(OUTPUT_CSV), "codebook_usage_architecture_5.json")
    with open(usage_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved codebook utilization to: {usage_path}")

    print("\nSaved cluster assignments to:")
    print(OUTPUT_CSV)

    print("\nPreview:")
    print(results_df.head())

    print("\nCluster counts:")
    print(results_df["cluster_index"].value_counts().sort_index())


if __name__ == "__main__":
    main()
