"""
extract_cluster_assignments.py

Load a trained VQ-VAE checkpoint, run all PBMC cells through the model,
and save one row per cell with:
    - barcode
    - cluster_index (discrete codebook assignment)

Output:
    PBMC/barcode_cluster_assignments.csv
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from Architecture_2 import VQVAE


# -----------------------------
# CONFIG
# -----------------------------
H5AD_PATH = "PBMC/PBMC_68k_filtered_normalized.h5ad"
CHECKPOINT_PATH = "PBMC/vqvae_checkpoint.pth"
OUTPUT_CSV = "PBMC/barcode_cluster_assignments.csv"
BATCH_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# HELPERS
# -----------------------------
def to_dense_matrix(x):
    """Convert AnnData matrix to dense numpy array if needed."""
    if isinstance(x, np.ndarray):
        return x
    return x.toarray()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> VQVAE:
    """Rebuild and load the VQ-VAE from a saved checkpoint."""
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
    ).to(device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# -----------------------------
# MAIN
# -----------------------------
def main():
    print(f"Using device: {DEVICE}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    print("Loading preprocessed AnnData...")
    adata = sc.read_h5ad(H5AD_PATH)

    print(f"AnnData shape: {adata.shape}")
    X = to_dense_matrix(adata.X)
    barcodes = list(adata.obs_names)

    if len(barcodes) != X.shape[0]:
        raise ValueError(
            f"Number of barcodes ({len(barcodes)}) does not match number of cells ({X.shape[0]})."
        )

    print("Loading trained VQ-VAE checkpoint...")
    model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    print("Extracting cluster assignments...")
    all_indices = []

    with torch.no_grad():
        for start in range(0, X.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, X.shape[0])
            x_batch = torch.tensor(X[start:end], dtype=torch.float32, device=DEVICE)

            indices = model.encode(x_batch)  # shape: (batch,)
            all_indices.extend(indices.cpu().numpy().tolist())

    if len(all_indices) != len(barcodes):
        raise ValueError(
            f"Number of predicted indices ({len(all_indices)}) does not match number of barcodes ({len(barcodes)})."
        )

    results_df = pd.DataFrame({
        "barcode": barcodes,
        "cluster_index": all_indices,
    })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved cluster assignments to:")
    print(OUTPUT_CSV)
    print("\nPreview:")
    print(results_df.head())

    print("\nCluster counts:")
    print(results_df["cluster_index"].value_counts().sort_index())


if __name__ == "__main__":
    main()