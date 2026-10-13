"""
extract_cluster_assignments_4.py

Load a trained orthogonal-codebook VQ-VAE checkpoint, assign every cell to a
codebook entry, and save barcode-to-cluster assignments.

THIS VERSION USES THE MSE ORTHOGONAL MODEL.
"""

import json
import os

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

CHECKPOINT_PATHS = {
    "PBMC": project_path("PBMC/vqvae_runs/train_val_test_orthogonal_mse/best_vqvae_orthogonal_mse_checkpoint.pth"),
    "Ath": project_path("Ath/vqvae_runs/train_val_test_orthogonal_mse/best_vqvae_orthogonal_mse_checkpoint.pth"),
}

OUTPUT_CSVS = {
    "PBMC": project_path(
        "PBMC/vqvae_runs/train_val_test_orthogonal_mse/barcode_cluster_assignments_final_optuna_orthogonal_mse.csv"),
    "Ath": project_path(
        "Ath/vqvae_runs/train_val_test_orthogonal_mse/barcode_cluster_assignments_final_optuna_orthogonal_mse.csv"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
CHECKPOINT_PATH = CHECKPOINT_PATHS[DATASET_NAME]
OUTPUT_CSV = OUTPUT_CSVS[DATASET_NAME]

BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_dense_matrix(x):
    if isinstance(x, np.ndarray):
        return x
    return x.toarray()


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
        orthogonal_reg_weight=hparams.get("orthogonal_reg_weight", 0.0),
        orthogonal_reg_max_codes=hparams.get("orthogonal_reg_max_codes", None),
        orthogonal_reg_active_codes_only=hparams.get("orthogonal_reg_active_codes_only", False),
    ).to(device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    print(f"Dataset: {DATASET_NAME}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"H5AD path: {H5AD_PATH}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("Model/loss type: Orthogonal VQ-VAE with MSE reconstruction")
    print(f"Device: {DEVICE}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    adata = sc.read_h5ad(H5AD_PATH)
    print(f"AnnData shape: {adata.shape}")

    model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    X_np = to_dense_matrix(adata.X).astype(np.float32)
    tensor_data = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    all_indices = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE)
            indices = model.encode(x_batch)
            all_indices.append(indices.cpu().numpy())

    cluster_indices = np.concatenate(all_indices)

    assignments_df = pd.DataFrame(
        {
            "barcode": adata.obs_names.astype(str),
            "cluster_index": cluster_indices.astype(int),
        }
    )

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    assignments_df.to_csv(OUTPUT_CSV, index=False)

    cluster_counts = assignments_df["cluster_index"].value_counts().sort_index()
    total_codes = model.vq.num_embeddings
    active_codes = int(cluster_counts.size)
    usage = {
        "total_codes": total_codes,
        "active_codes": active_codes,
        "usage_percent": 100.0 * active_codes / total_codes,
        "cluster_counts": {str(int(k)): int(v) for k, v in cluster_counts.items()},
    }
    usage_path = os.path.join(os.path.dirname(OUTPUT_CSV), "codebook_usage_orthogonal_mse.json")
    with open(usage_path, "w") as handle:
        json.dump(usage, handle, indent=2)
    print(f"Saved codebook utilization to: {usage_path}")

    print(f"Saved assignments to: {OUTPUT_CSV}")
    print("Cluster counts:")
    print(assignments_df["cluster_index"].value_counts().sort_index())


if __name__ == "__main__":
    main()
