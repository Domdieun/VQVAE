"""
extract_cluster_assignments_2.py

Load a trained Architecture_2 VQ-VAE checkpoint, assign every cell to a
discrete codebook entry, and save those assignments to a CSV.

Purpose:
This script is the bridge between a trained VQ-VAE model and downstream
biological analysis.

It does not retrain the model. Instead, it:
1. loads the saved Architecture_2 model checkpoint
2. rebuilds the VQ-VAE architecture from the stored hyperparameters
3. runs every cell through the encoder + vector quantizer
4. records the discrete codebook index chosen for each cell

What `cluster_index` means here:
- Each cell is passed through the encoder to produce a continuous latent vector.
- The vector quantizer then compares that latent vector to all codebook vectors.
- The nearest codebook vector is selected.
- The integer ID of that selected codebook vector is saved as `cluster_index`.

So `cluster_index` is not a cell type label by itself. It is the model's
discrete latent assignment for that cell. Later, this can be compared to known
annotations to see whether the learned discrete states correspond to biology.

Important:
- This version is for Architecture_2.
- Architecture_2 was trained using Negative Binomial loss.
- Therefore, this script should use the same raw-count h5ad file used during
  training, not the normalized/log1p h5ad file.

Output:
    <DATASET>/vqvae_runs/train_val_test_2/barcode_cluster_assignments_final_optuna_architecture_2.csv

The output CSV contains one row per cell with:
- `barcode`: the cell identifier from the AnnData object
- `cluster_index`: the VQ-VAE codebook entry assigned to that cell
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from Architecture_2 import VQVAE
from load_data_split_2 import project_path, PROJECT_ROOT, get_count_matrix



# CONFIG

# Change this to "PBMC" or "Ath".
DATASET_NAME = "PBMC"

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
}

CHECKPOINT_PATHS = {
    "PBMC": project_path("PBMC/vqvae_runs/train_val_test_2/best_vqvae_2_checkpoint.pth"),
    "Ath": project_path("Ath/vqvae_runs/train_val_test_2/best_vqvae_2_checkpoint.pth"),
}

OUTPUT_CSVS = {
    "PBMC": project_path("PBMC/vqvae_runs/train_val_test_2/barcode_cluster_assignments_final_optuna_architecture_2.csv"),
    "Ath": project_path("Ath/vqvae_runs/train_val_test_2/barcode_cluster_assignments_final_optuna_architecture_2.csv"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
CHECKPOINT_PATH = CHECKPOINT_PATHS[DATASET_NAME]
OUTPUT_CSV = OUTPUT_CSVS[DATASET_NAME]

BATCH_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# HELPERS

def to_dense_matrix(x):
    """
    Convert an AnnData expression matrix to a dense NumPy array if needed.

    The expression matrix may be stored either as:
    - a regular dense NumPy array
    - a sparse matrix to save memory

    PyTorch tensor construction in this script expects a dense array for the
    batched forward pass, so sparse matrices are converted here.

    Args:
        x:
            AnnData `.X` matrix, either dense or sparse.

    Returns:
        Dense NumPy array version of the matrix.
    """
    if isinstance(x, np.ndarray):
        return x

    return x.toarray()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> VQVAE:
    """
    Rebuild the Architecture_2 VQ-VAE and load its learned weights.

    The checkpoint stores both:
    - the model parameters (`model_state_dict`)
    - the hyperparameters needed to reconstruct the architecture

    This function first validates that the required architecture settings are
    present, then rebuilds the model with those settings, loads the trained
    weights, and switches the model to evaluation mode.

    Args:
        checkpoint_path:
            Path to the saved `.pth` checkpoint file.

        device:
            Torch device on which the model should be loaded.

    Returns:
        A VQVAE model ready for inference.

    Raises:
        KeyError:
            If required checkpoint fields are missing.
    """
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


def extract_indices_from_encode_output(encode_output):
    """
    Make cluster extraction robust to different encode(...) return formats.

    Some versions of the model may return only:
        indices

    Other versions may return something like:
        z_e, z_q, indices

    This helper keeps only the discrete codebook indices.
    """
    if isinstance(encode_output, tuple):
        indices = encode_output[-1]
    else:
        indices = encode_output

    return indices


def check_count_matrix(X_sample):
    """
    Print a small diagnostic to confirm that the matrix looks like raw counts.

    Negative Binomial models should be run on raw count-like input, not
    normalized/log1p data.
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
            "\nIf this h5ad contains normalized/log1p values, it should not be used"
            "\nwith the Architecture_2 model trained using Negative Binomial loss."
        )





def main():
    """
    Run the trained Architecture_2 VQ-VAE on all cells and save one discrete
    assignment per cell.

    workflow:
    1. Load the raw-count AnnData object.
    2. Extract the expression matrix and the cell barcodes.
    3. Rebuild the trained Architecture_2 VQ-VAE from the checkpoint.
    4. Process cells in batches through `model.encode(...)`.
    5. Save the resulting barcode-to-cluster mapping as a CSV.
    """
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")

    print(f"Loading AnnData: {H5AD_PATH}")
    adata = sc.read_h5ad(H5AD_PATH)

    print(f"AnnData shape: {adata.shape}")

    X = to_dense_matrix(get_count_matrix(adata)).astype(np.float32)

    # Check only a small sample so this does not slow down the script.
    sample_n = min(1000, X.shape[0])
    check_count_matrix(X[:sample_n])

    barcodes = list(adata.obs_names)

    if len(barcodes) != X.shape[0]:
        raise ValueError(
            f"Number of barcodes ({len(barcodes)}) does not match "
            f"number of cells ({X.shape[0]})."
        )

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    print(f"\nLoading trained Architecture_2 checkpoint: {CHECKPOINT_PATH}")
    model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
    if X.shape[1] != model.input_dim:
        raise ValueError(f"Count matrix has {X.shape[1]} genes but checkpoint expects {model.input_dim}.")

    print("Extracting cluster assignments...")
    all_indices = []

    with torch.no_grad():
        for start in range(0, X.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, X.shape[0])

            x_batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=DEVICE,
            )

            encode_output = model.encode(x_batch)
            indices = extract_indices_from_encode_output(encode_output)

            all_indices.extend(indices.cpu().numpy().tolist())

            if start % (BATCH_SIZE * 100) == 0:
                print(f"Processed {end}/{X.shape[0]} cells")

    if len(all_indices) != len(barcodes):
        raise ValueError(
            f"Number of predicted indices ({len(all_indices)}) does not match "
            f"number of barcodes ({len(barcodes)})."
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