"""
load_data_split_4.py

Generic AnnData loader for the orthogonal-codebook VQ-VAE.

THIS LOADER IS FOR MSE TRAINING.
Use normalized/log1p expression h5ad files, not raw-count h5ad files intended
for Negative Binomial loss.

Works for both:
- PBMC
- Arabidopsis / Ath

Path handling:
The h5ad paths are set explicitly below:
    PBMC -> PBMC/PBMC_68k_filtered_normalized.h5ad
    Ath  -> Ath/combined_cellbender_filter_matrix_main_normalized.h5ad

The script resolves those relative paths from PROJECT_ROOT. If PROJECT_ROOT is
wrong, set:

    export VQVAE_PROJECT_ROOT=/lustre/BIF/nobackup/park022/VQVAE
"""

import os
import random
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split


def guess_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent

    paths_to_try = [
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]

    for root_path in paths_to_try:
        root_path = root_path.resolve()
        if (root_path / "PBMC").exists() or (root_path / "Ath").exists():
            return root_path

    return script_dir.resolve()


PROJECT_ROOT = Path(os.environ.get("VQVAE_PROJECT_ROOT", guess_project_root())).resolve()


def project_path(relative_path: str) -> str:
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


DATASET_NAME = "PBMC"  # change to "PBMC" or "Ath"

# Clear single input path per dataset for the MSE orthogonal VQ-VAE.
H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_filtered_normalized.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main_normalized.h5ad"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_dense_matrix(x):
    if isinstance(x, np.ndarray):
        return x
    return x.toarray()


def check_matrix_for_mse(X_np, n_values: int = 10000):
    flat = X_np.ravel()

    if flat.size == 0:
        print("WARNING: Matrix is empty.")
        return

    sample = flat[: min(n_values, flat.size)]
    min_value = float(sample.min())
    max_value = float(sample.max())
    integer_like = bool(np.allclose(sample, np.round(sample), atol=1e-6))

    print("\nMSE input-matrix diagnostic:")
    print(f"Sample min value: {min_value}")
    print(f"Sample max value: {max_value}")
    print(f"Sample values look integer-like: {integer_like}")

    if min_value < 0:
        print("WARNING: Negative values found. The decoder uses ReLU and cannot reconstruct negative values well.")

    if integer_like and max_value > 20:
        print("WARNING: Values look like large raw counts.")
        print("For this MSE orthogonal VQ-VAE, normalized/log1p data is usually preferred.")

    print()


def get_dataloaders(
    h5ad_path: str = H5AD_PATH,
    batch_size: int = 256,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
):
    """
    Load an h5ad matrix and split it into train/validation/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, input_dim, adata
    """
    set_seed(seed)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading AnnData from: {h5ad_path}")

    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"Could not find AnnData file: {h5ad_path}")

    adata = sc.read_h5ad(h5ad_path)
    print(f"AnnData shape: {adata.shape}")

    X_np = to_dense_matrix(adata.X).astype(np.float32)
    check_matrix_for_mse(X_np)

    tensor_data = torch.tensor(X_np, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)

    n_total = len(dataset)
    n_train = int(TRAIN_FRAC * n_total)
    n_val = int(VAL_FRAC * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    input_dim = adata.n_vars

    print(f"Input dimension: {input_dim}")
    print(f"Train cells: {n_train}")
    print(f"Val cells:   {n_val}")
    print(f"Test cells:  {n_test}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader, input_dim, adata


if __name__ == "__main__":
    get_dataloaders(h5ad_path=H5AD_PATH, batch_size=256)
