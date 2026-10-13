"""
load_data_split_2_1.py

Generic AnnData loader for Architecture_2 VQ-VAE training with Negative
Binomial reconstruction loss.

Works for both:
- PBMC raw-count h5ad
- Arabidopsis raw-count h5ad

Important:
Negative Binomial loss should be trained on RAW COUNTS, not log-normalized data.

This loader tries to load raw counts in this order:
1. adata.layers["counts"]
2. adata.raw.X
3. adata.X as fallback, with a warning
"""

import os
import random
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split


def guess_project_root() -> Path:
    """
    Guess the project root so scripts still work when they are run from a
    subfolder such as a script subfolder/.

    Override manually with:
        export VQVAE_PROJECT_ROOT=/lustre/BIF/nobackup/park022/VQVAE
    or another folder that contains PBMC/ and Ath/.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]

    for candidate in candidates:
        if (candidate / "PBMC").exists() or (candidate / "Ath").exists():
            return candidate.resolve()

    return script_dir.resolve()


PROJECT_ROOT = Path(os.environ.get("VQVAE_PROJECT_ROOT", guess_project_root())).resolve()


def project_path(relative_path: str) -> str:
    return str(PROJECT_ROOT / relative_path)


DATASET_NAME = "PBMC"  # change to "PBMC" or "Ath"

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
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
    torch.cuda.manual_seed_all(seed)


def to_dense_matrix(x):
    if isinstance(x, np.ndarray):
        return x
    return x.toarray()


def get_count_matrix(adata):
    if "counts" in adata.layers:
        print("Using raw counts from adata.layers['counts'].")
        return adata.layers["counts"]

    if adata.raw is not None:
        print("Using raw counts from adata.raw.X.")
        return adata.raw.X

    print("\nWARNING:")
    print("No adata.layers['counts'] and no adata.raw found.")
    print("Falling back to adata.X.")
    print("This is only correct for Negative Binomial loss if adata.X contains raw counts.")
    print("If adata.X is normalized/log1p, NB loss is not appropriate.\n")

    return adata.X


def check_if_counts_look_integer(X_np, n_values: int = 10000):
    flat = X_np.ravel()

    if flat.size == 0:
        print("WARNING: Matrix is empty.")
        return

    sample = flat[: min(n_values, flat.size)]
    min_value = sample.min()
    max_value = sample.max()
    integer_like = np.allclose(sample, np.round(sample), atol=1e-6)

    print("\nCount-matrix diagnostic:")
    print(f"Sample min value: {min_value}")
    print(f"Sample max value: {max_value}")
    print(f"Sample values look integer-like: {integer_like}")

    if min_value < 0:
        print("WARNING: Negative values found. Raw counts should not be negative.")

    if not integer_like:
        print("WARNING: Values do not look integer-like.")
        print("This may mean you are using normalized/log1p data instead of raw counts.")
        print("Negative Binomial loss is usually intended for raw count data.")

    print()


def get_dataloaders(
    h5ad_path: str = H5AD_PATH,
    batch_size: int = 256,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
):
    set_seed(seed)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"AnnData shape: {adata.shape}")

    X_counts = get_count_matrix(adata)
    X_counts = to_dense_matrix(X_counts)

    check_if_counts_look_integer(X_counts)

    X_counts = torch.tensor(X_counts, dtype=torch.float32)
    dataset = TensorDataset(X_counts)

    n_total = len(dataset)
    n_train = int(TRAIN_FRAC * n_total)
    n_val = int(VAL_FRAC * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    input_dimension = X_counts.shape[1]

    print(f"Train cells: {len(train_set)}")
    print(f"Val cells:   {len(val_set)}")
    print(f"Test cells:  {len(test_set)}")
    print(f"Genes:       {input_dimension}")

    return train_loader, val_loader, test_loader, input_dimension, adata


if __name__ == "__main__":
    train_loader, val_loader, test_loader, input_dim, adata = get_dataloaders()

    print("\nData loading test complete.")
    print(f"Input dimension: {input_dim}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
