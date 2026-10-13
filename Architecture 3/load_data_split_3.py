
"""
load_data_split_3.py

Sparse-friendly loader for the normalized PBMC data used by Architecture_3.

Dataset:
    PBMC/PBMC_68k_filtered_normalized.h5ad
"""

import random

import numpy as np
import scanpy as sc
import torch

from scipy import sparse
from torch.utils.data import Dataset, DataLoader, random_split

from ema_pbmc_config_3 import (
    H5AD_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_FRACTION,
    VAL_FRACTION,
    TEST_FRACTION,
    project_path,
)


def set_seed(
    seed: int = RANDOM_SEED,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_csr_or_ndarray(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr()

    return np.asarray(matrix)


def dense_row(matrix, index):
    row = matrix[index]

    if sparse.issparse(row):
        row = row.toarray()

    return np.asarray(
        row,
        dtype=np.float32,
    ).reshape(-1)


def sample_matrix(
    matrix,
    n_rows: int = 100,
    n_cols: int = 100,
):
    sample = matrix[
        : min(n_rows, matrix.shape[0]),
        : min(n_cols, matrix.shape[1]),
    ]

    if sparse.issparse(sample):
        sample = sample.toarray()

    return np.asarray(sample, dtype=np.float32)


def check_normalized_expression(matrix) -> None:
    sample = sample_matrix(matrix)

    if not np.isfinite(sample).all():
        raise ValueError(
            "Expression matrix contains NaN or infinite values."
        )

    minimum = float(sample.min())
    maximum = float(sample.max())
    integer_like = bool(
        np.allclose(sample, np.round(sample), atol=1e-6)
    )

    print("\nExpression-matrix diagnostic:")
    print(f"Sample min value: {minimum}")
    print(f"Sample max value: {maximum}")
    print(f"Sample values integer-like: {integer_like}")

    if minimum < 0:
        raise ValueError(
            "Architecture_3 expects non-negative normalized expression."
        )

    if integer_like:
        print(
            "NOTE: The sampled values look integer-like. Confirm this is the "
            "normalized PBMC file rather than the raw-count file."
        )

    print()


class SparseExpressionDataset(Dataset):
    """
    Convert only requested cells to dense float32 tensors.
    """

    def __init__(self, matrix):
        self.matrix = ensure_csr_or_ndarray(matrix)
        self.n_obs, self.n_vars = self.matrix.shape

    def __len__(self):
        return self.n_obs

    def __getitem__(self, index):
        return (
            torch.from_numpy(
                dense_row(self.matrix, index)
            ),
        )


def load_adata_and_dataset(
    h5ad_path: str = H5AD_PATH,
):
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading AnnData: {h5ad_path}")

    adata = sc.read_h5ad(h5ad_path)
    matrix = ensure_csr_or_ndarray(adata.X)

    print(f"AnnData shape: {adata.shape}")
    print(f"Matrix storage: {type(matrix).__name__}")
    print(f"Matrix dtype: {matrix.dtype}")

    check_normalized_expression(matrix)

    return adata, SparseExpressionDataset(matrix)


def split_dataset(
    dataset,
    seed: int = RANDOM_SEED,
):
    n_total = len(dataset)
    n_train = int(TRAIN_FRACTION * n_total)
    n_val = int(VAL_FRACTION * n_total)
    n_test = n_total - n_train - n_val

    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )


def make_dataloaders(
    split,
    batch_size: int,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
):
    train_set, val_set, test_set = split

    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    return train_loader, val_loader, test_loader


def get_dataloaders(
    h5ad_path: str = H5AD_PATH,
    batch_size: int = 256,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
):
    set_seed(seed)

    adata, dataset = load_adata_and_dataset(h5ad_path)
    split = split_dataset(dataset, seed)

    loaders = make_dataloaders(
        split=split,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )

    print(f"Train cells: {len(split[0])}")
    print(f"Validation cells: {len(split[1])}")
    print(f"Test cells: {len(split[2])}")
    print(f"Genes: {dataset.n_vars}")

    return (*loaders, dataset.n_vars, adata)


def make_full_dataloader_from_adata(
    adata,
    batch_size: int = 512,
    num_workers: int = 0,
):
    dataset = SparseExpressionDataset(adata.X)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, dataset.n_vars


if __name__ == "__main__":
    train_loader, val_loader, test_loader, input_dim, adata = (
        get_dataloaders()
    )

    print("\nData loading test complete.")
    print(f"Input dimension: {input_dim}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
