"""
load_data_split_5.py

Sparse-friendly AnnData loader for Architecture_5.

Architecture_5 uses a scVI-style NB/ZINB likelihood:
    mu = observed_library_size * px_scale

Therefore each minibatch must contain:
    1. raw counts x
    2. observed library size for each cell

Important:
    Use raw counts, not normalized/log1p values.
    This loader keeps sparse matrices sparse and densifies only one minibatch
    at a time to avoid loading the full PBMC/Ath matrix as dense float32.
"""

import os
import random
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from scipy import sparse
from torch.utils.data import Dataset, DataLoader, random_split


def guess_project_root() -> Path:
    """
    Guess the project root containing PBMC/ and Ath/.
    Override manually with:
        export VQVAE_PROJECT_ROOT=/lustre/BIF/nobackup/park022/VQVAE
    """

    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir.parent.parent,
        script_dir.parent,
        script_dir,
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "PBMC").exists() or (candidate / "Ath").exists():
            return candidate

    return script_dir.resolve()


PROJECT_ROOT = Path(os.environ.get("VQVAE_PROJECT_ROOT", guess_project_root())).resolve()


def project_path(relative_path: str) -> str:
    path = Path(relative_path)

    if path.is_absolute():
        return str(path)

    return str(PROJECT_ROOT / path)


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


def get_count_matrix(adata):
    """
    Return raw counts in priority order:
        adata.layers["counts"]
        adata.raw.X
        adata.X

    Falling back to adata.X is correct only if adata.X contains raw counts.
    """

    if "counts" in adata.layers:
        print("Using raw counts from adata.layers['counts'].")
        return adata.layers["counts"]

    if adata.raw is not None:
        print("Using raw counts from adata.raw.X.")
        if adata.raw.n_vars != adata.n_vars:
            missing = adata.var_names.difference(adata.raw.var_names)
            if len(missing):
                raise ValueError(f"Genes missing from adata.raw: {list(missing[:5])}")
            return adata.raw[:, adata.var_names].X
        return adata.raw.X

    print("\nWARNING:")
    print("No adata.layers['counts'] and no adata.raw found.")
    print("Falling back to adata.X.")
    print("This is only correct if adata.X contains raw counts.")
    print("For NB/ZINB loss, do not use normalized/log1p values.\n")

    return adata.X



def read_count_adata(h5ad_path: str, dataset_name: str = None):
    return sc.read_h5ad(h5ad_path)


def ensure_csr_or_ndarray(X):
    if sparse.issparse(X):
        return X.tocsr()

    return np.asarray(X)


def sparse_or_dense_to_numpy_row(X, idx):
    row = X[idx]

    if sparse.issparse(row):
        row = row.toarray()

    row = np.asarray(row, dtype=np.float32).reshape(-1)
    return row


def compute_library_sizes(X):
    if sparse.issparse(X):
        library_sizes = np.asarray(X.sum(axis=1)).ravel()
    else:
        library_sizes = np.asarray(X.sum(axis=1)).ravel()

    library_sizes = library_sizes.astype(np.float32)
    library_sizes = np.maximum(library_sizes, 1.0)

    return library_sizes


def sample_values_for_check(X, n_rows: int = 100, n_cols: int = 100):
    rows = min(n_rows, X.shape[0])
    cols = min(n_cols, X.shape[1])
    sample = X[:rows, :cols]

    if sparse.issparse(sample):
        sample = sample.toarray()

    return np.asarray(sample, dtype=np.float32)


def check_if_counts_look_integer(X, n_rows: int = 100, n_cols: int = 100):
    sample = sample_values_for_check(X, n_rows=n_rows, n_cols=n_cols)

    flat = sample.ravel()

    if flat.size == 0:
        print("WARNING: Matrix sample is empty.")
        return

    min_value = float(flat.min())
    max_value = float(flat.max())
    integer_like = bool(np.allclose(flat, np.round(flat), atol=1e-6))

    print("\nCount-matrix diagnostic:")
    print(f"Sample min value: {min_value}")
    print(f"Sample max value: {max_value}")
    print(f"Sample values look integer-like: {integer_like}")

    if min_value < 0:
        print("WARNING: Negative values found. Raw counts should not be negative.")

    if not integer_like:
        print("WARNING: Values do not look integer-like.")
        print("This may mean you are using normalized/log1p data instead of raw counts.")
        print("Architecture_5 expects raw counts for NB/ZINB training.")

    print()


class CountLibraryDataset(Dataset):
    """
    Dataset returning:
        x_counts, library_size

    It stores X as sparse CSR if possible and converts only requested rows
    to dense arrays.
    """

    def __init__(self, X):
        self.X = ensure_csr_or_ndarray(X)
        self.library_sizes = compute_library_sizes(self.X)
        self.n_obs, self.n_vars = self.X.shape

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        x = sparse_or_dense_to_numpy_row(self.X, idx)
        library_size = np.float32(self.library_sizes[idx])

        return (
            torch.from_numpy(x),
            torch.tensor(library_size, dtype=torch.float32),
        )


def get_dataloaders(
    h5ad_path: str = H5AD_PATH,
    batch_size: int = 256,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
    dataset_name: str = None,
):
    set_seed(seed)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading AnnData from: {h5ad_path}")

    adata = read_count_adata(h5ad_path, dataset_name=dataset_name)
    print(f"AnnData shape: {adata.shape}")

    X_counts = get_count_matrix(adata)
    X_counts = ensure_csr_or_ndarray(X_counts)

    check_if_counts_look_integer(X_counts)

    library_sizes = compute_library_sizes(X_counts)
    print("Library-size diagnostic:")
    print(f"Min library size:    {library_sizes.min():.2f}")
    print(f"Median library size: {np.median(library_sizes):.2f}")
    print(f"Max library size:    {library_sizes.max():.2f}")
    print()

    dataset = CountLibraryDataset(X_counts)

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
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    input_dimension = dataset.n_vars

    print(f"Train cells: {len(train_set)}")
    print(f"Val cells:   {len(val_set)}")
    print(f"Test cells:  {len(test_set)}")
    print(f"Genes:       {input_dimension}")

    return train_loader, val_loader, test_loader, input_dimension, adata


def make_full_dataloader_from_adata(
    adata,
    batch_size: int = 512,
    num_workers: int = 0,
):
    X_counts = get_count_matrix(adata)
    X_counts = ensure_csr_or_ndarray(X_counts)
    dataset = CountLibraryDataset(X_counts)

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
    train_loader, val_loader, test_loader, input_dim, adata = get_dataloaders()

    print("\nData loading test complete.")
    print(f"Input dimension: {input_dim}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    x, library_size = next(iter(train_loader))

    print("\nExample batch:")
    print(f"Batch x shape:              {x.shape}")
    print(f"Batch library_size shape:   {library_size.shape}")
    print(f"Batch x dtype:              {x.dtype}")
    print(f"Batch x min:                {x.min().item()}")
    print(f"Batch x max:                {x.max().item()}")
    print(f"Batch library_size min:     {library_size.min().item()}")
    print(f"Batch library_size max:     {library_size.max().item()}")
