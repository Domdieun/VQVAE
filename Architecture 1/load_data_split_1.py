"""
load_data_split_1.py

Generic AnnData loader for VQ-VAE training.

Works for both:
- PBMC preprocessed h5ad
- Arabidopsis preprocessed h5ad

The only required change is H5AD_PATH.
"""

import random
import os
from pathlib import Path
import numpy as np
import scanpy as sc
import torch

from torch.utils.data import TensorDataset, DataLoader, random_split


DATASET_NAME = os.environ.get("VQVAE_DATASET", "PBMC")


def guess_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, *script_dir.parents, Path.cwd(), *Path.cwd().parents):
        if (candidate / "PBMC").is_dir() or (candidate / "Ath").is_dir():
            return candidate
    return script_dir


PROJECT_ROOT = Path(os.environ.get("VQVAE_PROJECT_ROOT", guess_project_root())).resolve()

H5AD_PATHS = {
    "PBMC": str(PROJECT_ROOT / "PBMC" / "PBMC_68k_filtered_normalized.h5ad"),
    "Ath": str(PROJECT_ROOT / "Ath" / "combined_cellbender_filter_matrix_main_normalized.h5ad"),
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


def get_dataloaders(
    h5ad_path: str = H5AD_PATH,
    batch_size: int = 256,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
):
    set_seed(seed)

    print(f"Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"AnnData shape: {adata.shape}")

    X = to_dense_matrix(adata.X)
    X = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X)

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

    input_dimension = X.shape[1]

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