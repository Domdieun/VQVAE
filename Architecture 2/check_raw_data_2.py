"""
check_raw_data_2.py

Quickly inspect whether the selected h5ad looks suitable for Architecture_2
Negative Binomial training.
"""

import os
import numpy as np
import scanpy as sc

from load_data_split_2 import project_path, PROJECT_ROOT, get_count_matrix


DATASET_NAME = "PBMC"  # change to "PBMC" or "Ath"

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Path: {H5AD_PATH}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")

    adata = sc.read_h5ad(H5AD_PATH)

    print(adata)
    print("Shape:", adata.shape)
    print("Layers:", adata.layers.keys())
    print("Has raw:", adata.raw is not None)

    X = get_count_matrix(adata)

    sample = X[:100, :100]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    sample = np.asarray(sample)

    print("Min:", sample.min())
    print("Max:", sample.max())
    print("First values:")
    print(sample[:5, :5])

    integer_like = np.allclose(sample, np.round(sample), atol=1e-6)
    print("Integer-like:", integer_like)


if __name__ == "__main__":
    main()
