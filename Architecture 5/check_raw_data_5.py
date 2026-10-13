"""
check_raw_data_5.py

Quickly inspect whether the selected h5ad is suitable for Architecture_5
NB/ZINB training with observed library-size scaling.
"""

import os

import numpy as np
import scanpy as sc

from load_data_split_5 import (
    project_path,
    PROJECT_ROOT,
    get_count_matrix,
    ensure_csr_or_ndarray,
    compute_library_sizes,
    sample_values_for_check,
    read_count_adata,
)


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

    adata = read_count_adata(H5AD_PATH, dataset_name=DATASET_NAME)

    print("\n=== AnnData ===")
    print(adata)
    print("Shape:", adata.shape)
    print("Layers:", adata.layers.keys())
    print("Has raw:", adata.raw is not None)
    print("obs columns:", list(adata.obs.columns))
    print("var columns:", list(adata.var.columns))

    X = ensure_csr_or_ndarray(get_count_matrix(adata))
    sample = sample_values_for_check(X, n_rows=100, n_cols=100)

    print("\n=== Count sample ===")
    print("Min:", sample.min())
    print("Max:", sample.max())
    print("First values:")
    print(sample[:5, :5])

    integer_like = np.allclose(sample, np.round(sample), atol=1e-6)
    print("Integer-like:", integer_like)

    library_sizes = compute_library_sizes(X)

    print("\n=== Library sizes ===")
    print("Min:", float(library_sizes.min()))
    print("Median:", float(np.median(library_sizes)))
    print("Mean:", float(library_sizes.mean()))
    print("Max:", float(library_sizes.max()))

    if not integer_like:
        print("\nWARNING: Values do not look like raw counts.")
        print("Architecture_5 expects raw counts for NB/ZINB loss.")


if __name__ == "__main__":
    main()
