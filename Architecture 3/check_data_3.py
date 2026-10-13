
"""Inspect the normal normalized PBMC input and annotation barcode overlap."""

import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from ema_pbmc_config_3 import (
    H5AD_PATH,
    ANNOTATION_PATH,
    ANNOTATION_SEP,
    ANNOTATION_BARCODE_COL,
    ANNOTATION_LABEL_COL,
)
from load_data_split_3 import (
    check_normalized_expression,
    ensure_csr_or_ndarray,
)


def main():
    for path in [H5AD_PATH, ANNOTATION_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    adata = sc.read_h5ad(H5AD_PATH)
    matrix = ensure_csr_or_ndarray(adata.X)
    check_normalized_expression(matrix)

    annotations = pd.read_csv(
        ANNOTATION_PATH,
        sep=ANNOTATION_SEP,
        low_memory=False,
    )

    required = [
        ANNOTATION_BARCODE_COL,
        ANNOTATION_LABEL_COL,
    ]
    missing = [
        column for column in required
        if column not in annotations.columns
    ]

    if missing:
        raise KeyError(
            f"Annotation file is missing columns: {missing}"
        )

    h5ad_barcodes = set(
        adata.obs_names.astype(str).str.strip()
    )
    annotation_barcodes = set(
        annotations[ANNOTATION_BARCODE_COL]
        .astype(str)
        .str.strip()
    )

    if sparse.issparse(matrix):
        nnz = int(matrix.nnz)
    else:
        nnz = int(np.count_nonzero(matrix))

    report = {
        "h5ad_path": H5AD_PATH,
        "annotation_path": ANNOTATION_PATH,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "matrix_storage": type(matrix).__name__,
        "matrix_dtype": str(matrix.dtype),
        "nonzero_entries": nnz,
        "zero_fraction": float(
            1.0 - nnz / (adata.n_obs * adata.n_vars)
        ),
        "annotation_rows": int(len(annotations)),
        "annotation_celltypes": int(
            annotations[ANNOTATION_LABEL_COL].nunique()
        ),
        "barcode_overlap": int(
            len(h5ad_barcodes & annotation_barcodes)
        ),
        "h5ad_barcodes_without_annotation": int(
            len(h5ad_barcodes - annotation_barcodes)
        ),
        "annotation_barcodes_not_in_h5ad": int(
            len(annotation_barcodes - h5ad_barcodes)
        ),
    }

    print(json.dumps(report, indent=2))
    print("\nAnnotation counts:")
    print(
        annotations[ANNOTATION_LABEL_COL]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()
