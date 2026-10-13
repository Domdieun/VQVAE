
"""Compare Architecture_3 assignments with normal PBMC annotations."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from ema_pbmc_config_3 import (
    ASSIGNMENTS_CSV,
    ANNOTATION_PATH,
    ANNOTATION_SEP,
    ANNOTATION_BARCODE_COL,
    ANNOTATION_LABEL_COL,
    COMPARISON_OUTPUT_DIR,
    UNKNOWN_LABEL,
    EXCLUDE_UNKNOWN,
)


def compute_purity(data):
    rows = []
    correct = 0

    for cluster, group in data.groupby("cluster_index"):
        counts = group[ANNOTATION_LABEL_COL].value_counts()
        dominant_label = counts.idxmax()
        dominant_count = int(counts.max())
        correct += dominant_count

        rows.append({
            "cluster_index": cluster,
            "cluster_size": len(group),
            "dominant_celltype": dominant_label,
            "dominant_count": dominant_count,
            "purity": dominant_count / len(group),
            "n_unique_celltypes": group[
                ANNOTATION_LABEL_COL
            ].nunique(),
        })

    return (
        pd.DataFrame(rows).sort_values(
            "cluster_index"
        ),
        correct / len(data),
    )


def save_heatmap(percent, path):
    plt.figure(figsize=(14, 8))
    image = plt.imshow(
        percent.values,
        aspect="auto",
    )
    plt.colorbar(
        image,
        label="Percentage within EMA VQ cluster (%)",
    )
    plt.xticks(
        np.arange(percent.shape[1]),
        percent.columns,
        rotation=90,
    )
    plt.yticks(
        np.arange(percent.shape[0]),
        percent.index,
    )
    plt.xlabel("Known PBMC cell type")
    plt.ylabel("Architecture_3 cluster")
    plt.title(
        "Architecture_3 clusters vs PBMC annotations"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    for path in [ASSIGNMENTS_CSV, ANNOTATION_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    os.makedirs(
        COMPARISON_OUTPUT_DIR,
        exist_ok=True,
    )

    assignments = pd.read_csv(ASSIGNMENTS_CSV)
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

    assignments["merge_key"] = (
        assignments["barcode"]
        .astype(str)
        .str.strip()
    )
    annotations["merge_key"] = (
        annotations[ANNOTATION_BARCODE_COL]
        .astype(str)
        .str.strip()
    )

    merged = assignments.merge(
        annotations,
        on="merge_key",
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No PBMC barcodes matched between assignments and annotations."
        )

    before_filter = len(merged)

    merged = merged[
        merged[ANNOTATION_LABEL_COL].notna()
    ].copy()

    merged[ANNOTATION_LABEL_COL] = (
        merged[ANNOTATION_LABEL_COL]
        .astype(str)
    )

    if EXCLUDE_UNKNOWN:
        merged = merged[
            merged[ANNOTATION_LABEL_COL] != UNKNOWN_LABEL
        ].copy()

    if merged.empty:
        raise ValueError(
            "No annotated cells remain after filtering."
        )

    counts = pd.crosstab(
        merged["cluster_index"],
        merged[ANNOTATION_LABEL_COL],
    )
    percent = (
        counts.div(counts.sum(axis=1), axis=0)
        * 100
    )

    purity, overall_purity = compute_purity(merged)

    ari = adjusted_rand_score(
        merged[ANNOTATION_LABEL_COL],
        merged["cluster_index"],
    )
    nmi = normalized_mutual_info_score(
        merged[ANNOTATION_LABEL_COL],
        merged["cluster_index"],
    )

    merged.to_csv(
        os.path.join(
            COMPARISON_OUTPUT_DIR,
            "merged_assignments_annotations_ema.csv",
        ),
        index=False,
    )
    counts.to_csv(
        os.path.join(
            COMPARISON_OUTPUT_DIR,
            "cluster_vs_celltype_counts_ema.csv",
        )
    )
    percent.to_csv(
        os.path.join(
            COMPARISON_OUTPUT_DIR,
            "cluster_vs_celltype_percent_ema.csv",
        )
    )
    purity.to_csv(
        os.path.join(
            COMPARISON_OUTPUT_DIR,
            "cluster_purity_summary_ema.csv",
        ),
        index=False,
    )

    save_heatmap(
        percent,
        os.path.join(
            COMPARISON_OUTPUT_DIR,
            "cluster_vs_celltype_heatmap_ema.png",
        ),
    )

    metrics = {
        "architecture": "Architecture_3",
        "assignments_rows": int(len(assignments)),
        "annotation_rows": int(len(annotations)),
        "merged_rows_before_unknown_filter": int(
            before_filter
        ),
        "cells_used_for_metrics": int(len(merged)),
        "number_of_vq_clusters_used": int(
            merged["cluster_index"].nunique()
        ),
        "number_of_celltypes": int(
            merged[ANNOTATION_LABEL_COL].nunique()
        ),
        "overall_purity": float(overall_purity),
        "ari": float(ari),
        "nmi": float(nmi),
    }

    with open(
        os.path.join(
            COMPARISON_OUTPUT_DIR,
            "clustering_metrics_ema.json",
        ),
        "w",
    ) as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
