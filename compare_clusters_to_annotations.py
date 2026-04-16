"""
compare_clusters_to_annotations.py

Compare VQ-VAE cluster assignments to known PBMC cell type annotations.

Inputs:
    PBMC/barcode_cluster_assignments.csv
    68k_pbmc_barcodes_annotation.tsv

Outputs:
    PBMC/cluster_annotation_comparison/merged_assignments_annotations.csv
    PBMC/cluster_annotation_comparison/cluster_vs_celltype_contingency.csv
    PBMC/cluster_annotation_comparison/cluster_purity_summary.csv
    PBMC/cluster_annotation_comparison/cluster_vs_celltype_heatmap.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# -----------------------------
# CONFIG
# -----------------------------
ASSIGNMENTS_CSV = "PBMC/barcode_cluster_assignments.csv"
ANNOTATION_TSV = "68k_pbmc_barcodes_annotation.tsv"
OUTPUT_DIR = "PBMC/cluster_annotation_comparison"

ASSIGNMENT_BARCODE_COL = "barcode"
ASSIGNMENT_CLUSTER_COL = "cluster_index"

ANNOTATION_BARCODE_COL = "barcodes"
ANNOTATION_LABEL_COL = "celltype"


# -----------------------------
# HELPERS
# -----------------------------
def compute_cluster_purity(merged_df, cluster_col, label_col):
    """
    Compute per-cluster purity and overall purity.
    Purity(cluster) = size of dominant cell type in cluster / cluster size
    """
    summary_rows = []
    total_correct = 0
    total_cells = len(merged_df)

    for cluster_id, subdf in merged_df.groupby(cluster_col):
        counts = subdf[label_col].value_counts()
        dominant_label = counts.idxmax()
        dominant_count = counts.max()
        cluster_size = len(subdf)
        purity = dominant_count / cluster_size if cluster_size > 0 else np.nan

        total_correct += dominant_count

        summary_rows.append({
            "cluster_index": cluster_id,
            "cluster_size": cluster_size,
            "dominant_celltype": dominant_label,
            "dominant_count": dominant_count,
            "purity": purity,
            "n_unique_celltypes": subdf[label_col].nunique(),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_index").reset_index(drop=True)
    overall_purity = total_correct / total_cells if total_cells > 0 else np.nan
    return summary_df, overall_purity


def check_required_columns(df, required_cols, df_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ASSIGNMENTS_CSV):
        raise FileNotFoundError(f"Could not find assignments CSV: {ASSIGNMENTS_CSV}")

    if not os.path.exists(ANNOTATION_TSV):
        raise FileNotFoundError(f"Could not find annotation TSV: {ANNOTATION_TSV}")

    print("Loading cluster assignments...")
    assignments_df = pd.read_csv(ASSIGNMENTS_CSV)
    check_required_columns(
        assignments_df,
        [ASSIGNMENT_BARCODE_COL, ASSIGNMENT_CLUSTER_COL],
        "Assignments CSV"
    )

    print("Loading annotation file...")
    annotations_df = pd.read_csv(ANNOTATION_TSV, sep="\t")
    check_required_columns(
        annotations_df,
        [ANNOTATION_BARCODE_COL, ANNOTATION_LABEL_COL],
        "Annotation TSV"
    )

    print("Preparing barcode columns...")
    assignments_df[ASSIGNMENT_BARCODE_COL] = assignments_df[ASSIGNMENT_BARCODE_COL].astype(str).str.strip()
    annotations_df[ANNOTATION_BARCODE_COL] = annotations_df[ANNOTATION_BARCODE_COL].astype(str).str.strip()

    print("Merging assignments with annotations...")
    merged_df = assignments_df.merge(
        annotations_df,
        left_on=ASSIGNMENT_BARCODE_COL,
        right_on=ANNOTATION_BARCODE_COL,
        how="inner"
    )

    if merged_df.empty:
        raise ValueError(
            "Merged dataframe is empty. Check whether barcode formats match between "
            "the assignments CSV and annotation TSV."
        )

    print(f"Assignments rows: {len(assignments_df)}")
    print(f"Annotation rows: {len(annotations_df)}")
    print(f"Merged rows: {len(merged_df)}")

    # Contingency table
    contingency = pd.crosstab(
        merged_df[ASSIGNMENT_CLUSTER_COL],
        merged_df[ANNOTATION_LABEL_COL]
    )

    # Cluster purity summary
    purity_summary_df, overall_purity = compute_cluster_purity(
        merged_df,
        cluster_col=ASSIGNMENT_CLUSTER_COL,
        label_col=ANNOTATION_LABEL_COL,
    )

    # ARI / NMI
    ari = adjusted_rand_score(
        merged_df[ANNOTATION_LABEL_COL],
        merged_df[ASSIGNMENT_CLUSTER_COL]
    )
    nmi = normalized_mutual_info_score(
        merged_df[ANNOTATION_LABEL_COL],
        merged_df[ASSIGNMENT_CLUSTER_COL]
    )

    # Save outputs
    merged_csv = os.path.join(OUTPUT_DIR, "merged_assignments_annotations.csv")
    contingency_csv = os.path.join(OUTPUT_DIR, "cluster_vs_celltype_contingency.csv")
    purity_csv = os.path.join(OUTPUT_DIR, "cluster_purity_summary.csv")
    heatmap_png = os.path.join(OUTPUT_DIR, "cluster_vs_celltype_heatmap.png")

    merged_df.to_csv(merged_csv, index=False)
    contingency.to_csv(contingency_csv)
    purity_summary_df.to_csv(purity_csv, index=False)

    # Plot heatmap
    plt.figure(figsize=(14, 8))
    plt.imshow(contingency.values, aspect="auto")
    plt.colorbar(label="Number of cells")
    plt.xticks(
        ticks=np.arange(contingency.shape[1]),
        labels=contingency.columns,
        rotation=90
    )
    plt.yticks(
        ticks=np.arange(contingency.shape[0]),
        labels=contingency.index
    )
    plt.xlabel("Annotated cell type")
    plt.ylabel("Predicted cluster index")
    plt.title("Cluster vs Cell Type Contingency Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_png, dpi=200)
    plt.close()

    print("\n=== Comparison summary ===")
    print(f"Overall purity: {overall_purity:.4f}")
    print(f"ARI:            {ari:.4f}")
    print(f"NMI:            {nmi:.4f}")

    print("\nPer-cluster purity:")
    print(purity_summary_df)

    print("\nSaved files:")
    print(merged_csv)
    print(contingency_csv)
    print(purity_csv)
    print(heatmap_png)


if __name__ == "__main__":
    main()