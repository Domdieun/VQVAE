"""
compare_clusters_to_annotations_2.py

Compare Architecture_2 VQ-VAE cluster assignments against known cell type annotations.

Works for both:
- PBMC
- Arabidopsis / Ath

Change only DATASET_NAME in the CONFIG section.

This script is used after running:
    extract_cluster_assignments_2.py

It compares:
    barcode_cluster_assignments_final_optuna_architecture_2.csv

against known biological annotations.

Outputs:
1. merged_assignments_annotations.csv
2. cluster_vs_celltype_contingency_counts.csv
3. cluster_vs_celltype_contingency_percent.csv
4. cluster_purity_summary.csv
5. clustering_metrics.txt
6. cluster_vs_celltype_heatmap_percent.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from load_data_split_2 import project_path, PROJECT_ROOT



# CONFIG

DATASET_NAME = "PBMC"   # change to "PBMC" OR "Ath" when needed


DATASET_CONFIGS = {
    "PBMC": {
        "assignments_csv": project_path("PBMC/vqvae_runs/train_val_test_2/barcode_cluster_assignments_final_optuna_architecture_2.csv"),
        "annotation_path": project_path("PBMC/68k_pbmc_barcodes_annotation.tsv"),
        "annotation_has_header": True,
        "annotation_sep": "\t",
        "annotation_names": None,
        "annotation_barcode_col": "barcodes",
        "annotation_label_col": "celltype",
        "output_dir": project_path("PBMC/vqvae_runs/train_val_test_2/cluster_annotation_comparison"),
    },
    "Ath": {
        "assignments_csv": project_path("Ath/vqvae_runs/train_val_test_2/barcode_cluster_assignments_final_optuna_architecture_2.csv"),
        "annotation_path": project_path("Ath/Shahan_annotation.csv"),
        "annotation_has_header": True,
        "annotation_sep": ",",
        "annotation_names": None,
        "annotation_barcode_col": "cell_barcode_id",
        "annotation_label_col": "celltype.anno",
        "output_dir": project_path("Ath/vqvae_runs/train_val_test_2/cluster_annotation_comparison"),
    },
}

ATH_SAMPLE_MAP = {
    "SRX8574728_cellbender_filter": "sc_1",
    "SRX8574729_cellbender_filter": "sc_9_at",
    "SRX8574730_cellbender_filter": "sc_10_at",
    "SRX8574731_cellbender_filter": "sc_11",
    "SRX8574732_cellbender_filter": "sc_12",
    "SRX8574736_cellbender_filter": "sc_30",
    "SRX8574737_cellbender_filter": "sc_31",
    "SRX8574739_cellbender_filter": "sc_37",
    "SRX8574740_cellbender_filter": "sc_40",
    "SRX8574741_cellbender_filter": "sc_51",
    "SRX8574744_cellbender_filter": "col0",
    "SRX8574745_cellbender_filter": "tnw1",
    "SRX8574746_cellbender_filter": "tnw2",
}

CONFIG = DATASET_CONFIGS[DATASET_NAME]

ASSIGNMENTS_CSV = CONFIG["assignments_csv"]
ANNOTATION_PATH = CONFIG["annotation_path"]
ANNOTATION_HAS_HEADER = CONFIG["annotation_has_header"]
ANNOTATION_SEP = CONFIG["annotation_sep"]
ANNOTATION_NAMES = CONFIG["annotation_names"]
ANNOTATION_BARCODE_COL = CONFIG["annotation_barcode_col"]
ANNOTATION_LABEL_COL = CONFIG["annotation_label_col"]
OUTPUT_DIR = CONFIG["output_dir"]

ASSIGNMENT_BARCODE_COL = "barcode"
ASSIGNMENT_CLUSTER_COL = "cluster_index"

EXCLUDE_UNKNOWN = True
UNKNOWN_LABEL = "Unknown"


# HELPER FUNCTIONS


def check_required_columns(df, required_cols, df_name):
    """
    Check whether a DataFrame contains required columns.
    """

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def compute_cluster_purity(merged_df, cluster_col, label_col):
    """
    For each VQ-VAE cluster, find the most common known cell type.

    Purity = dominant cell type count / total cells in that cluster.

    Overall purity is:
        total number of cells belonging to their cluster's dominant label
        divided by
        total number of cells used for metrics
    """

    rows = []
    total_correct = 0
    total_cells = len(merged_df)

    for cluster_id, subdf in merged_df.groupby(cluster_col):
        counts = subdf[label_col].value_counts()

        dominant_label = counts.idxmax()
        dominant_count = counts.max()
        cluster_size = len(subdf)
        purity = dominant_count / cluster_size

        total_correct += dominant_count

        rows.append({
            "cluster_index": cluster_id,
            "cluster_size": cluster_size,
            "dominant_celltype": dominant_label,
            "dominant_count": dominant_count,
            "purity": purity,
            "n_unique_celltypes": subdf[label_col].nunique(),
        })

    purity_df = (
        pd.DataFrame(rows)
        .sort_values("cluster_index")
        .reset_index(drop=True)
    )

    overall_purity = total_correct / total_cells

    return purity_df, overall_purity


def load_assignments():
    """
    Load Architecture_2 VQ-VAE cluster assignments.

    Expected columns:
        barcode
        cluster_index
    """

    assignments_df = pd.read_csv(ASSIGNMENTS_CSV)

    check_required_columns(
        assignments_df,
        [ASSIGNMENT_BARCODE_COL, ASSIGNMENT_CLUSTER_COL],
        "Assignments CSV",
    )

    assignments_df[ASSIGNMENT_BARCODE_COL] = (
        assignments_df[ASSIGNMENT_BARCODE_COL]
        .astype(str)
        .str.strip()
    )

    assignments_df[ASSIGNMENT_CLUSTER_COL] = (
        assignments_df[ASSIGNMENT_CLUSTER_COL]
        .astype(str)
    )

    return assignments_df


def load_annotations():
    """
    Load known PBMC or Ath annotation table.
    """

    if ANNOTATION_HAS_HEADER:
        annotations_df = pd.read_csv(
            ANNOTATION_PATH,
            sep=ANNOTATION_SEP,
            low_memory=False,
        )
    else:
        annotations_df = pd.read_csv(
            ANNOTATION_PATH,
            sep=ANNOTATION_SEP,
            header=None,
            names=ANNOTATION_NAMES,
            low_memory=False,
        )

    check_required_columns(
        annotations_df,
        [ANNOTATION_BARCODE_COL, ANNOTATION_LABEL_COL],
        "Annotation file",
    )

    return annotations_df


def get_pbmc_barcode(x):
    """
    Clean PBMC barcode string.
    """

    return str(x).strip()


def get_ath_barcode_core_from_assignment(x):
    """
    Extract core barcode from the Architecture_2 assignment barcode.

    Example:
        AAACCCAAGCCATTCA-1_SRX8574732_cellbender_filter
    becomes:
        AAACCCAAGCCATTCA
    """

    return str(x).split("-1_")[0]


def get_ath_sample_from_assignment(x):
    """
    Extract sample name from the Architecture_2 assignment barcode.

    Example:
        AAACCCAAGCCATTCA-1_SRX8574732_cellbender_filter
    becomes:
        SRX8574732_cellbender_filter
    """

    parts = str(x).split("-1_")

    if len(parts) < 2:
        return None

    return parts[1]


def get_ath_barcode_core_from_annotation(x):
    """
    Extract core barcode from the Shahan annotation barcode.

    Example:
        AAACCCAAGCCATTCA_1
    becomes:
        AAACCCAAGCCATTCA
    """

    return str(x).rsplit("_", 1)[0]


def merge_pbmc(assignments_df, annotations_df):
    """
    Merge PBMC cluster assignments with PBMC annotations using barcode matching.
    """

    assignments_df = assignments_df.copy()
    annotations_df = annotations_df.copy()

    assignments_df["merge_key"] = (
        assignments_df[ASSIGNMENT_BARCODE_COL]
        .astype(str)
        .map(get_pbmc_barcode)
    )

    annotations_df["merge_key"] = (
        annotations_df[ANNOTATION_BARCODE_COL]
        .astype(str)
        .map(get_pbmc_barcode)
    )

    merged_df = assignments_df.merge(
        annotations_df,
        on="merge_key",
        how="inner",
    )

    return merged_df


def merge_ath(assignments_df, annotations_df):
    """
    Merge Ath cluster assignments with Shahan annotations.

    Arabidopsis needs sample-aware barcode matching.

    Assignment barcode example:
        AAACCCAAGCCATTCA-1_SRX8574732_cellbender_filter

    Annotation barcode example:
        AAACCCAAGCCATTCA_1

    Annotation sample:
        sc_12

    Matching key:
        AAACCCAAGCCATTCA__sc_12
    """

    assignments_df = assignments_df.copy()
    annotations_df = annotations_df.copy()

    assignments_df["barcode_core"] = (
        assignments_df[ASSIGNMENT_BARCODE_COL]
        .astype(str)
        .map(get_ath_barcode_core_from_assignment)
    )

    assignments_df["adata_sample"] = (
        assignments_df[ASSIGNMENT_BARCODE_COL]
        .astype(str)
        .map(get_ath_sample_from_assignment)
    )

    assignments_df["annotation_sample"] = (
        assignments_df["adata_sample"]
        .map(ATH_SAMPLE_MAP)
    )

    missing_sample_map = assignments_df["annotation_sample"].isna().sum()

    if missing_sample_map > 0:
        print(
            f"Warning: {missing_sample_map} assignment rows have samples "
            "not found in ATH_SAMPLE_MAP."
        )

    assignments_df["merge_key"] = (
        assignments_df["barcode_core"] +
        "__" +
        assignments_df["annotation_sample"].astype(str)
    )

    check_required_columns(
        annotations_df,
        ["orig.ident"],
        "Ath annotation file",
    )

    annotations_df["barcode_core"] = (
        annotations_df[ANNOTATION_BARCODE_COL]
        .astype(str)
        .map(get_ath_barcode_core_from_annotation)
    )

    annotations_df["annotation_sample"] = (
        annotations_df["orig.ident"]
        .astype(str)
    )

    annotations_df["merge_key"] = (
        annotations_df["barcode_core"] +
        "__" +
        annotations_df["annotation_sample"]
    )

    merged_df = assignments_df.merge(
        annotations_df,
        on="merge_key",
        how="inner",
        suffixes=("_assignment", "_annotation"),
    )

    return merged_df


def make_heatmap(contingency_percent, output_path):
    """
    Save a heatmap showing the percentage composition of each VQ-VAE cluster.
    """

    plt.figure(figsize=(14, 8))

    plt.imshow(
        contingency_percent.values,
        aspect="auto",
    )

    plt.colorbar(
        label="Percentage of cells within VQ-VAE cluster (%)"
    )

    plt.xticks(
        ticks=np.arange(contingency_percent.shape[1]),
        labels=contingency_percent.columns,
        rotation=90,
    )

    plt.yticks(
        ticks=np.arange(contingency_percent.shape[0]),
        labels=contingency_percent.index,
    )

    plt.xlabel(f"Known {DATASET_NAME} cell type")
    plt.ylabel("Architecture_2 VQ-VAE cluster index")
    plt.title(f"Architecture_2 VQ-VAE Clusters vs Known {DATASET_NAME} Cell Types (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ASSIGNMENTS_CSV):
        raise FileNotFoundError(f"Could not find assignments CSV: {ASSIGNMENTS_CSV}")

    if not os.path.exists(ANNOTATION_PATH):
        raise FileNotFoundError(f"Could not find annotation file: {ANNOTATION_PATH}")

    print(f"Dataset: {DATASET_NAME}")
    print(f"Assignments CSV: {ASSIGNMENTS_CSV}")
    print(f"Annotation path: {ANNOTATION_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nLoading Architecture_2 cluster assignments...")
    assignments_df = load_assignments()

    print("Loading annotations...")
    annotations_df = load_annotations()

    print("Merging assignments with annotations...")

    if DATASET_NAME == "Ath":
        merged_df = merge_ath(assignments_df, annotations_df)
    elif DATASET_NAME == "PBMC":
        merged_df = merge_pbmc(assignments_df, annotations_df)
    else:
        raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

    if merged_df.empty:
        raise ValueError(
            "No matching barcodes found between assignment file and annotation file."
        )

    print(f"\nAssignments rows: {len(assignments_df)}")
    print(f"Annotation rows:   {len(annotations_df)}")
    print(f"Merged rows:       {len(merged_df)}")

    merged_before_unknown_filter = len(merged_df)

    merged_df = merged_df[merged_df[ANNOTATION_LABEL_COL].notna()].copy()
    merged_df[ANNOTATION_LABEL_COL] = (
        merged_df[ANNOTATION_LABEL_COL]
        .astype(str)
        .str.strip()
    )

    if EXCLUDE_UNKNOWN:
        merged_df = merged_df[
            (merged_df[ANNOTATION_LABEL_COL] != UNKNOWN_LABEL)
            & (merged_df[ANNOTATION_LABEL_COL] != "")
        ].copy()

        print(
            f"Rows after excluding {UNKNOWN_LABEL}: "
            f"{len(merged_df)} / {merged_before_unknown_filter}"
        )

    if merged_df.empty:
        raise ValueError(
            "Merged dataframe is empty after filtering. "
            "Check annotation labels or set EXCLUDE_UNKNOWN = False."
        )

    contingency = pd.crosstab(
        merged_df[ASSIGNMENT_CLUSTER_COL],
        merged_df[ANNOTATION_LABEL_COL],
    )

    contingency_percent = (
        contingency.div(contingency.sum(axis=1), axis=0) * 100
    )

    purity_df, overall_purity = compute_cluster_purity(
        merged_df,
        cluster_col=ASSIGNMENT_CLUSTER_COL,
        label_col=ANNOTATION_LABEL_COL,
    )

    ari = adjusted_rand_score(
        merged_df[ANNOTATION_LABEL_COL],
        merged_df[ASSIGNMENT_CLUSTER_COL],
    )

    nmi = normalized_mutual_info_score(
        merged_df[ANNOTATION_LABEL_COL],
        merged_df[ASSIGNMENT_CLUSTER_COL],
    )

    merged_csv = os.path.join(
        OUTPUT_DIR,
        "merged_assignments_annotations_architecture_2.csv",
    )

    contingency_csv = os.path.join(
        OUTPUT_DIR,
        "cluster_vs_celltype_contingency_counts_architecture_2.csv",
    )

    contingency_percent_csv = os.path.join(
        OUTPUT_DIR,
        "cluster_vs_celltype_contingency_percent_architecture_2.csv",
    )

    purity_csv = os.path.join(
        OUTPUT_DIR,
        "cluster_purity_summary_architecture_2.csv",
    )

    heatmap_png = os.path.join(
        OUTPUT_DIR,
        "cluster_vs_celltype_heatmap_percent_architecture_2.png",
    )

    metrics_txt = os.path.join(
        OUTPUT_DIR,
        "clustering_metrics_architecture_2.txt",
    )

    merged_df.to_csv(merged_csv, index=False)
    contingency.to_csv(contingency_csv)
    contingency_percent.to_csv(contingency_percent_csv)
    purity_df.to_csv(purity_csv, index=False)

    with open(metrics_txt, "w") as f:
        f.write("Architecture: Architecture_2\n")
        f.write("Loss: Negative Binomial reconstruction loss + VQ loss\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Assignments CSV: {ASSIGNMENTS_CSV}\n")
        f.write(f"Annotation path: {ANNOTATION_PATH}\n")
        f.write(f"Assignments rows: {len(assignments_df)}\n")
        f.write(f"Annotation rows: {len(annotations_df)}\n")
        f.write(f"Merged rows before Unknown filter: {merged_before_unknown_filter}\n")
        f.write(f"Merged rows used for metrics: {len(merged_df)}\n")
        f.write(f"Excluded Unknown: {EXCLUDE_UNKNOWN}\n")
        f.write(f"Overall purity: {overall_purity:.4f}\n")
        f.write(f"ARI: {ari:.4f}\n")
        f.write(f"NMI: {nmi:.4f}\n")

    make_heatmap(contingency_percent, heatmap_png)

    print("\n=== Architecture_2 comparison summary ===")
    print(f"Rows used for metrics: {len(merged_df)}")
    print(f"Overall purity:        {overall_purity:.4f}")
    print(f"ARI:                   {ari:.4f}")
    print(f"NMI:                   {nmi:.4f}")

    print("\nPer-cluster purity:")
    print(purity_df)

    print("\nSaved files:")
    print(merged_csv)
    print(contingency_csv)
    print(contingency_percent_csv)
    print(purity_csv)
    print(metrics_txt)
    print(heatmap_png)


if __name__ == "__main__":
    main()