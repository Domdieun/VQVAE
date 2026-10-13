"""
compare_clusters_to_annotations_4.py

Compare orthogonal-codebook VQ-VAE cluster assignments against known cell type
annotations.

Works for both:
- PBMC
- Arabidopsis / Ath

Run after:
    python extract_cluster_assignments_4.py

Outputs:
1. merged_assignments_annotations_orthogonal_mse.csv
2. cluster_vs_celltype_contingency_counts_orthogonal_mse.csv
3. cluster_vs_celltype_contingency_percent_orthogonal_mse.csv
4. cluster_purity_summary_orthogonal_mse.csv
5. clustering_metrics_orthogonal_mse.txt
6. cluster_vs_celltype_heatmap_percent_orthogonal_mse.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from load_data_split_4 import project_path, PROJECT_ROOT


# CONFIG

DATASET_NAME = "PBMC"  # change to "PBMC" or "Ath"

DATASET_CONFIGS = {
    "PBMC": {
        "assignments_csv": project_path(
            "PBMC/vqvae_runs/train_val_test_orthogonal_mse/barcode_cluster_assignments_final_optuna_orthogonal_mse.csv"),
        "annotation_path": project_path("PBMC/68k_pbmc_barcodes_annotation.tsv"),
        "annotation_has_header": True,
        "annotation_sep": "\t",
        "annotation_names": None,
        "annotation_barcode_col": "barcodes",
        "annotation_label_col": "celltype",
        "output_dir": project_path("PBMC/vqvae_runs/train_val_test_orthogonal_mse/cluster_annotation_comparison"),
    },
    "Ath": {
        "assignments_csv": project_path(
            "Ath/vqvae_runs/train_val_test_orthogonal_mse/barcode_cluster_assignments_final_optuna_orthogonal_mse.csv"),
        "annotation_path": project_path("Ath/Shahan_annotations.csv"),
        "annotation_has_header": True,
        "annotation_sep": ",",
        "annotation_names": None,
        "annotation_barcode_col": "cell_barcode_id",
        "annotation_label_col": "celltype.anno",
        "output_dir": project_path("Ath/vqvae_runs/train_val_test_orthogonal_mse/cluster_annotation_comparison"),
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


def check_required_columns(df, columns, df_name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def load_assignments():
    assignments_df = pd.read_csv(ASSIGNMENTS_CSV)
    check_required_columns(
        assignments_df,
        [ASSIGNMENT_BARCODE_COL, ASSIGNMENT_CLUSTER_COL],
        "Assignments CSV",
    )
    assignments_df[ASSIGNMENT_BARCODE_COL] = assignments_df[ASSIGNMENT_BARCODE_COL].astype(str).str.strip()
    assignments_df[ASSIGNMENT_CLUSTER_COL] = assignments_df[ASSIGNMENT_CLUSTER_COL].astype(str)
    return assignments_df


def load_annotations():
    if ANNOTATION_HAS_HEADER:
        annotations_df = pd.read_csv(ANNOTATION_PATH, sep=ANNOTATION_SEP, low_memory=False)
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
    return str(x).strip()


def get_ath_barcode_core_from_assignment(x):
    return str(x).split("-1_")[0]


def get_ath_sample_from_assignment(x):
    parts = str(x).split("-1_")
    if len(parts) < 2:
        return None
    return parts[1]


def get_ath_barcode_core_from_annotation(x):
    # Handles examples like AAACCCAAGCCATTCA_1 by returning AAACCCAAGCCATTCA.
    return str(x).rsplit("_", 1)[0]


def merge_pbmc(assignments_df, annotations_df):
    assignments_df = assignments_df.copy()
    annotations_df = annotations_df.copy()

    assignments_df["merge_key"] = assignments_df[ASSIGNMENT_BARCODE_COL].map(get_pbmc_barcode)
    annotations_df["merge_key"] = annotations_df[ANNOTATION_BARCODE_COL].map(get_pbmc_barcode)

    return assignments_df.merge(annotations_df, on="merge_key", how="inner")


def merge_ath(assignments_df, annotations_df):
    assignments_df = assignments_df.copy()
    annotations_df = annotations_df.copy()

    assignments_df["barcode_core"] = assignments_df[ASSIGNMENT_BARCODE_COL].map(get_ath_barcode_core_from_assignment)
    assignments_df["adata_sample"] = assignments_df[ASSIGNMENT_BARCODE_COL].map(get_ath_sample_from_assignment)
    assignments_df["annotation_sample"] = assignments_df["adata_sample"].map(ATH_SAMPLE_MAP)

    missing_sample_map = assignments_df["annotation_sample"].isna().sum()
    if missing_sample_map > 0:
        print(
            f"Warning: {missing_sample_map} assignment rows have samples not found in ATH_SAMPLE_MAP."
        )

    assignments_df["merge_key"] = (
        assignments_df["barcode_core"] + "__" + assignments_df["annotation_sample"].astype(str)
    )

    check_required_columns(annotations_df, ["orig.ident"], "Ath annotation file")

    annotations_df["barcode_core"] = annotations_df[ANNOTATION_BARCODE_COL].map(get_ath_barcode_core_from_annotation)
    annotations_df["annotation_sample"] = annotations_df["orig.ident"].astype(str)
    annotations_df["merge_key"] = annotations_df["barcode_core"] + "__" + annotations_df["annotation_sample"]

    return assignments_df.merge(
        annotations_df,
        on="merge_key",
        how="inner",
        suffixes=("_assignment", "_annotation"),
    )


def compute_cluster_purity(merged_df, cluster_col, label_col):
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

        rows.append(
            {
                "cluster_index": cluster_id,
                "cluster_size": cluster_size,
                "dominant_celltype": dominant_label,
                "dominant_count": dominant_count,
                "purity": purity,
                "n_unique_celltypes": subdf[label_col].nunique(),
            }
        )

    purity_df = pd.DataFrame(rows).sort_values("cluster_index").reset_index(drop=True)
    overall_purity = total_correct / total_cells if total_cells > 0 else np.nan
    return purity_df, overall_purity


def make_heatmap(contingency_percent, output_path):
    plt.figure(figsize=(14, 8))
    plt.imshow(contingency_percent.values, aspect="auto")
    plt.colorbar(label="Percentage of cells within VQ-VAE cluster (%)")

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
    plt.ylabel("VQ-VAE cluster index")
    plt.title(f"Orthogonal VQ-VAE MSE Clusters vs Known {DATASET_NAME} Cell Types (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Assignments CSV: {ASSIGNMENTS_CSV}")
    print(f"Annotation path: {ANNOTATION_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Model/loss type: Orthogonal VQ-VAE with MSE reconstruction")

    if not os.path.exists(ASSIGNMENTS_CSV):
        raise FileNotFoundError(f"Could not find assignments CSV: {ASSIGNMENTS_CSV}")
    if not os.path.exists(ANNOTATION_PATH):
        raise FileNotFoundError(f"Could not find annotation file: {ANNOTATION_PATH}")

    assignments_df = load_assignments()
    annotations_df = load_annotations()

    if DATASET_NAME == "Ath":
        merged_df = merge_ath(assignments_df, annotations_df)
    elif DATASET_NAME == "PBMC":
        merged_df = merge_pbmc(assignments_df, annotations_df)
    else:
        raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

    if merged_df.empty:
        raise ValueError("No matching barcodes found between assignments and annotations.")

    print(f"Assignments rows: {len(assignments_df)}")
    print(f"Annotation rows:   {len(annotations_df)}")
    print(f"Merged rows before unknown filter: {len(merged_df)}")

    merged_df = merged_df[merged_df[ANNOTATION_LABEL_COL].notna()].copy()
    merged_df[ANNOTATION_LABEL_COL] = merged_df[ANNOTATION_LABEL_COL].astype(str)

    if EXCLUDE_UNKNOWN:
        before = len(merged_df)
        merged_df = merged_df[merged_df[ANNOTATION_LABEL_COL] != UNKNOWN_LABEL].copy()
        print(f"Rows after excluding {UNKNOWN_LABEL}: {len(merged_df)} / {before}")

    if merged_df.empty:
        raise ValueError("No rows remain after filtering Unknown annotations.")

    ari = adjusted_rand_score(
        merged_df[ANNOTATION_LABEL_COL],
        merged_df[ASSIGNMENT_CLUSTER_COL],
    )
    nmi = normalized_mutual_info_score(
        merged_df[ANNOTATION_LABEL_COL],
        merged_df[ASSIGNMENT_CLUSTER_COL],
    )

    purity_df, overall_purity = compute_cluster_purity(
        merged_df,
        ASSIGNMENT_CLUSTER_COL,
        ANNOTATION_LABEL_COL,
    )

    contingency_counts = pd.crosstab(
        merged_df[ASSIGNMENT_CLUSTER_COL],
        merged_df[ANNOTATION_LABEL_COL],
    )

    contingency_percent = contingency_counts.div(contingency_counts.sum(axis=1), axis=0) * 100

    merged_path = os.path.join(OUTPUT_DIR, "merged_assignments_annotations_orthogonal_mse.csv")
    counts_path = os.path.join(OUTPUT_DIR, "cluster_vs_celltype_contingency_counts_orthogonal_mse.csv")
    percent_path = os.path.join(OUTPUT_DIR, "cluster_vs_celltype_contingency_percent_orthogonal_mse.csv")
    purity_path = os.path.join(OUTPUT_DIR, "cluster_purity_summary_orthogonal_mse.csv")
    metrics_path = os.path.join(OUTPUT_DIR, "clustering_metrics_orthogonal_mse.txt")
    heatmap_path = os.path.join(OUTPUT_DIR, "cluster_vs_celltype_heatmap_percent_orthogonal_mse.png")

    merged_df.to_csv(merged_path, index=False)
    contingency_counts.to_csv(counts_path)
    contingency_percent.to_csv(percent_path)
    purity_df.to_csv(purity_path, index=False)
    make_heatmap(contingency_percent, heatmap_path)

    with open(metrics_path, "w") as f:
        f.write("Orthogonal VQ-VAE clustering metrics\n")
        f.write("Reconstruction loss: MSE\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Assignments rows: {len(assignments_df)}\n")
        f.write(f"Annotation rows: {len(annotations_df)}\n")
        f.write(f"Merged rows used for metrics: {len(merged_df)}\n")
        f.write(f"ARI: {ari}\n")
        f.write(f"NMI: {nmi}\n")
        f.write(f"Overall purity: {overall_purity}\n")

    print("\nMetrics:")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")
    print(f"Overall purity: {overall_purity:.6f}")
    print("\nSaved outputs:")
    print(merged_path)
    print(counts_path)
    print(percent_path)
    print(purity_path)
    print(metrics_path)
    print(heatmap_path)


if __name__ == "__main__":
    main()
