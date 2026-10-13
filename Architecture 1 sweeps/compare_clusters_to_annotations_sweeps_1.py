"""
compare_clusters_to_annotations_sweeps_1.py

Compare all 25 Architecture_1 factorial-grid runs with biological annotations.
The script saves per-run contingency tables, purity, ARI, NMI, a 25-row result
table, and metric heatmaps across codebook size and training batch size.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from hyperparameter_sweep_config_1 import (
    ANNOTATION_CONFIG,
    ATH_SAMPLE_MAP,
    BATCH_SIZES,
    CODEBOOK_SIZES,
    DATASET_NAME,
    EXCLUDE_UNKNOWN,
    RUN_ROOT,
    UNKNOWN_LABEL,
    build_experiments,
)

ASSIGNMENT_BARCODE_COL = "barcode"
ASSIGNMENT_CLUSTER_COL = "cluster_index"


def check_columns(df, columns, name):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def load_annotations():
    path = Path(ANNOTATION_CONFIG["path"])
    if not path.exists():
        raise FileNotFoundError(f"Could not find annotation file: {path}")

    if ANNOTATION_CONFIG["has_header"]:
        annotation_df = pd.read_csv(
            path,
            sep=ANNOTATION_CONFIG["sep"],
            low_memory=False,
        )
    else:
        annotation_df = pd.read_csv(
            path,
            sep=ANNOTATION_CONFIG["sep"],
            header=None,
            names=ANNOTATION_CONFIG["names"],
            low_memory=False,
        )

    check_columns(
        annotation_df,
        [ANNOTATION_CONFIG["barcode_col"], ANNOTATION_CONFIG["label_col"]],
        "Annotation table",
    )
    return annotation_df


def get_ath_barcode_core_from_assignment(value):
    return str(value).split("-1_")[0]


def get_ath_sample_from_assignment(value):
    parts = str(value).split("-1_", 1)
    return parts[1] if len(parts) == 2 else None


def get_ath_barcode_core_from_annotation(value):
    return str(value).rsplit("_", 1)[0]


def merge_assignments_and_annotations(assignments_df, annotations_df):
    barcode_col = ANNOTATION_CONFIG["barcode_col"]

    if DATASET_NAME == "PBMC":
        assignments_df = assignments_df.copy()
        annotations_df = annotations_df.copy()
        assignments_df["merge_key"] = (
            assignments_df[ASSIGNMENT_BARCODE_COL].astype(str).str.strip()
        )
        annotations_df["merge_key"] = (
            annotations_df[barcode_col].astype(str).str.strip()
        )
        return assignments_df.merge(annotations_df, on="merge_key", how="inner")

    if DATASET_NAME == "Ath":
        assignments_df = assignments_df.copy()
        annotations_df = annotations_df.copy()

        check_columns(annotations_df, ["orig.ident"], "Ath annotation table")

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
            assignments_df["adata_sample"].map(ATH_SAMPLE_MAP)
        )
        assignments_df["merge_key"] = (
            assignments_df["barcode_core"]
            + "__"
            + assignments_df["annotation_sample"].astype(str)
        )

        annotations_df["barcode_core"] = (
            annotations_df[barcode_col]
            .astype(str)
            .map(get_ath_barcode_core_from_annotation)
        )
        annotations_df["annotation_sample"] = annotations_df["orig.ident"].astype(str)
        annotations_df["merge_key"] = (
            annotations_df["barcode_core"]
            + "__"
            + annotations_df["annotation_sample"]
        )

        return assignments_df.merge(
            annotations_df,
            on="merge_key",
            how="inner",
            suffixes=("_assignment", "_annotation"),
        )

    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")


def compute_purity(merged_df, label_col):
    rows = []
    total_dominant = 0

    for cluster_id, subdf in merged_df.groupby(ASSIGNMENT_CLUSTER_COL):
        counts = subdf[label_col].value_counts()
        dominant_label = counts.idxmax()
        dominant_count = int(counts.max())
        cluster_size = len(subdf)
        total_dominant += dominant_count

        rows.append({
            "cluster_index": cluster_id,
            "cluster_size": cluster_size,
            "dominant_celltype": dominant_label,
            "dominant_count": dominant_count,
            "purity": dominant_count / cluster_size,
            "n_unique_celltypes": subdf[label_col].nunique(),
        })

    purity_df = pd.DataFrame(rows).sort_values("cluster_index")
    overall_purity = total_dominant / len(merged_df)
    return purity_df, overall_purity


def save_heatmap(contingency_percent, output_path, title):
    plt.figure(figsize=(14, 8))
    plt.imshow(contingency_percent.values, aspect="auto")
    plt.colorbar(label="Percentage within VQ-VAE cluster (%)")
    plt.xticks(
        np.arange(contingency_percent.shape[1]),
        contingency_percent.columns,
        rotation=90,
    )
    plt.yticks(
        np.arange(contingency_percent.shape[0]),
        contingency_percent.index,
    )
    plt.xlabel(f"Known {DATASET_NAME} cell type")
    plt.ylabel("VQ-VAE cluster index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compare_one_run(experiment, annotations_df):
    run_dir = Path(experiment["run_dir"])
    assignments_path = run_dir / "barcode_cluster_assignments.csv"
    output_dir = run_dir / "annotation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not assignments_path.exists():
        print(f"Skipping missing assignments: {assignments_path}")
        return None

    assignments_df = pd.read_csv(assignments_path)
    check_columns(
        assignments_df,
        [ASSIGNMENT_BARCODE_COL, ASSIGNMENT_CLUSTER_COL],
        "Assignments table",
    )

    merged_df = merge_assignments_and_annotations(assignments_df, annotations_df)
    if merged_df.empty:
        raise ValueError(f"No annotation matches for {experiment['run_name']}.")

    label_col = ANNOTATION_CONFIG["label_col"]
    merged_before_filter = len(merged_df)
    merged_df = merged_df[merged_df[label_col].notna()].copy()
    merged_df[label_col] = merged_df[label_col].astype(str).str.strip()
    merged_df = merged_df[merged_df[label_col].ne("") & merged_df[label_col].str.lower().ne("nan")].copy()

    if EXCLUDE_UNKNOWN:
        merged_df = merged_df[
            merged_df[label_col].str.casefold() != UNKNOWN_LABEL.casefold()
        ].copy()

    if merged_df.empty:
        raise ValueError(
            f"No annotated cells remain for {experiment['run_name']} after filtering."
        )

    contingency = pd.crosstab(
        merged_df[ASSIGNMENT_CLUSTER_COL],
        merged_df[label_col],
    )
    contingency_percent = contingency.div(contingency.sum(axis=1), axis=0) * 100
    purity_df, overall_purity = compute_purity(merged_df, label_col)

    ari = adjusted_rand_score(
        merged_df[label_col],
        merged_df[ASSIGNMENT_CLUSTER_COL],
    )
    nmi = normalized_mutual_info_score(
        merged_df[label_col],
        merged_df[ASSIGNMENT_CLUSTER_COL],
    )

    merged_df.to_csv(output_dir / "merged_assignments_annotations.csv", index=False)
    contingency.to_csv(output_dir / "cluster_vs_celltype_counts.csv")
    contingency_percent.to_csv(output_dir / "cluster_vs_celltype_percent.csv")
    purity_df.to_csv(output_dir / "cluster_purity_summary.csv", index=False)

    save_heatmap(
        contingency_percent,
        output_dir / "cluster_vs_celltype_heatmap_percent.png",
        title=(
            f"{DATASET_NAME}: K={experiment['num_embeddings']}, "
            f"batch={experiment['batch_size']}"
        ),
    )

    run_result_path = run_dir / "run_result.json"
    run_result = {}
    if run_result_path.exists():
        with run_result_path.open("r") as f:
            run_result = json.load(f)

    unique_codes_all_cells = int(assignments_df[ASSIGNMENT_CLUSTER_COL].nunique())
    usage_percent_all_cells = 100.0 * unique_codes_all_cells / experiment["num_embeddings"]

    metrics = {
        "run_name": experiment["run_name"],
        "num_embeddings": experiment["num_embeddings"],
        "batch_size": experiment["batch_size"],
        "sweep_membership": ",".join(experiment["sweep_membership"]),
        "assignment_rows": len(assignments_df),
        "merged_rows_before_unknown_filter": merged_before_filter,
        "cells_used_for_metrics": len(merged_df),
        "overall_purity": overall_purity,
        "ari": ari,
        "nmi": nmi,
        "best_val_loss": run_result.get("best_val_loss"),
        "test_total_loss": run_result.get("test_total_loss"),
        "test_recon_loss": run_result.get("test_recon_loss"),
        "test_vq_loss": run_result.get("test_vq_loss"),
        "unique_codes_used": unique_codes_all_cells,
        "codebook_usage_percent": usage_percent_all_cells,
        "unique_codes_used_test": run_result.get("unique_codes_used"),
        "codebook_usage_percent_test": run_result.get("codebook_usage_percent"),
    }

    with (output_dir / "clustering_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"{experiment['run_name']} | "
        f"ARI={ari:.4f} | NMI={nmi:.4f} | purity={overall_purity:.4f}"
    )
    return metrics


def save_grid_heatmap(df, metric, output_path, title, value_format=".4f"):
    """Save a codebook-size x batch-size heatmap for one metric."""
    pivot = (
        df.pivot(index="num_embeddings", columns="batch_size", values=metric)
        .reindex(index=CODEBOOK_SIZES, columns=BATCH_SIZES)
    )

    plt.figure(figsize=(10, 6))
    image = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(image, label=metric.replace("_", " "))
    plt.xticks(np.arange(len(BATCH_SIZES)), BATCH_SIZES)
    plt.yticks(np.arange(len(CODEBOOK_SIZES)), CODEBOOK_SIZES)
    plt.xlabel("Training batch size")
    plt.ylabel("Number of codebook vectors")
    plt.title(title)

    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.iat[row_index, column_index]
            if pd.notna(value):
                plt.text(
                    column_index,
                    row_index,
                    format(value, value_format),
                    ha="center",
                    va="center",
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_best_configuration_summary(comparison_df):
    """Save the best configuration under each useful selection criterion."""
    criteria = [
        ("highest_ari", "ari", False),
        ("highest_nmi", "nmi", False),
        ("highest_purity", "overall_purity", False),
        ("lowest_validation_loss", "best_val_loss", True),
    ]

    rows = []
    for criterion, metric, minimize in criteria:
        valid = comparison_df.dropna(subset=[metric])
        if valid.empty:
            continue
        row = valid.loc[valid[metric].idxmin() if minimize else valid[metric].idxmax()]
        rows.append({
            "criterion": criterion,
            "metric": metric,
            "metric_value": row[metric],
            "run_name": row["run_name"],
            "num_embeddings": int(row["num_embeddings"]),
            "batch_size": int(row["batch_size"]),
            "ari": row.get("ari"),
            "nmi": row.get("nmi"),
            "overall_purity": row.get("overall_purity"),
            "best_val_loss": row.get("best_val_loss"),
            "test_total_loss": row.get("test_total_loss"),
            "codebook_usage_percent": row.get("codebook_usage_percent"),
        })

    pd.DataFrame(rows).to_csv(
        RUN_ROOT / "best_configurations_by_metric.csv",
        index=False,
    )


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    annotations_df = load_annotations()

    metrics_rows = []
    for experiment in build_experiments():
        metrics = compare_one_run(experiment, annotations_df)
        if metrics is not None:
            metrics_rows.append(metrics)

    if not metrics_rows:
        raise RuntimeError("No runs were compared. Extract assignments first.")

    comparison_df = (
        pd.DataFrame(metrics_rows)
        .sort_values(["num_embeddings", "batch_size"])
        .reset_index(drop=True)
    )

    # Keep the former filenames for compatibility, while also saving a name
    # that explicitly describes the new 25-run factorial experiment.
    comparison_df.to_csv(
        RUN_ROOT / "unique_run_comparison_summary.csv",
        index=False,
    )
    comparison_df.to_csv(
        RUN_ROOT / "sweep_comparison_table.csv",
        index=False,
    )
    comparison_df.to_csv(
        RUN_ROOT / "full_factorial_comparison_table.csv",
        index=False,
    )

    heatmap_specs = [
        ("ari", "ari_grid_heatmap.png", "ARI across the 5 x 5 hyperparameter grid", ".4f"),
        ("nmi", "nmi_grid_heatmap.png", "NMI across the 5 x 5 hyperparameter grid", ".4f"),
        (
            "overall_purity",
            "purity_grid_heatmap.png",
            "Cluster purity across the 5 x 5 hyperparameter grid",
            ".4f",
        ),
        (
            "best_val_loss",
            "validation_loss_grid_heatmap.png",
            "Best validation loss across the 5 x 5 hyperparameter grid",
            ".5f",
        ),
        (
            "test_total_loss",
            "test_loss_grid_heatmap.png",
            "Test total loss across the 5 x 5 hyperparameter grid",
            ".5f",
        ),
        (
            "codebook_usage_percent",
            "codebook_usage_grid_heatmap.png",
            "Codebook usage (%) across the 5 x 5 hyperparameter grid",
            ".1f",
        ),
    ]

    for metric, filename, title, value_format in heatmap_specs:
        if metric in comparison_df.columns and comparison_df[metric].notna().any():
            save_grid_heatmap(
                comparison_df,
                metric=metric,
                output_path=RUN_ROOT / filename,
                title=title,
                value_format=value_format,
            )

    save_best_configuration_summary(comparison_df)

    expected_runs = len(CODEBOOK_SIZES) * len(BATCH_SIZES)
    print(
        f"\nCompared {len(comparison_df)}/{expected_runs} full-factorial runs."
    )
    print(
        "Saved final comparison table: "
        f"{RUN_ROOT / 'full_factorial_comparison_table.csv'}"
    )


if __name__ == "__main__":
    main()
