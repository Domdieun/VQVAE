
"""
compare_clusters_to_annotations_sweeps_2.py

Compare all 25 Architecture_2 grid runs against known annotations and create
codebook-size x batch-size metric heatmaps.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from hyperparameter_sweep_config_2 import (
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


def load_annotations():
    config = ANNOTATION_CONFIG
    if config["has_header"]:
        return pd.read_csv(config["path"], sep=config["sep"], low_memory=False)

    return pd.read_csv(
        config["path"],
        sep=config["sep"],
        header=None,
        names=config["names"],
        low_memory=False,
    )


def merge_tables(assignments, annotations):
    barcode_col = ANNOTATION_CONFIG["barcode_col"]

    if DATASET_NAME == "PBMC":
        assignments = assignments.copy()
        annotations = annotations.copy()
        assignments["merge_key"] = assignments["barcode"].astype(str).str.strip()
        annotations["merge_key"] = annotations[barcode_col].astype(str).str.strip()
        return assignments.merge(annotations, on="merge_key", how="inner")

    assignments = assignments.copy()
    annotations = annotations.copy()

    assignments["barcode_core"] = (
        assignments["barcode"].astype(str).str.split("-1_").str[0]
    )
    assignments["adata_sample"] = (
        assignments["barcode"].astype(str).str.split("-1_").str[1]
    )
    assignments["annotation_sample"] = assignments["adata_sample"].map(ATH_SAMPLE_MAP)
    assignments["merge_key"] = (
        assignments["barcode_core"]
        + "__"
        + assignments["annotation_sample"].astype(str)
    )

    annotations["barcode_core"] = (
        annotations[barcode_col].astype(str).str.rsplit("_", n=1).str[0]
    )
    annotations["merge_key"] = (
        annotations["barcode_core"]
        + "__"
        + annotations["orig.ident"].astype(str)
    )
    return assignments.merge(annotations, on="merge_key", how="inner")


def compute_purity(merged, label_col):
    rows = []
    correct = 0

    for cluster, group in merged.groupby(ASSIGNMENT_CLUSTER_COL):
        counts = group[label_col].value_counts()
        dominant_label = counts.idxmax()
        dominant_count = int(counts.max())
        correct += dominant_count

        rows.append({
            "cluster_index": cluster,
            "cluster_size": len(group),
            "dominant_celltype": dominant_label,
            "dominant_count": dominant_count,
            "purity": dominant_count / len(group),
            "n_unique_celltypes": group[label_col].nunique(),
        })

    return pd.DataFrame(rows), correct / len(merged)


def save_contingency_heatmap(table, output_path, title):
    plt.figure(figsize=(14, 8))
    plt.imshow(table.values, aspect="auto")
    plt.colorbar(label="Percentage within VQ-VAE cluster (%)")
    plt.xticks(np.arange(table.shape[1]), table.columns, rotation=90)
    plt.yticks(np.arange(table.shape[0]), table.index)
    plt.xlabel(f"Known {DATASET_NAME} cell type")
    plt.ylabel("VQ-VAE cluster")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_grid_heatmap(df, metric, output_path, title, fmt=".4f"):
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
    plt.ylabel("Codebook vectors")
    plt.title(title)

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iat[row, col]
            if pd.notna(value):
                plt.text(col, row, format(value, fmt), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations()
    label_col = ANNOTATION_CONFIG["label_col"]

    rows = []

    for experiment in build_experiments():
        run_dir = Path(experiment["run_dir"])
        assignments_path = run_dir / "barcode_cluster_assignments.csv"

        if not assignments_path.exists():
            print(f"Skipping missing assignments: {assignments_path}")
            continue

        assignments = pd.read_csv(assignments_path)
        merged = merge_tables(assignments, annotations)
        before_unknown = len(merged)

        merged = merged[merged[label_col].notna()].copy()
        merged[label_col] = merged[label_col].astype(str).str.strip()
        merged = merged[
            merged[label_col].ne("") &
            ~merged[label_col].str.casefold().isin(["nan", "none", "null"])
        ].copy()
        if EXCLUDE_UNKNOWN:
            merged = merged[
                merged[label_col].str.casefold() != UNKNOWN_LABEL.casefold()
            ].copy()

        if merged.empty:
            raise ValueError(
                f"No annotated cells remain for {experiment['run_name']}."
            )

        contingency = pd.crosstab(
            merged[ASSIGNMENT_CLUSTER_COL],
            merged[label_col],
        )
        contingency_percent = (
            contingency.div(contingency.sum(axis=1), axis=0) * 100
        )

        purity_df, purity = compute_purity(merged, label_col)
        ari = adjusted_rand_score(
            merged[label_col],
            merged[ASSIGNMENT_CLUSTER_COL],
        )
        nmi = normalized_mutual_info_score(
            merged[label_col],
            merged[ASSIGNMENT_CLUSTER_COL],
        )

        output_dir = run_dir / "annotation_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        merged.to_csv(output_dir / "merged_assignments_annotations.csv", index=False)
        contingency.to_csv(output_dir / "cluster_vs_celltype_counts.csv")
        contingency_percent.to_csv(
            output_dir / "cluster_vs_celltype_percent.csv"
        )
        purity_df.to_csv(output_dir / "cluster_purity_summary.csv", index=False)

        save_contingency_heatmap(
            contingency_percent,
            output_dir / "cluster_vs_celltype_heatmap_percent.png",
            (
                f"{DATASET_NAME}: K={experiment['num_embeddings']}, "
                f"batch={experiment['batch_size']}"
            ),
        )

        run_result = {}
        result_path = run_dir / "run_result.json"
        if result_path.exists():
            with result_path.open() as handle:
                run_result = json.load(handle)

        codes_used_all_cells = int(assignments[ASSIGNMENT_CLUSTER_COL].nunique())
        usage_all_cells = 100.0 * codes_used_all_cells / experiment["num_embeddings"]

        row = {
            "run_name": experiment["run_name"],
            "num_embeddings": experiment["num_embeddings"],
            "batch_size": experiment["batch_size"],
            "assignment_rows": len(assignments),
            "merged_rows_before_unknown_filter": before_unknown,
            "cells_used_for_metrics": len(merged),
            "ari": ari,
            "nmi": nmi,
            "overall_purity": purity,
            "best_val_loss": run_result.get("best_val_loss"),
            "test_total_loss": run_result.get("test_total_loss"),
            "test_recon_loss": run_result.get("test_recon_loss"),
            "test_vq_loss": run_result.get("test_vq_loss"),
            "unique_codes_used": codes_used_all_cells,
            "codebook_usage_percent": usage_all_cells,
            "unique_codes_used_test": run_result.get("unique_codes_used"),
            "codebook_usage_percent_test": run_result.get("codebook_usage_percent"),
        }
        rows.append(row)

        with (output_dir / "clustering_metrics.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        print(
            f"{experiment['run_name']} | "
            f"ARI={ari:.4f} | NMI={nmi:.4f} | purity={purity:.4f}"
        )

    if not rows:
        print("No completed assignment comparisons were found.")
        return

    comparison = pd.DataFrame(rows).sort_values(
        ["num_embeddings", "batch_size"]
    )
    comparison.to_csv(
        RUN_ROOT / "full_factorial_comparison_table.csv",
        index=False,
    )

    if comparison.empty:
        print("No comparison rows were produced.")
        return

    metrics = [
        ("ari", "ARI", ".4f"),
        ("nmi", "NMI", ".4f"),
        ("overall_purity", "Overall purity", ".4f"),
        ("best_val_loss", "Best validation loss", ".4f"),
        ("test_total_loss", "Test total loss", ".4f"),
        ("codebook_usage_percent", "Codebook usage (%)", ".1f"),
    ]

    for metric, title, fmt in metrics:
        if metric in comparison and comparison[metric].notna().any():
            save_grid_heatmap(
                comparison,
                metric,
                RUN_ROOT / f"{metric}_grid_heatmap.png",
                f"Architecture_2 full grid: {title}",
                fmt,
            )

    criteria = [
        ("highest_ari", "ari", False),
        ("highest_nmi", "nmi", False),
        ("highest_purity", "overall_purity", False),
        ("lowest_validation_loss", "best_val_loss", True),
    ]

    best_rows = []
    for criterion, metric, minimize in criteria:
        valid = comparison.dropna(subset=[metric])
        if valid.empty:
            continue
        index = valid[metric].idxmin() if minimize else valid[metric].idxmax()
        selected = valid.loc[index]
        best_rows.append({
            "criterion": criterion,
            "metric": metric,
            "metric_value": selected[metric],
            "run_name": selected["run_name"],
            "num_embeddings": int(selected["num_embeddings"]),
            "batch_size": int(selected["batch_size"]),
            "ari": selected["ari"],
            "nmi": selected["nmi"],
            "overall_purity": selected["overall_purity"],
            "best_val_loss": selected["best_val_loss"],
            "test_total_loss": selected["test_total_loss"],
            "codebook_usage_percent": selected["codebook_usage_percent"],
        })

    pd.DataFrame(best_rows).to_csv(
        RUN_ROOT / "best_configurations_by_metric.csv",
        index=False,
    )

    print(
        f"\nSaved combined table: "
        f"{RUN_ROOT / 'full_factorial_comparison_table.csv'}"
    )


if __name__ == "__main__":
    main()
