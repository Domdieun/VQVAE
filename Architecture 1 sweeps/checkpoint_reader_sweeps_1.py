"""Print a compact checkpoint summary for every Architecture_1 sweep run."""

from pathlib import Path

import pandas as pd
import torch

from hyperparameter_sweep_config_1 import RUN_ROOT, build_experiments


def main():
    rows = []

    for experiment in build_experiments():
        checkpoint_path = Path(experiment["run_dir"]) / "best_vqvae_checkpoint.pth"
        if not checkpoint_path.exists():
            print(f"Missing: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        hparams = checkpoint.get("hyperparameters", {})
        final_losses = checkpoint.get("final_losses", {})
        usage = checkpoint.get("codebook_usage", {})
        assignment_path = Path(experiment["run_dir"]) / "barcode_cluster_assignments.csv"
        all_cells_unique = None
        all_cells_usage_percent = None
        if assignment_path.exists():
            assignments = pd.read_csv(assignment_path, usecols=["cluster_index"])
            all_cells_unique = int(assignments["cluster_index"].nunique())
            codebook_size = hparams.get("num_embeddings")
            if codebook_size:
                all_cells_usage_percent = 100.0 * all_cells_unique / codebook_size

        rows.append({
            "run_name": experiment["run_name"],
            "sweep_membership": ",".join(experiment["sweep_membership"]),
            "epoch": checkpoint.get("epoch"),
            "num_embeddings": hparams.get("num_embeddings"),
            "batch_size": hparams.get("batch_size"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "test_total_loss": final_losses.get("total"),
            "test_recon_loss": final_losses.get("recon"),
            "test_vq_loss": final_losses.get("vq"),
            "unique_codes_test": usage.get("unique_codes"),
            "usage_percent_test": usage.get("usage_percent"),
            "unique_codes_all_cells": all_cells_unique,
            "usage_percent_all_cells": all_cells_usage_percent,
        })

    if not rows:
        raise RuntimeError("No completed checkpoints were found.")

    summary_df = pd.DataFrame(rows).sort_values(["num_embeddings", "batch_size"])
    print(summary_df.to_string(index=False))
    summary_df.to_csv(RUN_ROOT / "checkpoint_summary.csv", index=False)


if __name__ == "__main__":
    main()
