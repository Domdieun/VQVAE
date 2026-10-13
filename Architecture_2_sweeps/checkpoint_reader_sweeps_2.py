
"""
checkpoint_reader_sweeps_2.py

Create one compact CSV summary for all Architecture_2 sweep checkpoints.
"""

from pathlib import Path

import pandas as pd
import torch

from hyperparameter_sweep_config_2 import RUN_ROOT, build_experiments


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
        assignments_path = Path(experiment["run_dir"]) / "barcode_cluster_assignments.csv"
        unique_all = None
        usage_all = None
        if assignments_path.exists():
            assigned = pd.read_csv(assignments_path, usecols=["cluster_index"])
            unique_all = int(assigned["cluster_index"].nunique())
            if hparams.get("num_embeddings"):
                usage_all = 100.0 * unique_all / hparams["num_embeddings"]

        rows.append({
            "run_name": experiment["run_name"],
            "epoch": checkpoint.get("epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "input_dim": hparams.get("input_dim"),
            "hidden_dim": hparams.get("hidden_dim"),
            "latent_dim": hparams.get("latent_dim"),
            "num_embeddings": hparams.get("num_embeddings"),
            "commitment_cost": hparams.get("commitment_cost"),
            "learning_rate": hparams.get("learning_rate"),
            "weight_decay": hparams.get("weight_decay"),
            "batch_size": hparams.get("batch_size"),
            "test_total_loss": final_losses.get("total"),
            "test_nb_loss": final_losses.get("recon"),
            "test_vq_loss": final_losses.get("vq"),
            "unique_codes_test": usage.get("unique_codes"),
            "usage_percent_test": usage.get("usage_percent"),
            "unique_codes_all_cells": unique_all,
            "usage_percent_all_cells": usage_all,
            "checkpoint_path": str(checkpoint_path),
        })

    output = RUN_ROOT / "checkpoint_summary.csv"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
