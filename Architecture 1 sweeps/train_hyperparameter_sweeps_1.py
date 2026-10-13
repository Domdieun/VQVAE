"""
train_hyperparameter_sweeps_1.py

Train a full 5 x 5 Architecture_1 VQ-VAE factorial experiment.

Every codebook size in [10, 11, 12, 13, 14] is paired with every batch size
in [32, 64, 128, 256, 512], producing 25 unique model runs.
"""

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

from Architecture_1 import VQVAE
from hyperparameter_sweep_config_1 import (
    BASE_HYPERPARAMETERS,
    DATASET_NAME,
    H5AD_PATH,
    LR_SCHEDULER_PATIENCE,
    NUM_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    RUN_ROOT,
    SKIP_COMPLETED_RUNS,
    build_experiments,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # The architecture uses deterministic linear/BatchNorm operations. These
    # settings further reduce run-to-run variation where CUDA permits it.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_split_datasets():
    """Load the expression matrix once and create one fixed data split."""
    print(f"Loading AnnData once for all training runs: {H5AD_PATH}")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"AnnData shape: {adata.shape}")

    expression = adata.X
    if hasattr(expression, "toarray"):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=np.float32)

    tensor = torch.from_numpy(expression)
    dataset = TensorDataset(tensor)

    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=split_generator,
    )

    print(f"Train cells: {len(train_set)}")
    print(f"Val cells:   {len(val_set)}")
    print(f"Test cells:  {len(test_set)}")
    print(f"Genes:       {adata.n_vars}")

    input_dim = adata.n_vars
    del adata, expression
    return train_set, val_set, test_set, input_dim


def make_loaders(split_datasets, batch_size):
    """Build loaders with a run-specific batch size from the fixed split."""
    train_set, val_set, test_set = split_datasets
    shuffle_generator = torch.Generator().manual_seed(RANDOM_SEED)
    common = {
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=shuffle_generator,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def compute_loss(x, x_recon, vq_loss):
    recon_loss = F.mse_loss(x_recon, x)
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss


def train_one_epoch(model, data_loader, optimiser):
    model.train()
    totals = {"total": 0.0, "recon": 0.0, "vq": 0.0}
    n_samples = 0

    for (x_batch,) in data_loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        batch_n = x_batch.shape[0]

        optimiser.zero_grad(set_to_none=True)
        x_recon, vq_loss, _ = model(x_batch)
        total_loss, recon_loss = compute_loss(x_batch, x_recon, vq_loss)
        total_loss.backward()
        optimiser.step()

        # Weight batch means by cell count so metrics remain comparable when
        # batch sizes differ or the final batch is smaller.
        totals["total"] += total_loss.item() * batch_n
        totals["recon"] += recon_loss.item() * batch_n
        totals["vq"] += vq_loss.item() * batch_n
        n_samples += batch_n

    if n_samples == 0:
        raise RuntimeError("Training DataLoader contains zero cells.")

    return {key: value / n_samples for key, value in totals.items()}


def evaluate(model, data_loader):
    model.eval()
    totals = {"total": 0.0, "recon": 0.0, "vq": 0.0}
    n_samples = 0

    with torch.no_grad():
        for (x_batch,) in data_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            batch_n = x_batch.shape[0]
            x_recon, vq_loss, _ = model(x_batch)
            total_loss, recon_loss = compute_loss(x_batch, x_recon, vq_loss)

            totals["total"] += total_loss.item() * batch_n
            totals["recon"] += recon_loss.item() * batch_n
            totals["vq"] += vq_loss.item() * batch_n
            n_samples += batch_n

    if n_samples == 0:
        raise RuntimeError("Evaluation DataLoader contains zero cells.")

    return {key: value / n_samples for key, value in totals.items()}


def count_codebook_usage(model, data_loader, num_embeddings):
    model.eval()
    code_counts = torch.zeros(num_embeddings, dtype=torch.long)

    with torch.no_grad():
        for (x_batch,) in data_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            indices = model.encode(x_batch).cpu()
            code_counts += torch.bincount(indices, minlength=num_embeddings)

    unique_codes = int((code_counts > 0).sum().item())
    usage_percent = 100.0 * unique_codes / num_embeddings

    return unique_codes, usage_percent, code_counts.tolist()


def save_curves(history, output_path: Path, title: str) -> None:
    epochs = range(1, len(history["train_total"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_total"], label="Train total")
    plt.plot(epochs, history["val_total"], label="Validation total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_run(experiment, base_params, split_datasets, input_dim):
    run_dir = Path(experiment["run_dir"])
    checkpoint_path = run_dir / "best_vqvae_checkpoint.pth"
    history_path = run_dir / "loss_history.json"
    curve_path = run_dir / "training_curves.png"
    result_path = run_dir / "run_result.json"

    if (
        SKIP_COMPLETED_RUNS
        and checkpoint_path.exists()
        and result_path.exists()
    ):
        print(f"\nSkipping completed run: {experiment['run_name']}")
        with result_path.open("r") as f:
            result = json.load(f)

        # Normalize metadata from runs created by the earlier nine-run design.
        # The learned checkpoint is unchanged; only its experiment label is now
        # part of the full 5 x 5 factorial grid.
        result["sweep_membership"] = experiment["sweep_membership"]
        return result

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_SEED)

    num_embeddings = int(experiment["num_embeddings"])
    batch_size = int(experiment["batch_size"])

    print("\n" + "=" * 80)
    print(f"Run: {experiment['run_name']}")
    print(f"Sweep membership: {', '.join(experiment['sweep_membership'])}")
    print(f"num_embeddings: {num_embeddings}")
    print(f"batch_size: {batch_size}")
    print("=" * 80)

    # The underlying split is fixed; only DataLoader batch size changes.
    train_loader, val_loader, test_loader = make_loaders(
        split_datasets,
        batch_size=batch_size,
    )

    hparams = {
        **base_params,
        "input_dim": input_dim,
        "num_embeddings": num_embeddings,
        "batch_size": batch_size,
    }

    model = VQVAE(
        input_dim=input_dim,
        hidden_dim=hparams["hidden_dim"],
        latent_dim=hparams["latent_dim"],
        num_embeddings=num_embeddings,
        commitment_cost=hparams["commitment_cost"],
    ).to(DEVICE)

    optimiser = Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=LR_SCHEDULER_PATIENCE,
    )

    history = {
        "dataset_name": DATASET_NAME,
        "run_name": experiment["run_name"],
        "sweep_membership": experiment["sweep_membership"],
        "h5ad_path": str(H5AD_PATH),
        "hyperparameters": {
            **hparams,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
        },
        "train_total": [],
        "train_recon": [],
        "train_vq": [],
        "val_total": [],
        "val_recon": [],
        "val_vq": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimiser)
        val_metrics = evaluate(model, val_loader)
        scheduler.step(val_metrics["total"])

        history["train_total"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_vq"].append(train_metrics["vq"])
        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_vq"].append(val_metrics["vq"])
        history["learning_rate"].append(optimiser.param_groups[0]["lr"])

        print(
            f"{experiment['run_name']} | Epoch [{epoch:03d}/{NUM_EPOCHS}] | "
            f"Train total: {train_metrics['total']:.6f} | "
            f"Val total: {val_metrics['total']:.6f} | "
            f"Val recon: {val_metrics['recon']:.6f} | "
            f"Val VQ: {val_metrics['vq']:.6f}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "dataset_name": DATASET_NAME,
                    "run_name": experiment["run_name"],
                    "sweep_membership": experiment["sweep_membership"],
                    "h5ad_path": str(H5AD_PATH),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "hyperparameters": hparams,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_metrics,
                    "val_losses": val_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping {experiment['run_name']} after epoch {epoch}.")
            break

    with history_path.open("w") as f:
        json.dump(history, f, indent=4)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader)
    unique_codes, usage_percent, code_counts = count_codebook_usage(
        model,
        test_loader,
        num_embeddings,
    )

    checkpoint["final_losses"] = test_metrics
    checkpoint["codebook_usage"] = {
        "unique_codes": unique_codes,
        "num_embeddings": num_embeddings,
        "usage_percent": usage_percent,
        "counts": code_counts,
    }
    torch.save(checkpoint, checkpoint_path)

    save_curves(
        history,
        curve_path,
        title=(
            f"{DATASET_NAME} Architecture_1: "
            f"K={num_embeddings}, batch={batch_size}"
        ),
    )

    result = {
        "dataset": DATASET_NAME,
        "run_name": experiment["run_name"],
        "sweep_membership": experiment["sweep_membership"],
        "num_embeddings": num_embeddings,
        "batch_size": batch_size,
        "hidden_dim": hparams["hidden_dim"],
        "latent_dim": hparams["latent_dim"],
        "commitment_cost": hparams["commitment_cost"],
        "learning_rate": hparams["learning_rate"],
        "weight_decay": hparams["weight_decay"],
        "best_epoch": best_epoch,
        "epochs_trained": len(history["train_total"]),
        "best_val_loss": best_val_loss,
        "test_total_loss": test_metrics["total"],
        "test_recon_loss": test_metrics["recon"],
        "test_vq_loss": test_metrics["vq"],
        "unique_codes_used": unique_codes,
        "codebook_usage_percent": usage_percent,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
    }

    with result_path.open("w") as f:
        json.dump(result, f, indent=4)

    print(
        f"Completed {experiment['run_name']} | "
        f"test={test_metrics['total']:.6f} | "
        f"codes={unique_codes}/{num_embeddings}"
    )

    del model, optimiser, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    if not Path(H5AD_PATH).exists():
        raise FileNotFoundError(f"Could not find AnnData file: {H5AD_PATH}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments()
    base_params = BASE_HYPERPARAMETERS[DATASET_NAME]

    print(f"Dataset: {DATASET_NAME}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Unique model runs: {len(experiments)}")
    print("Full factorial entries: 5 codebook sizes x 5 batch sizes = 25")

    manifest_rows = []
    for experiment in experiments:
        manifest_rows.append({
            "run_name": experiment["run_name"],
            "num_embeddings": experiment["num_embeddings"],
            "batch_size": experiment["batch_size"],
            "sweep_membership": ",".join(experiment["sweep_membership"]),
            "run_dir": str(experiment["run_dir"]),
        })

    pd.DataFrame(manifest_rows).to_csv(
        RUN_ROOT / "experiment_manifest.csv",
        index=False,
    )

    train_set, val_set, test_set, input_dim = load_split_datasets()
    split_datasets = (train_set, val_set, test_set)

    results = [
        train_run(experiment, base_params, split_datasets, input_dim)
        for experiment in experiments
    ]
    results_df = pd.DataFrame(results).sort_values(
        ["num_embeddings", "batch_size"]
    )
    results_df.to_csv(RUN_ROOT / "training_summary.csv", index=False)

    print("\nAll training runs completed.")
    print(f"Summary: {RUN_ROOT / 'training_summary.csv'}")


if __name__ == "__main__":
    main()
