
"""
train_hyperparameter_sweeps_2.py

Train all 25 Architecture_2 Negative-Binomial VQ-VAE combinations.

Grid:
    codebooks: 10, 11, 12, 13, 14
    batches:   32, 64, 128, 256, 512

All other PBMC hyperparameters remain fixed.
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

from Architecture_2 import VQVAE, negative_binomial_loss
from hyperparameter_sweep_config_2 import (
    BASE_HYPERPARAMETERS,
    DATASET_NAME,
    H5AD_PATH,
    LR_SCHEDULER_PATIENCE,
    NUM_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    RUN_ROOT,
    SKIP_COMPLETED_RUNS,
    TRAIN_FRACTION,
    VAL_FRACTION,
    build_experiments,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def matrix_to_dense_float32(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def get_count_matrix(adata):
    if "counts" in adata.layers:
        print("Using adata.layers['counts'].")
        return adata.layers["counts"]
    if adata.raw is not None:
        print("Using adata.raw.X.")
        return adata.raw.X

    print("Using adata.X as raw counts.")
    return adata.X


def check_count_matrix(matrix):
    values = matrix.data if hasattr(matrix, "data") else np.asarray(matrix).ravel()
    values = np.asarray(values)

    if values.size == 0:
        raise ValueError("Count matrix contains no values.")
    if not np.isfinite(values).all():
        raise ValueError("Count matrix contains NaN or infinite values.")
    if values.min() < 0:
        raise ValueError("Count matrix contains negative values.")
    if not np.allclose(values, np.round(values), atol=1e-6):
        raise ValueError(
            "Count matrix contains fractional values. "
            "Architecture_2 NB training requires raw counts."
        )


def load_fixed_split():
    print(f"Loading AnnData once: {H5AD_PATH}")
    adata = sc.read_h5ad(H5AD_PATH)
    print(f"AnnData shape: {adata.shape}")

    matrix = get_count_matrix(adata)
    check_count_matrix(matrix)
    dense = matrix_to_dense_float32(matrix)

    dataset = TensorDataset(torch.from_numpy(dense))
    n_total = len(dataset)
    n_train = int(TRAIN_FRACTION * n_total)
    n_val = int(VAL_FRACTION * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    split = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    print(f"Train cells: {n_train}")
    print(f"Val cells:   {n_val}")
    print(f"Test cells:  {n_test}")
    print(f"Genes:       {dense.shape[1]}")

    return split, dense.shape[1]


def make_loaders(split, batch_size):
    train_set, val_set, test_set = split

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    common = {
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
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


def compute_loss(x, mu, theta, vq_loss):
    recon = negative_binomial_loss(x=x, mu=mu, theta=theta)
    return recon + vq_loss, recon


def run_epoch(model, loader, optimiser=None):
    training = optimiser is not None
    model.train(training)

    totals = {"total": 0.0, "recon": 0.0, "vq": 0.0}
    n_cells = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            batch_n = x_batch.shape[0]

            if training:
                optimiser.zero_grad(set_to_none=True)

            mu, theta, vq_loss, _ = model(x_batch)
            total_loss, recon_loss = compute_loss(
                x_batch, mu, theta, vq_loss
            )

            if training:
                total_loss.backward()
                optimiser.step()

            totals["total"] += total_loss.item() * batch_n
            totals["recon"] += recon_loss.item() * batch_n
            totals["vq"] += vq_loss.item() * batch_n
            n_cells += batch_n

    if n_cells == 0:
        raise RuntimeError("DataLoader produced zero cells.")

    return {key: value / n_cells for key, value in totals.items()}


def count_codebook_usage(model, loader, num_embeddings):
    model.eval()
    indices = []

    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            batch_indices = model.encode(x_batch)
            if isinstance(batch_indices, tuple):
                batch_indices = batch_indices[-1]
            indices.append(batch_indices.detach().cpu().view(-1))

    all_indices = torch.cat(indices)
    unique_codes = int(all_indices.unique().numel())
    return unique_codes, 100.0 * unique_codes / num_embeddings


def save_curve(history, output_path, run_name):
    epochs = range(1, len(history["train_total"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_total"], label="Train total")
    plt.plot(epochs, history["val_total"], label="Validation total")
    plt.xlabel("Epoch")
    plt.ylabel("NB + VQ loss")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_one_configuration(experiment, split, input_dim):
    run_name = experiment["run_name"]
    run_dir = Path(experiment["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "best_vqvae_checkpoint.pth"
    result_path = run_dir / "run_result.json"

    if (
        SKIP_COMPLETED_RUNS
        and checkpoint_path.exists()
        and result_path.exists()
    ):
        print(f"Skipping completed run: {run_name}")
        with result_path.open() as handle:
            return json.load(handle)

    set_seed(RANDOM_SEED)

    fixed = BASE_HYPERPARAMETERS[DATASET_NAME]
    hparams = {
        **fixed,
        "num_embeddings": experiment["num_embeddings"],
        "batch_size": experiment["batch_size"],
    }

    print("\n" + "=" * 88)
    print(f"Training {run_name}")
    print(json.dumps(hparams, indent=2))
    print("=" * 88)

    train_loader, val_loader, test_loader = make_loaders(
        split,
        hparams["batch_size"],
    )

    model = VQVAE(
        input_dim=input_dim,
        hidden_dim=hparams["hidden_dim"],
        latent_dim=hparams["latent_dim"],
        num_embeddings=hparams["num_embeddings"],
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
        "train_total": [],
        "train_recon": [],
        "train_vq": [],
        "val_total": [],
        "val_recon": [],
        "val_vq": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimiser)
        val_metrics = run_epoch(model, val_loader)
        scheduler.step(val_metrics["total"])

        history["train_total"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_vq"].append(train_metrics["vq"])
        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_vq"].append(val_metrics["vq"])
        history["learning_rate"].append(optimiser.param_groups[0]["lr"])

        print(
            f"{run_name} | Epoch [{epoch:03d}/{NUM_EPOCHS}] | "
            f"Train total {train_metrics['total']:.6f} | "
            f"Train NB {train_metrics['recon']:.6f} | "
            f"Train VQ {train_metrics['vq']:.6f} | "
            f"Val total {val_metrics['total']:.6f} | "
            f"Val NB {val_metrics['recon']:.6f} | "
            f"Val VQ {val_metrics['vq']:.6f}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "dataset_name": DATASET_NAME,
                    "architecture": "Architecture_2",
                    "loss": "negative_binomial_plus_vq",
                    "h5ad_path": str(H5AD_PATH),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "hyperparameters": {
                        "input_dim": input_dim,
                        **hparams,
                    },
                    "best_val_loss": best_val_loss,
                    "train_losses": train_metrics,
                    "val_losses": val_metrics,
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint at epoch {epoch}.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping {run_name} at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader)
    unique_codes, usage_percent = count_codebook_usage(
        model,
        test_loader,
        hparams["num_embeddings"],
    )

    checkpoint["final_losses"] = test_metrics
    checkpoint["codebook_usage"] = {
        "unique_codes": unique_codes,
        "num_embeddings": hparams["num_embeddings"],
        "usage_percent": usage_percent,
    }
    torch.save(checkpoint, checkpoint_path)

    history_payload = {
        "dataset_name": DATASET_NAME,
        "architecture": "Architecture_2",
        "h5ad_path": str(H5AD_PATH),
        "hyperparameters": {
            "input_dim": input_dim,
            **hparams,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
        },
        **history,
    }

    with (run_dir / "loss_history.json").open("w") as handle:
        json.dump(history_payload, handle, indent=2)

    save_curve(history, run_dir / "training_curves.png", run_name)

    result = {
        "run_name": run_name,
        "num_embeddings": hparams["num_embeddings"],
        "batch_size": hparams["batch_size"],
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
        "checkpoint_path": str(checkpoint_path),
    }

    with result_path.open("w") as handle:
        json.dump(result, handle, indent=2)

    return result


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_SEED)

    experiments = build_experiments()
    if len(experiments) != 25:
        raise RuntimeError(f"Expected 25 experiments, got {len(experiments)}.")

    print(f"Project output root: {RUN_ROOT}")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Unique model runs: {len(experiments)}")
    print("Grid: 5 codebook sizes x 5 batch sizes")

    manifest = pd.DataFrame([
        {
            "run_name": exp["run_name"],
            "num_embeddings": exp["num_embeddings"],
            "batch_size": exp["batch_size"],
            "run_dir": str(exp["run_dir"]),
        }
        for exp in experiments
    ])
    manifest.to_csv(RUN_ROOT / "experiment_manifest.csv", index=False)

    with (RUN_ROOT / "experiment_config.json").open("w") as handle:
        json.dump(
            {
                "dataset": DATASET_NAME,
                "h5ad_path": str(H5AD_PATH),
                "fixed_hyperparameters": BASE_HYPERPARAMETERS[DATASET_NAME],
                "num_experiments": len(experiments),
            },
            handle,
            indent=2,
        )

    split, input_dim = load_fixed_split()

    rows = []
    for index, experiment in enumerate(experiments, start=1):
        print(f"\nRun {index}/{len(experiments)}")
        rows.append(train_one_configuration(experiment, split, input_dim))

    pd.DataFrame(rows).to_csv(
        RUN_ROOT / "training_summary.csv",
        index=False,
    )

    print("\nAll training runs completed.")
    print(f"Summary: {RUN_ROOT / 'training_summary.csv'}")


if __name__ == "__main__":
    main()
