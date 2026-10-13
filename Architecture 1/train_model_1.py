"""
train_model_1.py

Train the final VQ-VAE model using a train/validation/test split.

This script is used after Optuna hyperparameter optimization. The best
hyperparameters selected by Optuna are manually inserted into the CONFIG section
and used to train a final VQ-VAE model.

Workflow:
1. Load the normalized AnnData object for the selected dataset.
2. Create train, validation, and test DataLoaders.
3. Build a VQ-VAE model using the selected hyperparameters.
4. Train the model on the training set.
5. Monitor validation loss after each epoch.
6. Save the best model checkpoint based on validation loss.
7. Stop training early if validation loss no longer improves.
8. Evaluate the best saved model on the held-out test set.
9. Save training history and loss curves.
"""

import os
from load_data_split_1 import guess_project_root as get_project_root
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_data_split_1 import get_dataloaders, set_seed
from Architecture_1 import VQVAE


DATASET_NAME = os.environ.get("VQVAE_DATASET", "PBMC")

# Scripts are expected to live in VQVAE/Pipeline1_MSE/2HL.
# This points back to the main VQVAE directory so outputs go to VQVAE/Ath and VQVAE/PBMC.
PROJECT_ROOT = Path(os.environ.get("VQVAE_PROJECT_ROOT", get_project_root())).resolve()

H5AD_PATHS = {
    "PBMC": str(PROJECT_ROOT / "PBMC" / "PBMC_68k_filtered_normalized.h5ad"),
    "Ath": str(PROJECT_ROOT / "Ath" / "combined_cellbender_filter_matrix_main_normalized.h5ad"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
RANDOM_SEED = 42

HYPERPARAMETERS = {
    "PBMC": {
        "hidden_dim": 128,
        "latent_dim": 8,
        "num_embeddings": 12,
        "commitment_cost": 1.9396462944591644,
        "learning_rate": 0.00011995141620829452,
        "weight_decay": 1.6229179584481823e-07,
        "batch_size": 64,
    },
    "Ath": {
        "hidden_dim": 128,
        "latent_dim": 128,
        "num_embeddings": 20,
        "commitment_cost": 0.7754711260109922,
        "learning_rate": 0.0029985564396042726,
        "weight_decay": 8.543860846911426e-08,
        "batch_size": 128,
    },
}

SELECTED_PARAMS = HYPERPARAMETERS[DATASET_NAME]

HIDDEN_DIM = SELECTED_PARAMS["hidden_dim"]
LATENT_DIM = SELECTED_PARAMS["latent_dim"]
NUM_EMBEDDINGS = SELECTED_PARAMS["num_embeddings"]
COMMITMENT_COST = SELECTED_PARAMS["commitment_cost"]
LEARNING_RATE = SELECTED_PARAMS["learning_rate"]
WEIGHT_DECAY = SELECTED_PARAMS["weight_decay"]
BATCH_SIZE = SELECTED_PARAMS["batch_size"]

NUM_EPOCHS = 30
PATIENCE = 8

OUTPUT_DIR = str(PROJECT_ROOT / DATASET_NAME / "vqvae_runs" / "train_val_test")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_vqvae_checkpoint.pth")
CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "loss_history.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(x, x_recon, vq_loss):
    recon_loss = F.mse_loss(x_recon, x)
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss


def train_one_epoch(model, train_loader, optimiser, device):
    model.train()

    total_sum = 0.0
    recon_sum = 0.0
    vq_sum = 0.0

    for batch in train_loader:
        x_batch = batch[0].to(device)

        x_recon, vq_loss, _ = model(x_batch)
        total_loss, recon_loss = compute_loss(x_batch, x_recon, vq_loss)

        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()

        total_sum += total_loss.item()
        recon_sum += recon_loss.item()
        vq_sum += vq_loss.item()

    n_batches = len(train_loader)

    return {
        "total": total_sum / n_batches,
        "recon": recon_sum / n_batches,
        "vq": vq_sum / n_batches,
    }


def evaluate(model, data_loader, device):
    model.eval()

    total_sum = 0.0
    recon_sum = 0.0
    vq_sum = 0.0

    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch[0].to(device)

            x_recon, vq_loss, _ = model(x_batch)
            total_loss, recon_loss = compute_loss(x_batch, x_recon, vq_loss)

            total_sum += total_loss.item()
            recon_sum += recon_loss.item()
            vq_sum += vq_loss.item()

    n_batches = len(data_loader)

    return {
        "total": total_sum / n_batches,
        "recon": recon_sum / n_batches,
        "vq": vq_sum / n_batches,
    }


def count_codebook_usage(model, data_loader, device, num_embeddings):
    model.eval()

    all_indices = []

    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch[0].to(device)
            indices = model.encode(x_batch)
            all_indices.append(indices.cpu())

    all_indices = torch.cat(all_indices)
    unique_codes = all_indices.unique().numel()
    usage_percent = 100 * unique_codes / num_embeddings

    return unique_codes, usage_percent


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    print(f"Dataset: {DATASET_NAME}")
    print(f"H5AD path: {H5AD_PATH}")
    print(f"Training on: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, input_dimension, _ = get_dataloaders(
        h5ad_path=H5AD_PATH,
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED,
    )

    model = VQVAE(
        input_dim=input_dimension,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_embeddings=NUM_EMBEDDINGS,
        commitment_cost=COMMITMENT_COST,
    ).to(DEVICE)

    optimiser = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=5,
    )

    history = {
        "dataset_name": DATASET_NAME,
        "h5ad_path": H5AD_PATH,
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "num_embeddings": NUM_EMBEDDINGS,
            "commitment_cost": COMMITMENT_COST,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
        },
        "train_total": [],
        "train_recon": [],
        "train_vq": [],
        "val_total": [],
        "val_recon": [],
        "val_vq": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimiser, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        scheduler.step(val_metrics["total"])

        history["train_total"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_vq"].append(train_metrics["vq"])

        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_vq"].append(val_metrics["vq"])

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}] | "
            f"Train total: {train_metrics['total']:.4f} | "
            f"Val total: {val_metrics['total']:.4f} | "
            f"Val recon: {val_metrics['recon']:.4f} | "
            f"Val VQ: {val_metrics['vq']:.4f}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "dataset_name": DATASET_NAME,
                    "h5ad_path": H5AD_PATH,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "hyperparameters": {
                        "input_dim": input_dimension,
                        "hidden_dim": HIDDEN_DIM,
                        "latent_dim": LATENT_DIM,
                        "num_embeddings": NUM_EMBEDDINGS,
                        "commitment_cost": COMMITMENT_COST,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "batch_size": BATCH_SIZE,
                    },
                    "best_val_loss": best_val_loss,
                    "train_losses": train_metrics,
                    "val_losses": val_metrics,
                },
                CHECKPOINT_PATH,
            )

            print(f"  Saved new best model with val loss {best_val_loss:.4f}")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after {epoch} epochs.")
            break

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, DEVICE)

    unique_codes, usage_percent = count_codebook_usage(
        model,
        test_loader,
        DEVICE,
        NUM_EMBEDDINGS,
    )

    checkpoint["final_losses"] = test_metrics
    checkpoint["codebook_usage"] = {
        "unique_codes": unique_codes,
        "num_embeddings": NUM_EMBEDDINGS,
        "usage_percent": usage_percent,
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    print("\n=== Final test results ===")
    print(f"Test total loss: {test_metrics['total']:.4f}")
    print(f"Test recon loss: {test_metrics['recon']:.4f}")
    print(f"Test VQ loss:    {test_metrics['vq']:.4f}")
    print(
        f"Codebook usage on test set: "
        f"{unique_codes}/{NUM_EMBEDDINGS} ({usage_percent:.1f}%)"
    )

    epochs = range(1, len(history["train_total"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_total"], label="Train total")
    plt.plot(epochs, history["val_total"], label="Validation total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VQ-VAE Train vs Validation Loss ({DATASET_NAME})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()

    print(f"\nBest checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Loss history saved to:    {HISTORY_PATH}")
    print(f"Training curve saved to:  {CURVE_PATH}")


if __name__ == "__main__":
    main()