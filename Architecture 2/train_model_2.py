"""
train_model_2.py

Train the final VQ-VAE model using Architecture_2 and load_data_split_2.

This script is used after Optuna hyperparameter optimization. The best
hyperparameters selected by Optuna are manually inserted into the CONFIG section
and used to train a final VQ-VAE model.

Workflow:
1. Load the raw-count AnnData object for the selected dataset.
2. Create train, validation, and test DataLoaders using load_data_split_2.
3. Build a VQ-VAE model using Architecture_2.
4. Train the model on the training set.
5. Monitor validation loss after each epoch.
6. Save the best model checkpoint based on validation loss.
7. Stop training early if validation loss no longer improves.
8. Evaluate the best saved model on the held-out test set.
9. Save training history and loss curves.

Important:
- Architecture_2 returns:
      mu, theta, vq_loss, encoding_indices

- The reconstruction loss is Negative Binomial loss:
      total loss = NB reconstruction loss + VQ loss

- For Negative Binomial loss, use raw counts, not normalized/log1p data.

Run:
    python train_model_2.py
"""

import os
import json

import torch
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_data_split_2 import get_dataloaders, set_seed, project_path, PROJECT_ROOT
from Architecture_2 import VQVAE



# CONFIG

DATASET_NAME = "PBMC"   # change to "PBMC" when needed

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]

RANDOM_SEED = 42

# Replace these with the best parameters from your Architecture_2 Optuna run.
HYPERPARAMETERS = {
    "PBMC": {
        "hidden_dim": 768,
        "latent_dim": 128,
        "num_embeddings": 15,
        "commitment_cost": 1.012229629179095,
        "learning_rate": 0.0003042077888794354,
        "weight_decay": 3.3950954367462734e-09,
        "batch_size": 128,
    },
    "Ath": {
        "hidden_dim": 256,
        "latent_dim": 96,
        "num_embeddings": 10,
        "commitment_cost": 0.7049961189783545,
        "learning_rate": 0.0044749326652717926,
        "weight_decay": 7.153267862744501e-08,
        "batch_size": 512,
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

OUTPUT_DIR = project_path(f"{DATASET_NAME}/vqvae_runs/train_val_test_2")

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "best_vqvae_2_checkpoint.pth",
)

CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "training_curves_2.png",
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "loss_history_2.json",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NEGATIVE BINOMIAL LOSS


def negative_binomial_loss(x, mu, theta, eps=1e-8):
    """
    Negative Binomial negative log-likelihood.

    Args:
        x:
            Observed raw counts.
            Shape: batch_size x input_dim

        mu:
            Predicted mean of the Negative Binomial distribution.
            Must be positive.
            Shape: batch_size x input_dim

        theta:
            Predicted dispersion parameter.
            Must be positive.
            Shape can be batch_size x input_dim or input_dim.

        eps:
            Small value for numerical stability.

    Returns:
        Mean Negative Binomial loss.
    """
    x = torch.clamp(x, min=0.0)
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)

    log_likelihood = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )

    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood.mean()


def compute_loss(x, mu, theta, vq_loss):
    """
    Compute total Architecture_2 loss.
    total loss = Negative Binomial reconstruction loss + VQ loss
    """
    recon_loss = negative_binomial_loss(
        x=x,
        mu=mu,
        theta=theta,
    )

    total_loss = recon_loss + vq_loss

    return total_loss, recon_loss


# TRAINING / EVALUATION

def train_one_epoch(model, train_loader, optimiser, device):
    """
    Train the model for one epoch.

    Returns:
        Dictionary with average total, NB reconstruction, and VQ losses.
    """
    model.train()

    total_sum = 0.0
    recon_sum = 0.0
    vq_sum = 0.0

    for batch in train_loader:
        x_batch = batch[0].to(device)

        mu, theta, vq_loss, _ = model(x_batch)

        total_loss, recon_loss = compute_loss(
            x=x_batch,
            mu=mu,
            theta=theta,
            vq_loss=vq_loss,
        )

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
    """
    Evaluate the model.

    Returns:
        Dictionary with average total, NB reconstruction, and VQ losses.
    """
    model.eval()

    total_sum = 0.0
    recon_sum = 0.0
    vq_sum = 0.0

    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch[0].to(device)

            mu, theta, vq_loss, _ = model(x_batch)

            total_loss, recon_loss = compute_loss(
                x=x_batch,
                mu=mu,
                theta=theta,
                vq_loss=vq_loss,
            )

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
    """
    Count how many codebook vectors are used on a given dataset.
    """
    model.eval()

    all_indices = []

    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch[0].to(device)

            indices = model.encode(x_batch)

            # In case Architecture_2.encode returns more than just indices.
            if isinstance(indices, tuple):
                indices = indices[-1]

            all_indices.append(indices.cpu().view(-1))

    all_indices = torch.cat(all_indices)

    unique_codes = all_indices.unique().numel()
    usage_percent = 100 * unique_codes / num_embeddings

    return unique_codes, usage_percent


# MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    print(f"Project root: {PROJECT_ROOT}")
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
        "architecture": "Architecture_2",
        "loss": "negative_binomial_plus_vq",
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
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimiser=optimiser,
            device=DEVICE,
        )

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=DEVICE,
        )

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
            f"Train NB: {train_metrics['recon']:.4f} | "
            f"Train VQ: {train_metrics['vq']:.4f} | "
            f"Val total: {val_metrics['total']:.4f} | "
            f"Val NB: {val_metrics['recon']:.4f} | "
            f"Val VQ: {val_metrics['vq']:.4f}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "dataset_name": DATASET_NAME,
                    "h5ad_path": H5AD_PATH,
                    "architecture": "Architecture_2",
                    "loss": "negative_binomial_plus_vq",
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

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
    )

    unique_codes, usage_percent = count_codebook_usage(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        num_embeddings=NUM_EMBEDDINGS,
    )

    checkpoint["final_losses"] = test_metrics
    checkpoint["codebook_usage"] = {
        "unique_codes": int(unique_codes),
        "num_embeddings": int(NUM_EMBEDDINGS),
        "usage_percent": float(usage_percent),
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    print("\n=== Final test results ===")
    print(f"Test total loss: {test_metrics['total']:.4f}")
    print(f"Test NB loss:    {test_metrics['recon']:.4f}")
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
    plt.title(f"Architecture 2 VQ-VAE Train vs Validation Loss ({DATASET_NAME})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()

    print(f"\nBest checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Loss history saved to:    {HISTORY_PATH}")
    print(f"Training curve saved to:  {CURVE_PATH}")


if __name__ == "__main__":
    main()