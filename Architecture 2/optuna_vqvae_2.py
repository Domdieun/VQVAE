"""
optuna_vqvae_2.py

Hyperparameter optimization for VQ-VAE Architecture_2 using Optuna.

Purpose:
This script trains multiple VQ-VAE models with different hyperparameters and
selects the best setting based on validation loss.

Important:
- This version is for Architecture_2.
- Architecture_2 returns:
      mu, theta, vq_loss, encoding_indices
- The reconstruction loss is Negative Binomial loss.
- Optuna minimizes validation total loss:
      total loss = NB reconstruction loss + VQ loss
- The test set is not used during hyperparameter tuning.
- The best hyperparameters are saved to a JSON file.

Run:
    python optuna_vqvae_2.py
"""

import os
import json
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna

from Architecture_2 import VQVAE
from load_data_split_2 import get_dataloaders, project_path, PROJECT_ROOT


# CONFIG

DATASET_NAME = "PBMC"   # change to "PBMC" when needed or "Ath"

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
}

OUTPUT_DIRS = {
    "PBMC": project_path("PBMC/vqvae_runs/optuna_2"),
    "Ath": project_path("Ath/vqvae_runs/optuna_2"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
OUTPUT_DIR = OUTPUT_DIRS[DATASET_NAME]

STUDY_NAME = f"{DATASET_NAME}_vqvae_2_validation_loss"

N_TRIALS = 100
MAX_EPOCHS = 30
PATIENCE = 8

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# REPRODUCIBILITY

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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
            Shape can be:
                batch_size x input_dim
            or:
                input_dim

        eps:
            Small value for numerical stability.

    Returns:
        Mean Negative Binomial loss.
    """
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


# TRAINING / EVALUATION

def train_one_epoch(model, train_loader, optimizer):
    """
    Train the model for one epoch.

    Returns:
        Dictionary with average total, reconstruction, and VQ losses.
    """
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    n_batches = 0

    for (x_batch,) in train_loader:
        x_batch = x_batch.to(DEVICE)

        optimizer.zero_grad()

        mu, theta, vq_loss, _ = model(x_batch)

        recon_loss = negative_binomial_loss(
            x=x_batch,
            mu=mu,
            theta=theta,
        )

        loss = recon_loss + vq_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        n_batches += 1

    return {
        "total": total_loss / n_batches,
        "recon": total_recon_loss / n_batches,
        "vq": total_vq_loss / n_batches,
    }


def evaluate_loss(model, val_loader):
    """
    Evaluate the model on validation data.

    Returns:
        Dictionary with average total, reconstruction, and VQ losses.
    """
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for (x_batch,) in val_loader:
            x_batch = x_batch.to(DEVICE)

            mu, theta, vq_loss, _ = model(x_batch)

            recon_loss = negative_binomial_loss(
                x=x_batch,
                mu=mu,
                theta=theta,
            )

            loss = recon_loss + vq_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            n_batches += 1

    return {
        "total": total_loss / n_batches,
        "recon": total_recon_loss / n_batches,
        "vq": total_vq_loss / n_batches,
    }


# -----------------------------
# OPTUNA OBJECTIVE
# -----------------------------

def objective(trial):
    """
    Optuna objective function.

    Each trial:
    1. samples hyperparameters
    2. creates dataloaders using the sampled batch size
    3. trains an Architecture_2 VQ-VAE
    4. tracks validation loss
    5. returns the best validation loss
    """
    set_seed(RANDOM_SEED)

    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [128, 256, 512, 768, 1024],
    )

    latent_dim = trial.suggest_categorical(
        "latent_dim",
        [8, 16, 32, 64, 96, 128, 256],
    )

    num_embeddings = trial.suggest_categorical(
        "num_embeddings",
        [8, 10, 12, 15, 20],
    )

    commitment_cost = trial.suggest_float(
        "commitment_cost",
        0.01,
        2.0,
        log=True,
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        5e-3,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-9,
        1e-3,
        log=True,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [64, 128, 256, 512, 1024],
    )

    train_loader, val_loader, test_loader, input_dimension, adata = get_dataloaders(
        h5ad_path=H5AD_PATH,
        batch_size=batch_size,
        seed=RANDOM_SEED,
    )

    model = VQVAE(
        input_dim=input_dimension,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
    ).to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_train_loss = None
    best_val_recon_loss = None
    best_val_vq_loss = None

    train_loss_history = []
    val_loss_history = []

    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
        )

        val_metrics = evaluate_loss(
            model=model,
            val_loader=val_loader,
        )
        scheduler.step(val_metrics["total"])

        train_loss_history.append(train_metrics)
        val_loss_history.append(val_metrics)

        trial.report(val_metrics["total"], step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_train_loss = train_metrics["total"]
            best_val_recon_loss = val_metrics["recon"]
            best_val_vq_loss = val_metrics["vq"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Trial {trial.number} | "
            f"Epoch {epoch:03d} | "
            f"Train total: {train_metrics['total']:.6f} | "
            f"Train NB: {train_metrics['recon']:.6f} | "
            f"Train VQ: {train_metrics['vq']:.6f} | "
            f"Val total: {val_metrics['total']:.6f} | "
            f"Val NB: {val_metrics['recon']:.6f} | "
            f"Val VQ: {val_metrics['vq']:.6f}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping trial {trial.number} "
                f"at epoch {epoch}"
            )
            break

    trial.set_user_attr("best_train_loss", best_train_loss)
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("best_val_recon_loss", best_val_recon_loss)
    trial.set_user_attr("best_val_vq_loss", best_val_vq_loss)
    trial.set_user_attr("epochs_trained", epoch)
    trial.set_user_attr("train_loss_history", train_loss_history)
    trial.set_user_attr("val_loss_history", val_loss_history)

    print(
        f"Trial {trial.number} finished | "
        f"Best train loss: {best_train_loss:.6f} | "
        f"Best val loss: {best_val_loss:.6f}"
    )

    return best_val_loss



# MAIN

def main():
    set_seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Using AnnData: {H5AD_PATH}")
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find AnnData file: {H5AD_PATH}")

    storage_path = os.path.join(
        OUTPUT_DIR,
        "optuna_study.db",
    )

    storage_name = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    print("\n=== Best trial ===")
    print(f"Best value: {study.best_value}")
    print("Best params:")
    print(study.best_params)

    best_trial = study.best_trial

    best_result = {
        "dataset": DATASET_NAME,
        "architecture": "Architecture_2",
        "objective": "validation_negative_binomial_plus_vq_loss",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": best_trial.number,
        "best_trial_attrs": best_trial.user_attrs,
    }

    best_json = os.path.join(
        OUTPUT_DIR,
        "best_params_2.json",
    )

    with open(best_json, "w") as f:
        json.dump(best_result, f, indent=4)

    trials_df = study.trials_dataframe()

    trials_csv = os.path.join(
        OUTPUT_DIR,
        "optuna_trials_2.csv",
    )

    trials_df.to_csv(trials_csv, index=False)

    print("\nSaved:")
    print(best_json)
    print(trials_csv)
    print(storage_path)


if __name__ == "__main__":
    main()