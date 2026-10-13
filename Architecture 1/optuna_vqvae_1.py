"""
optuna_vqvae.py

Hyperparameter optimization for VQ-VAE using Optuna.

Purpose:
This script trains multiple VQ-VAE models with different hyperparameters and
selects the best setting based on validation loss.

Important:
- The model is trained using reconstruction loss + VQ loss.
- Optuna minimizes validation total loss.
- The test set is not used during hyperparameter tuning.
- The best hyperparameters are saved to a JSON file.

Run:
    python optuna_vqvae_1.py
"""

import os
from load_data_split_1 import guess_project_root as get_project_root
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna

from Architecture_1 import VQVAE
from load_data_split_1 import get_dataloaders


# -----------------------------
# CONFIG
# -----------------------------

# Scripts are expected to live in VQVAE/Pipeline1_MSE/2HL.
# This points back to the main VQVAE directory so files are read from/written to VQVAE/Ath and VQVAE/PBMC.
PROJECT_ROOT = Path(os.environ.get("VQVAE_PROJECT_ROOT", get_project_root())).resolve()

DATASET_NAME = os.environ.get("VQVAE_DATASET", "PBMC")

H5AD_PATHS = {
    "PBMC": str(PROJECT_ROOT / "PBMC" / "PBMC_68k_filtered_normalized.h5ad"),
    "Ath": str(PROJECT_ROOT / "Ath" / "combined_cellbender_filter_matrix_main_normalized.h5ad"),
}

OUTPUT_DIRS = {
    "PBMC": str(PROJECT_ROOT / "PBMC" / "vqvae_runs" / "optuna"),
    "Ath": str(PROJECT_ROOT / "Ath" / "vqvae_runs" / "optuna"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
OUTPUT_DIR = OUTPUT_DIRS[DATASET_NAME]

STUDY_NAME = f"{DATASET_NAME}_vqvae_validation_loss"

N_TRIALS = 100
MAX_EPOCHS = 30
PATIENCE = 8

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# REPRODUCIBILITY
# -----------------------------

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# TRAINING / EVALUATION
# -----------------------------

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

        x_recon, vq_loss, _ = model(x_batch)

        recon_loss = F.mse_loss(x_recon, x_batch)
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

            x_recon, vq_loss, _ = model(x_batch)

            recon_loss = F.mse_loss(x_recon, x_batch)
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
    3. trains a VQ-VAE
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
            f"Val total: {val_metrics['total']:.6f} | "
            f"Val recon: {val_metrics['recon']:.6f} | "
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


# -----------------------------
# MAIN
# -----------------------------

def main():
    set_seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}")
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
        "objective": "validation_loss",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": best_trial.number,
        "best_trial_attrs": best_trial.user_attrs,
    }

    best_json = os.path.join(
        OUTPUT_DIR,
        "best_params.json",
    )

    with open(best_json, "w") as f:
        json.dump(best_result, f, indent=4)

    trials_df = study.trials_dataframe()

    trials_csv = os.path.join(
        OUTPUT_DIR,
        "optuna_trials.csv",
    )

    trials_df.to_csv(trials_csv, index=False)

    print("\nSaved:")
    print(best_json)
    print(trials_csv)
    print(storage_path)


if __name__ == "__main__":
    main()