"""
optuna_vqvae_4.py

Hyperparameter optimization for Architecture_4.py using Optuna.

THIS VERSION USES MSE RECONSTRUCTION LOSS.
The codebook size is fixed per dataset and is not tuned by Optuna.

Total training objective:
    total loss = MSE reconstruction loss + VQ auxiliary loss

where VQ auxiliary loss contains:
    codebook loss + commitment loss + weighted orthogonal regularisation loss

Run:
    python optuna_vqvae_4.py
"""

import json
import os
import random

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_data_split_4 import (
    get_dataloaders,
    project_path,
    PROJECT_ROOT,
)
from Architecture_4 import VQVAE


# CONFIG

DATASET_NAME = "PBMC"  # change to "PBMC" or "Ath"

# Clear single input path per dataset for this MSE model.
H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_filtered_normalized.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main_normalized.h5ad"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]

OUTPUT_DIRS = {
    "PBMC": project_path("PBMC/vqvae_runs/optuna_orthogonal_mse"),
    "Ath": project_path("Ath/vqvae_runs/optuna_orthogonal_mse"),
}

OUTPUT_DIR = OUTPUT_DIRS[DATASET_NAME]

# Fixed codebook size per dataset. Optuna does not tune this.
FIXED_NUM_EMBEDDINGS_BY_DATASET = {
    "PBMC": 11,
    "Ath": 14,
}
NUM_EMBEDDINGS = FIXED_NUM_EMBEDDINGS_BY_DATASET[DATASET_NAME]

STUDY_NAME = f"{DATASET_NAME}_orthogonal_vqvae_mse_fixed_codebook"
N_TRIALS = 100
MAX_EPOCHS = 30
PATIENCE = 8
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, optimizer):
    model.train()

    totals = {
        "total": 0.0,
        "recon": 0.0,
        "vq": 0.0,
        "orthogonal": 0.0,
        "weighted_orthogonal": 0.0,
    }
    n_batches = 0

    for (x_batch,) in train_loader:
        x_batch = x_batch.to(DEVICE)

        optimizer.zero_grad()
        x_recon, vq_loss, _, loss_dict = model(x_batch)

        recon_loss = F.mse_loss(x_recon, x_batch)
        loss = recon_loss + vq_loss

        loss.backward()
        optimizer.step()

        totals["total"] += loss.item()
        totals["recon"] += recon_loss.item()
        totals["vq"] += vq_loss.item()
        totals["orthogonal"] += loss_dict["orthogonal_loss"].item()
        totals["weighted_orthogonal"] += loss_dict["weighted_orthogonal_loss"].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def evaluate_loss(model, val_loader):
    model.eval()

    totals = {
        "total": 0.0,
        "recon": 0.0,
        "vq": 0.0,
        "orthogonal": 0.0,
        "weighted_orthogonal": 0.0,
    }
    n_batches = 0

    with torch.no_grad():
        for (x_batch,) in val_loader:
            x_batch = x_batch.to(DEVICE)
            x_recon, vq_loss, _, loss_dict = model(x_batch)

            recon_loss = F.mse_loss(x_recon, x_batch)
            loss = recon_loss + vq_loss

            totals["total"] += loss.item()
            totals["recon"] += recon_loss.item()
            totals["vq"] += vq_loss.item()
            totals["orthogonal"] += loss_dict["orthogonal_loss"].item()
            totals["weighted_orthogonal"] += loss_dict["weighted_orthogonal_loss"].item()
            n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def objective(trial):
    set_seed(RANDOM_SEED)

    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 768, 1024])
    latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64, 96, 128, 256])
    commitment_cost = trial.suggest_float("commitment_cost", 0.01, 2.0, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    # Orthogonal loss is tuned, but codebook size is fixed.
    orthogonal_reg_weight = trial.suggest_categorical(
        "orthogonal_reg_weight",
        [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1],
    )

    train_loader, val_loader, test_loader, input_dim, adata = get_dataloaders(
        h5ad_path=H5AD_PATH,
        batch_size=batch_size,
        seed=RANDOM_SEED,
    )

    model = VQVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=NUM_EMBEDDINGS,
        commitment_cost=commitment_cost,
        orthogonal_reg_weight=orthogonal_reg_weight,
        orthogonal_reg_max_codes=None,
        orthogonal_reg_active_codes_only=False,
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
    best_val_weighted_orthogonal_loss = None
    epochs_without_improvement = 0

    train_history = []
    val_history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer)
        val_metrics = evaluate_loss(model, val_loader)
        scheduler.step(val_metrics["total"])

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        trial.report(val_metrics["total"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_train_loss = train_metrics["total"]
            best_val_recon_loss = val_metrics["recon"]
            best_val_vq_loss = val_metrics["vq"]
            best_val_weighted_orthogonal_loss = val_metrics["weighted_orthogonal"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Trial {trial.number} | Epoch {epoch:03d} | "
            f"Train total {train_metrics['total']:.6f} | "
            f"Val total {val_metrics['total']:.6f} | "
            f"Val recon {val_metrics['recon']:.6f} | "
            f"Val VQ {val_metrics['vq']:.6f} | "
            f"Val weighted orth {val_metrics['weighted_orthogonal']:.6f}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping trial {trial.number} at epoch {epoch}.")
            break

    trial.set_user_attr("best_train_loss", best_train_loss)
    trial.set_user_attr("best_val_recon_loss", best_val_recon_loss)
    trial.set_user_attr("best_val_vq_loss", best_val_vq_loss)
    trial.set_user_attr("best_val_weighted_orthogonal_loss", best_val_weighted_orthogonal_loss)
    trial.set_user_attr("num_embeddings_fixed", NUM_EMBEDDINGS)
    trial.set_user_attr("train_history", train_history)
    trial.set_user_attr("val_history", val_history)

    return best_val_loss


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"H5AD path: {H5AD_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Reconstruction loss: MSE")
    print(f"Fixed num_embeddings: {NUM_EMBEDDINGS}")
    print(f"Device: {DEVICE}")

    set_seed(RANDOM_SEED)

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(objective, n_trials=N_TRIALS)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["num_embeddings"] = NUM_EMBEDDINGS
    best_params["orthogonal_reg_max_codes"] = None
    best_params["orthogonal_reg_active_codes_only"] = False

    results = {
        "dataset_name": DATASET_NAME,
        "reconstruction_loss": "mse",
        "study_name": STUDY_NAME,
        "best_trial_number": best_trial.number,
        "best_value_val_total_loss": best_trial.value,
        "best_params": best_params,
        "best_user_attrs": best_trial.user_attrs,
    }

    best_json = os.path.join(OUTPUT_DIR, "best_params_orthogonal_mse.json")
    with open(best_json, "w") as f:
        json.dump(results, f, indent=4)

    trials_csv = os.path.join(OUTPUT_DIR, "optuna_trials_orthogonal_mse.csv")
    study.trials_dataframe().to_csv(trials_csv, index=False)

    print("\nBest trial:")
    print(f"  Trial number: {best_trial.number}")
    print(f"  Best val total loss: {best_trial.value:.6f}")
    print(f"  Best params: {best_params}")
    print(f"Saved best params to: {best_json}")
    print(f"Saved trials table to: {trials_csv}")


if __name__ == "__main__":
    main()
