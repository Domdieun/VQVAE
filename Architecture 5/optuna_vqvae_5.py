"""
optuna_vqvae_5.py

Hyperparameter optimization for Architecture_5 using validation loss.

Architecture_5 is a scVI-inspired VQ-VAE:
    - raw-count input
    - observed library size passed to decoder
    - decoder predicts px_scale
    - mu = library_size * px_scale
    - NB reconstruction loss
    - total loss = reconstruction loss + VQ loss
"""

import os
import json
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna

from Architecture_5 import VQVAE, reconstruction_loss
from load_data_split_5 import get_dataloaders, project_path, PROJECT_ROOT


# CONFIG

DATASET_NAME = "PBMC"   # change to "PBMC" or "Ath"

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
}

OUTPUT_DIRS = {
    "PBMC": project_path("PBMC/vqvae_runs/optuna_5"),
    "Ath": project_path("Ath/vqvae_runs/optuna_5"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]
OUTPUT_DIR = OUTPUT_DIRS[DATASET_NAME]

STUDY_NAME = f"{DATASET_NAME}_vqvae_5_scvi_library_size_nb_aligned"

N_TRIALS = 100
MAX_EPOCHS = 30
PATIENCE = 8
RANDOM_SEED = 42

FIXED_NUM_EMBEDDINGS = {"PBMC": 11, "Ath": 64}[DATASET_NAME]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(outputs, x, gene_likelihood, vq_weight):
    recon = reconstruction_loss(
        x=x,
        mu=outputs["mu"],
        theta=outputs["theta"],
        zi_logits=outputs["zi_logits"],
        gene_likelihood=gene_likelihood,
    )

    total = recon + vq_weight * outputs["vq_loss"]

    return total, recon


def train_one_epoch(model, train_loader, optimizer, gene_likelihood, vq_weight):
    model.train()

    total_loss = 0.0
    recon_loss_sum = 0.0
    vq_loss_sum = 0.0
    n_batches = 0

    for x_batch, library_batch in train_loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        library_batch = library_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(x_batch, library_batch)

        loss, recon_loss = compute_loss(
            outputs=outputs,
            x=x_batch,
            gene_likelihood=gene_likelihood,
            vq_weight=vq_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=200.0)
        optimizer.step()

        total_loss += loss.item()
        recon_loss_sum += recon_loss.item()
        vq_loss_sum += outputs["vq_loss"].item()
        n_batches += 1

    return {
        "total": total_loss / n_batches,
        "recon": recon_loss_sum / n_batches,
        "vq": vq_loss_sum / n_batches,
    }


def evaluate_loss(model, val_loader, gene_likelihood, vq_weight):
    model.eval()

    total_loss = 0.0
    recon_loss_sum = 0.0
    vq_loss_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for x_batch, library_batch in val_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            library_batch = library_batch.to(DEVICE, non_blocking=True)

            outputs = model(x_batch, library_batch)

            loss, recon_loss = compute_loss(
                outputs=outputs,
                x=x_batch,
                gene_likelihood=gene_likelihood,
                vq_weight=vq_weight,
            )

            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            vq_loss_sum += outputs["vq_loss"].item()
            n_batches += 1

    return {
        "total": total_loss / n_batches,
        "recon": recon_loss_sum / n_batches,
        "vq": vq_loss_sum / n_batches,
    }


def objective(trial):
    set_seed(RANDOM_SEED)

    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [128, 256, 512, 768, 1024],
    )

    latent_dim = trial.suggest_categorical(
        "latent_dim",
        [8, 16, 32, 64, 96, 128, 256],
    )

    if FIXED_NUM_EMBEDDINGS is None:
        num_embeddings = trial.suggest_categorical(
            "num_embeddings",
            [8, 10, 12, 15, 20],
        )
    else:
        num_embeddings = FIXED_NUM_EMBEDDINGS

    commitment_cost = trial.suggest_float(
        "commitment_cost",
        0.01,
        2.0,
        log=True,
    )

    dropout_rate = 0.0
    gene_likelihood = "nb"

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

    vq_weight = 1.0

    train_loader, val_loader, _test_loader, input_dimension, _adata = get_dataloaders(
        h5ad_path=H5AD_PATH,
        batch_size=batch_size,
        seed=RANDOM_SEED,
        dataset_name=DATASET_NAME,
    )

    model = VQVAE(
        input_dim=input_dimension,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        dropout_rate=dropout_rate,
        gene_likelihood=gene_likelihood,
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
            gene_likelihood=gene_likelihood,
            vq_weight=vq_weight,
        )

        val_metrics = evaluate_loss(
            model=model,
            val_loader=val_loader,
            gene_likelihood=gene_likelihood,
            vq_weight=vq_weight,
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
            f"Train recon: {train_metrics['recon']:.6f} | "
            f"Train VQ: {train_metrics['vq']:.6f} | "
            f"Val total: {val_metrics['total']:.6f} | "
            f"Val recon: {val_metrics['recon']:.6f} | "
            f"Val VQ: {val_metrics['vq']:.6f} | "
            f"Likelihood: {gene_likelihood} | "
            f"Codebook: {num_embeddings}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping trial {trial.number} at epoch {epoch}")
            break

    trial.set_user_attr("best_train_loss", best_train_loss)
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("best_val_recon_loss", best_val_recon_loss)
    trial.set_user_attr("best_val_vq_loss", best_val_vq_loss)
    trial.set_user_attr("epochs_trained", epoch)
    trial.set_user_attr("train_loss_history", train_loss_history)
    trial.set_user_attr("val_loss_history", val_loss_history)
    trial.set_user_attr("num_embeddings", num_embeddings)

    print(
        f"Trial {trial.number} finished | "
        f"Best train loss: {best_train_loss:.6f} | "
        f"Best val loss: {best_val_loss:.6f}"
    )

    return best_val_loss


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Using AnnData: {H5AD_PATH}")
    print(f"Using device: {DEVICE}")
    print(f"Fixed num_embeddings: {FIXED_NUM_EMBEDDINGS}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find AnnData file: {H5AD_PATH}")

    storage_path = os.path.join(OUTPUT_DIR, "optuna_study_5_nb_aligned.db")
    storage_name = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=== Best trial ===")
    print(f"Best value: {study.best_value}")

    best_params = dict(study.best_params)

    if FIXED_NUM_EMBEDDINGS is not None:
        best_params["num_embeddings"] = FIXED_NUM_EMBEDDINGS
    best_params["dropout_rate"] = 0.0
    best_params["gene_likelihood"] = "nb"
    best_params["vq_weight"] = 1.0

    print("Best params:")
    print(best_params)

    best_trial = study.best_trial

    best_result = {
        "dataset": DATASET_NAME,
        "architecture": "Architecture_5_scvi_library_size_vqvae",
        "objective": "validation_reconstruction_plus_vq_loss",
        "h5ad_path": H5AD_PATH,
        "best_value": study.best_value,
        "best_params": best_params,
        "best_trial_number": best_trial.number,
        "best_trial_attrs": best_trial.user_attrs,
    }

    best_json = os.path.join(OUTPUT_DIR, "best_params_5.json")
    with open(best_json, "w") as f:
        json.dump(best_result, f, indent=4)

    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(OUTPUT_DIR, "optuna_trials_5.csv")
    trials_df.to_csv(trials_csv, index=False)

    print("\nSaved:")
    print(best_json)
    print(trials_csv)
    print(storage_path)


if __name__ == "__main__":
    main()
