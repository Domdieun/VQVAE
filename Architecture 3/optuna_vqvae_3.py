
"""
optuna_vqvae_3.py

Normal validation-loss Optuna tuning for Architecture_3.

Optuna minimizes:

    validation MSE reconstruction loss + EMA commitment loss

The test set is not used for model selection.

Tuned parameters:
    hidden_dim
    latent_dim
    num_embeddings
    commitment_cost
    ema_decay
    learning_rate
    weight_decay
    batch_size

ema_epsilon remains fixed at 1e-5.
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

from Architecture_3 import VQVAE
from ema_pbmc_config_3 import (
    H5AD_PATH,
    OPTUNA_OUTPUT_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from load_data_split_3 import (
    load_adata_and_dataset,
    split_dataset,
    make_dataloaders,
)

N_TRIALS = 100
MAX_EPOCHS = 30
PATIENCE = 8

#conventional choice for small enough to not affect normal values, but large enough to prevent numerical problems
EMA_EPSILON = 1e-5

STUDY_NAME = "PBMC_Architecture_3_validation_loss"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    sums = {
        "total": 0.0,
        "recon": 0.0,
        "vq": 0.0,
    }
    n_cells = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for (x_batch,) in loader:
            x_batch = x_batch.to(
                DEVICE,
                non_blocking=True,
            )
            batch_n = x_batch.shape[0]

            if training:
                optimizer.zero_grad(set_to_none=True)

            x_recon, vq_loss, _indices = model(x_batch)
            recon_loss = F.mse_loss(x_recon, x_batch)
            total_loss = recon_loss + vq_loss

            if training:
                total_loss.backward()
                optimizer.step()

            sums["total"] += total_loss.item() * batch_n
            sums["recon"] += recon_loss.item() * batch_n
            sums["vq"] += vq_loss.item() * batch_n
            n_cells += batch_n

    return {
        key: value / n_cells
        for key, value in sums.items()
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
    ema_decay = trial.suggest_categorical(
        "ema_decay",
        [0.90, 0.95, 0.99, 0.995, 0.999],
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

    train_loader, val_loader, _test_loader = (
        make_dataloaders(
            split=DATA_SPLIT,
            batch_size=batch_size,
            seed=RANDOM_SEED,
        )
    )

    model = VQVAE(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        ema_decay=ema_decay,
        ema_epsilon=EMA_EPSILON,
    ).to(DEVICE)

    # The EMA embedding has requires_grad=False and is excluded explicitly.
    optimizer = Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    #schedule the learning rate on validation total loss, with factor 0.5 and patience 5.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    best_train_loss = None
    best_val_recon = None
    best_val_vq = None
    without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
        )

        scheduler.step(val_metrics["total"])

        trial.report(val_metrics["total"], epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_train_loss = train_metrics["total"]
            best_val_recon = val_metrics["recon"]
            best_val_vq = val_metrics["vq"]
            without_improvement = 0
        else:
            without_improvement += 1

        print(
            f"Trial {trial.number} | "
            f"Epoch {epoch:03d} | "
            f"Train total {train_metrics['total']:.6f} | "
            f"Val total {val_metrics['total']:.6f} | "
            f"Val MSE {val_metrics['recon']:.6f} | "
            f"Val commitment {val_metrics['vq']:.6f} | "
            f"EMA decay {ema_decay}"
        )

        if without_improvement >= PATIENCE:
            print(
                f"Early stopping trial {trial.number} "
                f"at epoch {epoch}."
            )
            break

    trial.set_user_attr(
        "best_train_loss",
        best_train_loss,
    )
    trial.set_user_attr(
        "best_val_loss",
        best_val_loss,
    )
    trial.set_user_attr(
        "best_val_recon_loss",
        best_val_recon,
    )
    trial.set_user_attr(
        "best_val_commitment_loss",
        best_val_vq,
    )
    trial.set_user_attr(
        "epochs_trained",
        epoch,
    )
    trial.set_user_attr(
        "ema_update_steps",
        int(model.vq.ema_update_steps.item()),
    )

    return best_val_loss


def main():
    global DATA_SPLIT
    global INPUT_DIM

    set_seed(RANDOM_SEED)
    os.makedirs(OPTUNA_OUTPUT_DIR, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"H5AD: {H5AD_PATH}")
    print(f"Device: {DEVICE}")

    _adata, dataset = load_adata_and_dataset(H5AD_PATH)
    DATA_SPLIT = split_dataset(dataset, RANDOM_SEED)
    INPUT_DIM = dataset.n_vars

    storage_path = os.path.join(
        OPTUNA_OUTPUT_DIR,
        "optuna_study_ema.db",
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    result = {
        "dataset": "PBMC",
        "h5ad_path": H5AD_PATH,
        "architecture": "Architecture_3",
        "objective": "validation_mse_plus_commitment_loss",
        "ema_epsilon": EMA_EPSILON,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_trial_attrs": study.best_trial.user_attrs,
    }

    best_json = os.path.join(
        OPTUNA_OUTPUT_DIR,
        "best_params_ema.json",
    )
    trials_csv = os.path.join(
        OPTUNA_OUTPUT_DIR,
        "optuna_trials_ema.csv",
    )

    with open(best_json, "w") as handle:
        json.dump(result, handle, indent=2)

    study.trials_dataframe().to_csv(
        trials_csv,
        index=False,
    )

    print(json.dumps(result, indent=2))
    print(f"Study database: {storage_path}")


if __name__ == "__main__":
    main()
