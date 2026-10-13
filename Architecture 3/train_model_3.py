
"""
train_model_3.py

Train the final Architecture_3 model after normal Optuna tuning.
"""

import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Architecture_3 import VQVAE
from ema_pbmc_config_3 import (
    H5AD_PATH,
    OPTUNA_OUTPUT_DIR,
    TRAIN_OUTPUT_DIR,
    CHECKPOINT_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from load_data_split_3 import (
    get_dataloaders,
    set_seed,
)

NUM_EPOCHS = 30
PATIENCE = 8
EMA_EPSILON = 1e-5

BEST_PARAMS_PATH = os.path.join(
    OPTUNA_OUTPUT_DIR,
    "best_params_ema.json",
)

# Used only when final training is run before Optuna.
FALLBACK_HYPERPARAMETERS = {
    "hidden_dim": 512,
    "latent_dim": 8,
    "num_embeddings": 20,
    "commitment_cost": 0.14893649866850991,
    "ema_decay": 0.9,
    "learning_rate":  0.00015261240433965642,
    "weight_decay": 1.2277151423401724e-05,
    "batch_size": 128,
}

CURVE_PATH = os.path.join(
    TRAIN_OUTPUT_DIR,
    "training_curves_ema.png",
)
HISTORY_PATH = os.path.join(
    TRAIN_OUTPUT_DIR,
    "loss_history_ema.json",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_selected_params():
    if os.path.exists(BEST_PARAMS_PATH):
        print(
            f"Loading Optuna parameters: {BEST_PARAMS_PATH}"
        )

        with open(BEST_PARAMS_PATH) as handle:
            data = json.load(handle)

        return data.get("best_params", data)

    print(
        "No Optuna result found. Using fallback Architecture 3 "
        "parameters with ema_decay=0.9."
    )
    return dict(FALLBACK_HYPERPARAMETERS)


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


def collect_codebook_usage(
    model,
    loader,
):
    model.eval()
    batches = []

    with torch.no_grad():
        for (x_batch,) in loader:
            batches.append(
                model.encode(
                    x_batch.to(DEVICE)
                ).cpu()
            )

    indices = torch.cat(batches)

    statistics = model.vq.codebook_usage_from_indices(
        indices
    )

    return {
        "unique_codes": statistics["unique_codes"],
        "num_embeddings": model.vq.num_embeddings,
        "usage_percent": statistics["usage_percent"],
        "perplexity": statistics["perplexity"],
        "counts": statistics["counts"].tolist(),
    }


def main():
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    params = load_selected_params()

    required = [
        "hidden_dim",
        "latent_dim",
        "num_embeddings",
        "commitment_cost",
        "ema_decay",
        "learning_rate",
        "weight_decay",
        "batch_size",
    ]
    missing = [
        key for key in required
        if key not in params
    ]

    if missing:
        raise KeyError(
            f"Selected hyperparameters are missing: {missing}"
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"H5AD: {H5AD_PATH}")
    print(f"Device: {DEVICE}")
    print("Selected hyperparameters:")
    print(json.dumps(params, indent=2))

    (
        train_loader,
        val_loader,
        test_loader,
        input_dim,
        _adata,
    ) = get_dataloaders(
        h5ad_path=H5AD_PATH,
        batch_size=int(params["batch_size"]),
        seed=RANDOM_SEED,
    )

    model = VQVAE(
        input_dim=input_dim,
        hidden_dim=int(params["hidden_dim"]),
        latent_dim=int(params["latent_dim"]),
        num_embeddings=int(params["num_embeddings"]),
        commitment_cost=float(
            params["commitment_cost"]
        ),
        ema_decay=float(params["ema_decay"]),
        ema_epsilon=EMA_EPSILON,
    ).to(DEVICE)

    optimizer = Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    history = {
        "dataset_name": "PBMC",
        "h5ad_path": H5AD_PATH,
        "architecture": "Architecture_3",
        "loss": "mse_plus_commitment",
        "codebook_update": "exponential_moving_average",
        "hyperparameters": {
            **params,
            "ema_epsilon": EMA_EPSILON,
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
    best_epoch = None
    without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
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

        for key in ("total", "recon", "vq"):
            history[f"train_{key}"].append(
                train_metrics[key]
            )
            history[f"val_{key}"].append(
                val_metrics[key]
            )

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}] | "
            f"Train total {train_metrics['total']:.6f} | "
            f"Train MSE {train_metrics['recon']:.6f} | "
            f"Train commitment {train_metrics['vq']:.6f} | "
            f"Val total {val_metrics['total']:.6f} | "
            f"Val MSE {val_metrics['recon']:.6f} | "
            f"Val commitment {val_metrics['vq']:.6f} | "
            f"EMA updates {int(model.vq.ema_update_steps.item())}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            without_improvement = 0

            torch.save(
                {
                    "dataset_name": "PBMC",
                    "h5ad_path": H5AD_PATH,
                    "architecture": "Architecture_3",
                    "loss": "mse_plus_commitment",
                    "codebook_update": (
                        "exponential_moving_average"
                    ),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state": optimizer.state_dict(),
                    "hyperparameters": {
                        "input_dim": input_dim,
                        **params,
                        "ema_epsilon": EMA_EPSILON,
                    },
                    "best_val_loss": best_val_loss,
                    "train_losses": train_metrics,
                    "val_losses": val_metrics,
                    "ema_state": {
                        "update_steps": int(
                            model.vq.ema_update_steps.item()
                        ),
                        "cluster_size": (
                            model.vq.ema_cluster_size
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                    },
                },
                CHECKPOINT_PATH,
            )
            print(
                f"  Saved best checkpoint at epoch {epoch}."
            )
        else:
            without_improvement += 1

        if without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = run_epoch(
        model,
        test_loader,
    )
    codebook_usage = collect_codebook_usage(
        model,
        test_loader,
    )

    checkpoint["final_losses"] = test_metrics
    checkpoint["codebook_usage"] = codebook_usage
    torch.save(checkpoint, CHECKPOINT_PATH)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["test_metrics"] = test_metrics
    history["codebook_usage"] = codebook_usage

    with open(HISTORY_PATH, "w") as handle:
        json.dump(history, handle, indent=2)

    epochs = range(
        1,
        len(history["train_total"]) + 1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_total"],
        label="Train total",
    )
    plt.plot(
        epochs,
        history["val_total"],
        label="Validation total",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE + commitment loss")
    plt.title("PBMC Architecture 3 with EMA codebook")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()

    print("\n=== Final test results ===")
    print(json.dumps(test_metrics, indent=2))
    print("\n=== Codebook usage ===")
    print(json.dumps(codebook_usage, indent=2))
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
