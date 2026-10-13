"""
train_model_4.py

Train the final orthogonal-codebook VQ-VAE after Optuna.

THIS VERSION USES MSE RECONSTRUCTION LOSS.
Use normalized/log1p h5ad files.

Run after:
    python optuna_vqvae_4.py

or manually edit HYPERPARAMETERS below.
"""

import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_data_split_4 import (
    get_dataloaders,
    set_seed,
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
RANDOM_SEED = 42

BEST_PARAMS_JSON = project_path(
    f"{DATASET_NAME}/vqvae_runs/optuna_orthogonal_mse/best_params_orthogonal_mse.json"
)

# Fallback values. These are only used if BEST_PARAMS_JSON does not exist.
# num_embeddings is fixed per dataset, as requested.
HYPERPARAMETERS = {
    "PBMC": {
        "hidden_dim": 768,
        "latent_dim": 8,
        "num_embeddings": 11,
        "commitment_cost": 0.9709335672732289,
        "orthogonal_reg_weight": 0.0001,
        "orthogonal_reg_max_codes": None,
        "orthogonal_reg_active_codes_only": False,
        "learning_rate": 0.00047650725066811407,
        "weight_decay": 0.00010625744807079807,
        "batch_size": 1024,
    },
    "Ath": {
        "hidden_dim": 768,
        "latent_dim": 96,
        "num_embeddings": 14,
        "commitment_cost": 1.8737511623388106,
        "orthogonal_reg_weight": 0.01,
        "orthogonal_reg_max_codes": None,
        "orthogonal_reg_active_codes_only": False,
        "learning_rate": 0.0006751317050492229,
        "weight_decay": 1.8806384323947415e-05,
        "batch_size": 128,
    },
}

NUM_EPOCHS = 30
PATIENCE = 8

OUTPUT_DIR = project_path(f"{DATASET_NAME}/vqvae_runs/train_val_test_orthogonal_mse")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_vqvae_orthogonal_mse_checkpoint.pth")
CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curves_orthogonal_mse.png")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "loss_history_orthogonal_mse.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_selected_params():
    if os.path.exists(BEST_PARAMS_JSON):
        print(f"Loading best orthogonal VQ-VAE parameters from: {BEST_PARAMS_JSON}")
        with open(BEST_PARAMS_JSON, "r") as f:
            data = json.load(f)

        if "best_params" in data:
            return data["best_params"]
        return data

    print(f"Best params JSON not found: {BEST_PARAMS_JSON}")
    print("Using fallback HYPERPARAMETERS in this script.")
    return HYPERPARAMETERS[DATASET_NAME]


def compute_loss(x_batch, x_recon, vq_loss):
    recon_loss = F.mse_loss(x_recon, x_batch)
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss


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
        total_loss, recon_loss = compute_loss(x_batch, x_recon, vq_loss)

        total_loss.backward()
        optimizer.step()

        totals["total"] += total_loss.item()
        totals["recon"] += recon_loss.item()
        totals["vq"] += vq_loss.item()
        totals["orthogonal"] += loss_dict["orthogonal_loss"].item()
        totals["weighted_orthogonal"] += loss_dict["weighted_orthogonal_loss"].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def evaluate(model, loader):
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
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE)
            x_recon, vq_loss, _, loss_dict = model(x_batch)
            total_loss, recon_loss = compute_loss(x_batch, x_recon, vq_loss)

            totals["total"] += total_loss.item()
            totals["recon"] += recon_loss.item()
            totals["vq"] += vq_loss.item()
            totals["orthogonal"] += loss_dict["orthogonal_loss"].item()
            totals["weighted_orthogonal"] += loss_dict["weighted_orthogonal_loss"].item()
            n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def save_curves(history):
    epochs = range(1, len(history["train_total"]) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_total"], label="Train total")
    plt.plot(epochs, history["val_total"], label="Val total")
    plt.plot(epochs, history["train_recon"], label="Train MSE recon")
    plt.plot(epochs, history["val_recon"], label="Val MSE recon")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{DATASET_NAME} Orthogonal VQ-VAE MSE Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    params = load_selected_params()

    hidden_dim = int(params["hidden_dim"])
    latent_dim = int(params["latent_dim"])
    num_embeddings = int(params["num_embeddings"])
    commitment_cost = float(params["commitment_cost"])
    orthogonal_reg_weight = float(params.get("orthogonal_reg_weight", 0.0))
    orthogonal_reg_max_codes = params.get("orthogonal_reg_max_codes", None)
    orthogonal_reg_active_codes_only = bool(params.get("orthogonal_reg_active_codes_only", False))
    learning_rate = float(params["learning_rate"])
    weight_decay = float(params["weight_decay"])
    batch_size = int(params["batch_size"])

    print(f"Dataset: {DATASET_NAME}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"H5AD path: {H5AD_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Reconstruction loss: MSE")
    print(f"Selected parameters: {params}")
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, input_dim, adata = get_dataloaders(
        h5ad_path=H5AD_PATH,
        batch_size=batch_size,
        seed=RANDOM_SEED,
    )

    model = VQVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        orthogonal_reg_weight=orthogonal_reg_weight,
        orthogonal_reg_max_codes=orthogonal_reg_max_codes,
        orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
    ).to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {
        "train_total": [],
        "train_recon": [],
        "train_vq": [],
        "train_weighted_orthogonal": [],
        "val_total": [],
        "val_recon": [],
        "val_vq": [],
        "val_weighted_orthogonal": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader)
        scheduler.step(val_metrics["total"])

        history["train_total"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_vq"].append(train_metrics["vq"])
        history["train_weighted_orthogonal"].append(train_metrics["weighted_orthogonal"])
        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_vq"].append(val_metrics["vq"])
        history["val_weighted_orthogonal"].append(val_metrics["weighted_orthogonal"])

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train total {train_metrics['total']:.6f} | "
            f"Val total {val_metrics['total']:.6f} | "
            f"Val MSE {val_metrics['recon']:.6f} | "
            f"Val VQ {val_metrics['vq']:.6f} | "
            f"Val weighted orth {val_metrics['weighted_orthogonal']:.6f}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "hyperparameters": {
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "latent_dim": latent_dim,
                    "num_embeddings": num_embeddings,
                    "commitment_cost": commitment_cost,
                    "orthogonal_reg_weight": orthogonal_reg_weight,
                    "orthogonal_reg_max_codes": orthogonal_reg_max_codes,
                    "orthogonal_reg_active_codes_only": orthogonal_reg_active_codes_only,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "reconstruction_loss": "mse",
                },
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "dataset_name": DATASET_NAME,
            }

            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Saved new best checkpoint: {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    print("Loading best checkpoint for test evaluation...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader)
    history["test_metrics"] = test_metrics

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    save_curves(history)

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=4))
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved history: {HISTORY_PATH}")
    print(f"Saved curves: {CURVE_PATH}")


if __name__ == "__main__":
    main()
