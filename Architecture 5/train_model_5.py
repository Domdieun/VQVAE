"""
train_model_5.py

Train Architecture_5: a scVI-inspired VQ-VAE that explicitly accounts for
library-size technical variation.

Main difference from Architecture_2:
    Architecture_2: decoder directly predicts mu
    Architecture_5: decoder predicts px_scale, then mu = library_size * px_scale

Run after optuna_vqvae_5.py, or manually edit HYPERPARAMETERS below.
"""

import os
import json

import matplotlib.pyplot as plt
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Architecture_5 import VQVAE, reconstruction_loss
from load_data_split_5 import get_dataloaders, set_seed, project_path, PROJECT_ROOT


# CONFIG

DATASET_NAME = "PBMC"   # change to "PBMC" or "Ath"

H5AD_PATHS = {
    "PBMC": project_path("PBMC/PBMC_68k_UNFILTERED.h5ad"),
    "Ath": project_path("Ath/combined_cellbender_filter_matrix_main.h5ad"),
}

H5AD_PATH = H5AD_PATHS[DATASET_NAME]

RANDOM_SEED = 42

BEST_PARAMS_JSON = project_path(f"{DATASET_NAME}/vqvae_runs/optuna_5/best_params_5.json")

# Defaults intended as reasonable starting points.
# Optuna results, if present, override these.
HYPERPARAMETERS = {
    "PBMC": {
        "hidden_dim": 128,
        "latent_dim": 64,
        "num_embeddings": 11,
        "commitment_cost":  0.16741141593655337,
        "dropout_rate": 0.0,
        "gene_likelihood": "nb",
        "learning_rate":  0.00019970958243557984,
        "weight_decay": 3.0345110356441345e-09,
        "batch_size": 128,
        "vq_weight": 1.0,
    },
    "Ath": {
        "hidden_dim": 256,
        "latent_dim": 32,
        "num_embeddings": 64,
        "commitment_cost": 0.25,
        "dropout_rate": 0.1,
        "gene_likelihood": "zinb",
        "learning_rate": 4e-4,
        "weight_decay": 1e-6,
        "batch_size": 256,
        "vq_weight": 1.0,
    },
}

NUM_EPOCHS = 30
PATIENCE = 8

OUTPUT_DIR = project_path(f"{DATASET_NAME}/vqvae_runs/train_val_test_5")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_vqvae_5_checkpoint.pth")
CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curves_5.png")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "loss_history_5.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_selected_params():
    if os.path.exists(BEST_PARAMS_JSON):
        print(f"Loading best Architecture_5 parameters from: {BEST_PARAMS_JSON}")
        with open(BEST_PARAMS_JSON, "r") as f:
            data = json.load(f)

        if "best_params" in data:
            return data["best_params"]

        return data

    print("No Architecture_5 best_params_5.json found.")
    print("Using fallback HYPERPARAMETERS in train_model_5.py.")

    return HYPERPARAMETERS[DATASET_NAME]


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


def train_one_epoch(model, train_loader, optimiser, device, gene_likelihood, vq_weight):
    model.train()

    total_sum = 0.0
    recon_sum = 0.0
    vq_sum = 0.0

    for x_batch, library_batch in train_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        library_batch = library_batch.to(device, non_blocking=True)

        outputs = model(x_batch, library_batch)

        total_loss, recon_loss = compute_loss(
            outputs=outputs,
            x=x_batch,
            gene_likelihood=gene_likelihood,
            vq_weight=vq_weight,
        )

        optimiser.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=200.0)
        optimiser.step()

        total_sum += total_loss.item()
        recon_sum += recon_loss.item()
        vq_sum += outputs["vq_loss"].item()

    n_batches = len(train_loader)

    return {
        "total": total_sum / n_batches,
        "recon": recon_sum / n_batches,
        "vq": vq_sum / n_batches,
    }


def evaluate(model, data_loader, device, gene_likelihood, vq_weight):
    model.eval()

    total_sum = 0.0
    recon_sum = 0.0
    vq_sum = 0.0

    with torch.no_grad():
        for x_batch, library_batch in data_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            library_batch = library_batch.to(device, non_blocking=True)

            outputs = model(x_batch, library_batch)

            total_loss, recon_loss = compute_loss(
                outputs=outputs,
                x=x_batch,
                gene_likelihood=gene_likelihood,
                vq_weight=vq_weight,
            )

            total_sum += total_loss.item()
            recon_sum += recon_loss.item()
            vq_sum += outputs["vq_loss"].item()

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
        for x_batch, _library_batch in data_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            indices = model.encode(x_batch)
            all_indices.append(indices.cpu().view(-1))

    all_indices = torch.cat(all_indices)
    unique_codes = all_indices.unique().numel()
    usage_percent = 100 * unique_codes / num_embeddings

    return unique_codes, usage_percent


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    selected_params = load_selected_params()

    hidden_dim = selected_params["hidden_dim"]
    latent_dim = selected_params["latent_dim"]
    num_embeddings = selected_params["num_embeddings"]
    commitment_cost = selected_params["commitment_cost"]
    dropout_rate = selected_params.get("dropout_rate", 0.1)
    gene_likelihood = selected_params.get("gene_likelihood", "nb")
    learning_rate = selected_params["learning_rate"]
    weight_decay = selected_params["weight_decay"]
    batch_size = selected_params["batch_size"]
    vq_weight = selected_params.get("vq_weight", 1.0)

    if DATASET_NAME == "PBMC" and (gene_likelihood != "nb" or num_embeddings != 11):
        raise ValueError("The PBMC Architecture 5 configuration requires NB likelihood and 11 codebook entries.")
    if DATASET_NAME == "PBMC" and (dropout_rate != 0.0 or vq_weight != 1.0):
        raise ValueError("The selected PBMC configuration must use dropout=0 and vq_weight=1.")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"H5AD path: {H5AD_PATH}")
    print(f"Training on: {DEVICE}")
    print("Selected hyperparameters:")
    print(selected_params)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, input_dimension, _ = get_dataloaders(
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

    optimiser = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
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
        "architecture": "Architecture_5_scvi_library_size_vqvae",
        "loss": f"{gene_likelihood}_plus_vq",
        "hyperparameters": {
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_embeddings": num_embeddings,
            "commitment_cost": commitment_cost,
            "dropout_rate": dropout_rate,
            "gene_likelihood": gene_likelihood,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "vq_weight": vq_weight,
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
            gene_likelihood=gene_likelihood,
            vq_weight=vq_weight,
        )

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=DEVICE,
            gene_likelihood=gene_likelihood,
            vq_weight=vq_weight,
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
            f"Train recon: {train_metrics['recon']:.4f} | "
            f"Train VQ: {train_metrics['vq']:.4f} | "
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
                    "architecture": "Architecture_5_scvi_library_size_vqvae",
                    "loss": f"{gene_likelihood}_plus_vq",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "hyperparameters": {
                        "input_dim": input_dimension,
                        "hidden_dim": hidden_dim,
                        "latent_dim": latent_dim,
                        "num_embeddings": num_embeddings,
                        "commitment_cost": commitment_cost,
                        "dropout_rate": dropout_rate,
                        "gene_likelihood": gene_likelihood,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "vq_weight": vq_weight,
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
        gene_likelihood=gene_likelihood,
        vq_weight=vq_weight,
    )

    unique_codes, usage_percent = count_codebook_usage(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        num_embeddings=num_embeddings,
    )

    checkpoint["final_losses"] = test_metrics
    checkpoint["codebook_usage"] = {
        "unique_codes": int(unique_codes),
        "num_embeddings": int(num_embeddings),
        "usage_percent": float(usage_percent),
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    print("\n=== Final test results ===")
    print(f"Test total loss: {test_metrics['total']:.4f}")
    print(f"Test recon loss: {test_metrics['recon']:.4f}")
    print(f"Test VQ loss:    {test_metrics['vq']:.4f}")
    print(
        f"Codebook usage on test set: "
        f"{unique_codes}/{num_embeddings} ({usage_percent:.1f}%)"
    )

    epochs = range(1, len(history["train_total"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_total"], label="Train total")
    plt.plot(epochs, history["val_total"], label="Validation total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Architecture 5 Train vs Validation Loss ({DATASET_NAME})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()

    print(f"\nBest checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Loss history saved to:    {HISTORY_PATH}")
    print(f"Training curve saved to:  {CURVE_PATH}")


if __name__ == "__main__":
    main()
