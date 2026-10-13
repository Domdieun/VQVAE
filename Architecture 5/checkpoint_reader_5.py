"""
checkpoint_reader_5.py

Inspect a saved Architecture_5 VQ-VAE checkpoint.
"""

import os
import torch

from load_data_split_5 import project_path, PROJECT_ROOT


DATASET_NAME = "PBMC"  # change to "PBMC" or "Ath"

CHECKPOINT_PATHS = {
    "PBMC": project_path("PBMC/vqvae_runs/train_val_test_5/best_vqvae_5_checkpoint.pth"),
    "Ath": project_path("Ath/vqvae_runs/train_val_test_5/best_vqvae_5_checkpoint.pth"),
}

CHECKPOINT_PATH = CHECKPOINT_PATHS[DATASET_NAME]


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    print("\n=== Checkpoint keys ===")
    print(ckpt.keys())

    print("\n=== Architecture ===")
    print(ckpt.get("architecture", "No architecture found"))

    print("\n=== Loss ===")
    print(ckpt.get("loss", "No loss found"))

    print("\n=== Epoch ===")
    print(ckpt.get("epoch", "No epoch found"))

    print("\n=== Hyperparameters ===")
    print(ckpt.get("hyperparameters", "No hyperparameters found"))

    print("\n=== Best validation loss ===")
    print(ckpt.get("best_val_loss", "No best_val_loss found"))

    print("\n=== Train losses ===")
    print(ckpt.get("train_losses", "No train_losses found"))

    print("\n=== Validation losses ===")
    print(ckpt.get("val_losses", "No val_losses found"))

    print("\n=== Final losses ===")
    print(ckpt.get("final_losses", "No final_losses found"))

    print("\n=== Codebook usage ===")
    print(ckpt.get("codebook_usage", "No codebook_usage found"))

    print("\n=== Model state dict keys ===")
    if "model_state_dict" in ckpt:
        for k in ckpt["model_state_dict"].keys():
            print(k)
    else:
        print("No model_state_dict found")


if __name__ == "__main__":
    main()
