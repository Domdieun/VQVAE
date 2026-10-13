"""
checkpoint_reader_1.py

Inspect a saved VQ-VAE checkpoint.
"""

import os
from pathlib import Path
import torch
from load_data_split_1 import PROJECT_ROOT

DATASET_NAME = os.environ.get("VQVAE_DATASET", "PBMC")
CHECKPOINT_PATH = str(PROJECT_ROOT / DATASET_NAME / "vqvae_runs" / "train_val_test" / "best_vqvae_checkpoint.pth")


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find checkpoint: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    print("\n=== Checkpoint keys ===")
    print(ckpt.keys())

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

    print("\n=== Model state dict keys ===")
    if "model_state_dict" in ckpt:
        for k in ckpt["model_state_dict"].keys():
            print(k)
    else:
        print("No model_state_dict found")


if __name__ == "__main__":
    main()