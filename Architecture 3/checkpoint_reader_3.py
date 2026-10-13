
"""Inspect an Architecture_3 checkpoint and its EMA state."""

import os

import torch

from ema_pbmc_config_3 import CHECKPOINT_PATH


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    keys = [
        "dataset_name",
        "h5ad_path",
        "architecture",
        "loss",
        "codebook_update",
        "epoch",
        "best_val_loss",
        "hyperparameters",
        "train_losses",
        "val_losses",
        "final_losses",
        "codebook_usage",
        "ema_state",
    ]

    for key in keys:
        print(f"\n=== {key} ===")
        print(checkpoint.get(key))

    state_dict = checkpoint.get(
        "model_state_dict",
        {},
    )

    ema_keys = [
        key for key in state_dict
        if (
            "ema_cluster_size" in key
            or "ema_embedding_sum" in key
            or "ema_update_steps" in key
        )
    ]

    print("\n=== EMA buffers stored in state_dict ===")
    for key in ema_keys:
        value = state_dict[key]
        print(f"{key}: shape={tuple(value.shape)}")


if __name__ == "__main__":
    main()
