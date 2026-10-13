
"""Extract direct EMA-codebook assignments for every normal PBMC cell."""

import os

import pandas as pd
import scanpy as sc
import torch

from Architecture_3 import VQVAE
from ema_pbmc_config_3 import (
    H5AD_PATH,
    CHECKPOINT_PATH,
    ASSIGNMENTS_CSV,
    ASSIGNMENT_BATCH_SIZE,
)
from load_data_split_3 import (
    make_full_dataloader_from_adata,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )
    params = checkpoint["hyperparameters"]

    model = VQVAE(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        latent_dim=params["latent_dim"],
        num_embeddings=params["num_embeddings"],
        commitment_cost=params["commitment_cost"],
        ema_decay=params["ema_decay"],
        ema_epsilon=params.get(
            "ema_epsilon",
            1e-5,
        ),
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model


def main():
    for path in [H5AD_PATH, CHECKPOINT_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    adata = sc.read_h5ad(H5AD_PATH)

    loader, _input_dim = make_full_dataloader_from_adata(
        adata,
        batch_size=ASSIGNMENT_BATCH_SIZE,
    )

    model = load_model()
    indices = []

    with torch.no_grad():
        for (x_batch,) in loader:
            indices.extend(
                model.encode(
                    x_batch.to(DEVICE)
                )
                .cpu()
                .numpy()
                .tolist()
            )

    output = pd.DataFrame({
        "barcode": adata.obs_names.astype(str),
        "cluster_index": indices,
    })

    os.makedirs(
        os.path.dirname(ASSIGNMENTS_CSV),
        exist_ok=True,
    )
    output.to_csv(
        ASSIGNMENTS_CSV,
        index=False,
    )

    print(f"Saved: {ASSIGNMENTS_CSV}")
    print("\nCluster counts:")
    print(
        output["cluster_index"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()
