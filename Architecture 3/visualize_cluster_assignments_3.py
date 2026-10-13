
"""
Create PBMC expression-space and z_e UMAPs for Architecture_3.

No z_q UMAP is calculated.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import torch

from Architecture_3 import VQVAE
from ema_pbmc_config_3 import (
    H5AD_PATH,
    CHECKPOINT_PATH,
    ASSIGNMENTS_CSV,
    ANNOTATION_PATH,
    ANNOTATION_SEP,
    ANNOTATION_BARCODE_COL,
    ANNOTATION_LABEL_COL,
    VISUALIZATION_OUTPUT_DIR,
    INFERENCE_BATCH_SIZE,
)
from load_data_split_3 import (
    make_full_dataloader_from_adata,
)

N_PCS = 30
N_NEIGHBORS = 15
RANDOM_STATE = 42

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


def save_umap(
    adata,
    color,
    title,
    path,
):
    figure = sc.pl.umap(
        adata,
        color=color,
        title=title,
        show=False,
        return_fig=True,
    )
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    for path in [
        H5AD_PATH,
        CHECKPOINT_PATH,
        ASSIGNMENTS_CSV,
        ANNOTATION_PATH,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    os.makedirs(
        VISUALIZATION_OUTPUT_DIR,
        exist_ok=True,
    )

    adata = sc.read_h5ad(H5AD_PATH)

    assignments = pd.read_csv(ASSIGNMENTS_CSV)
    annotations = pd.read_csv(
        ANNOTATION_PATH,
        sep=ANNOTATION_SEP,
        low_memory=False,
    )

    cluster_map = assignments.set_index(
        assignments["barcode"].astype(str).str.strip()
    )["cluster_index"]

    annotation_map = annotations.set_index(
        annotations[ANNOTATION_BARCODE_COL]
        .astype(str)
        .str.strip()
    )[ANNOTATION_LABEL_COL]

    barcodes = adata.obs_names.astype(str).str.strip()

    adata.obs["cluster_index"] = (
        barcodes.map(cluster_map)
        .astype(str)
        .astype("category")
    )
    adata.obs["celltype"] = (
        barcodes.map(annotation_map)
        .fillna("Unknown")
        .astype("category")
    )

    expression_adata = adata.copy()

    sc.tl.pca(
        expression_adata,
        n_comps=min(
            N_PCS,
            expression_adata.n_vars - 1,
        ),
        svd_solver="arpack",
        random_state=RANDOM_STATE,
    )
    sc.pp.neighbors(
        expression_adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=min(
            N_PCS,
            expression_adata.n_vars - 1,
        ),
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(
        expression_adata,
        random_state=RANDOM_STATE,
    )

    save_umap(
        expression_adata,
        "cluster_index",
        "PBMC expression UMAP by EMA VQ cluster",
        os.path.join(
            VISUALIZATION_OUTPUT_DIR,
            "expression_umap_by_vq_cluster_ema.png",
        ),
    )
    save_umap(
        expression_adata,
        "celltype",
        "PBMC expression UMAP by cell type",
        os.path.join(
            VISUALIZATION_OUTPUT_DIR,
            "expression_umap_by_celltype_ema.png",
        ),
    )

    model = load_model()

    loader, _input_dim = make_full_dataloader_from_adata(
        adata,
        batch_size=INFERENCE_BATCH_SIZE,
    )

    z_e_batches = []

    with torch.no_grad():
        for (x_batch,) in loader:
            z_e_batches.append(
                model.encode_continuous(
                    x_batch.to(DEVICE)
                ).cpu()
            )

    z_e = torch.cat(z_e_batches).numpy()

    latent_adata = sc.AnnData(
        X=z_e,
        obs=adata.obs.copy(),
    )

    sc.pp.neighbors(
        latent_adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X",
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(
        latent_adata,
        random_state=RANDOM_STATE,
    )

    save_umap(
        latent_adata,
        "cluster_index",
        "PBMC Architecture_3 z_e UMAP by VQ cluster",
        os.path.join(
            VISUALIZATION_OUTPUT_DIR,
            "z_e_umap_by_vq_cluster_ema.png",
        ),
    )
    save_umap(
        latent_adata,
        "celltype",
        "PBMC Architecture_3 z_e UMAP by cell type",
        os.path.join(
            VISUALIZATION_OUTPUT_DIR,
            "z_e_umap_by_celltype_ema.png",
        ),
    )

    print(
        f"Saved visualizations: "
        f"{VISUALIZATION_OUTPUT_DIR}"
    )
    print("No z_q UMAP was calculated.")


if __name__ == "__main__":
    main()
