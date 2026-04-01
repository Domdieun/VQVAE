#!/usr/bin/env python3

"""
evaluate_and_plot.py

This script loads the trained VQ-VAE weights, passes the entire dataset
through the network sequentially to get the discrete cluster ID for each cell,
and then uses Scanpy to generate a 2D UMAP visualization of the clusters.
"""
from datetime import datetime
import os
import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Import your architecture blueprint
from Architecture_1 import scRNA_VQVAE


def main():
    # 1. Load the Data (Strictly Un-shuffled)
    print("Loading preprocessed AnnData...")
    # Update this path if your file is in a different folder
    adata = sc.read_h5ad('PBMC/PBMC_68k_filtered_normalized.h5ad')

    class scRNADatasetEval(Dataset):
        def __init__(self, anndata):
            dense_matrix = anndata.X.toarray() if hasattr(anndata.X, 'toarray') else anndata.X
            self.X = torch.tensor(dense_matrix, dtype=torch.float32)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx]

    # Notice: shuffle=False and drop_last=False!
    eval_dataset = scRNADatasetEval(adata)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, drop_last=False)

    # 2. Find and Load the Trained Model
    ckpt_dir = './saved_models/'
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Could not find {ckpt_dir}. Did the training script finish?")

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError("No .ckpt files found in the saved_models directory.")

    # Grab the most recently saved checkpoint
    best_model_path = os.path.join(ckpt_dir, ckpt_files[-1])
    print(f"Loading trained weights from: {best_model_path}")

    # Load the model and put it in evaluation mode (turns off dropout, EMA updates, etc.)
    model = scRNA_VQVAE.load_from_checkpoint(best_model_path)
    model.eval()

    # Use GPU if available to speed up inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 3. The Inference Pass (Get Cluster IDs)
    print("Pushing cells through the VQ-VAE to extract cluster IDs...")
    all_cluster_ids = []

    with torch.no_grad():  # Turn off gradients to save memory
        for batch in eval_loader:
            batch = batch.to(device)

            # Step A: Compress the genes
            enc = model.encoder(batch)

            # Step B: Snap to codebook and get the cluster index
            _, _, encoding_indices = model.vq_layer(enc)

            # Store the cluster ID for each cell
            all_cluster_ids.extend(encoding_indices.cpu().numpy().flatten())

    # 4. Attach to Scanpy and Analyze
    # Convert numbers to strings/categories so Scanpy knows they are distinct groups, not a color gradient
    adata.obs['vq_cluster'] = [str(int(c)) for c in all_cluster_ids]
    adata.obs['vq_cluster'] = adata.obs['vq_cluster'].astype('category')

    # Quick diagnostic: How many of the 512 clusters did the model actually use?
    unique_clusters = adata.obs['vq_cluster'].nunique()
    print(f"\nDiagnostic: Out of 512 available clusters, the VQ-VAE used {unique_clusters} unique clusters.")

    # 5. Generate and Save UMAP
    print("\nCalculating PCA and UMAP geometry...")
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)

    #Create Directory for Results
    output_dir = "cluster_results"
    os.makedirs(output_dir, exist_ok=True)

    # Create a unique filename using the current date/time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"vqvae_clusters_{timestamp}.png"

    print(f"Rendering plot and saving to {output_dir}/{save_path}...")

    # Save the plot
    sc.pl.umap(
        adata,
        color=['vq_cluster'],
        legend_loc='on data',
        title='VQ-VAE Learned Clusters',
        show=False,  # Don't stop the script to show the window
        save=f"_{save_path}"  # Scanpy adds 'umap' prefix automatically
    )

    # Move the file into your specific folder (Scanpy defaults to a 'figures' folder)
    if os.path.exists(f"figures/umap_{save_path}"):
        os.rename(f"figures/umap_{save_path}", os.path.join(output_dir, save_path))

    print(f"Done! Plot saved in {output_dir}")



if __name__ == "__main__":
    main()