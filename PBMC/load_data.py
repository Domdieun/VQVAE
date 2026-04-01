"""
2_load_data.py

This script quickly loads the fully preprocessed single-cell dataset
and prepares the PyTorch Dataset and DataLoader for deep learning models.
"""

import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("Loading preprocessed dataset checkpoint...")
# Load the perfectly clean dataset we generated in the first script
anndata = sc.read_h5ad('PBMC/PBMC_68k_filtered_normalized.h5ad')

# Dynamically grab the exact number of genes
input_dimension = anndata.shape[1]
print(f"Dataset successfully loaded! Matrix shape: {anndata.shape}")


# Create PyTorch DataLoader
print("Setting up PyTorch DataLoader...")

class scRNADataset(Dataset):
    def __init__(self, adata):
        if isinstance(adata.X, np.ndarray):
            dense_matrix = adata.X
        else:
            dense_matrix = adata.X.toarray()
        self.X = torch.tensor(dense_matrix, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

dataset = scRNADataset(anndata)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

print("DataLoader is active and ready for the neural network!")