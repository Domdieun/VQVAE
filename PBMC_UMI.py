"""
PBMC_UMI.py

This script processes raw 10x Genomics single-cell RNA sequencing data (PBMCs).
It performs data loading, Quality Control (QC) visualization, biological filtering, 
normalization, and log-transformation. Finally, it saves the processed dataset 
and prepares a PyTorch DataLoader for downstream deep learning (VQ-VAE).
"""

# Import necessary libraries
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Data
# ==========================================
print("Loading data ...")
# AnnData is the standard container for scRNA data in python
# anndata.X: the main matrix (Cells x Genes)
# anndata.obs: cell barcodes (rows)
# anndata.var: genes (columns)
anndata = sc.read_10x_mtx(
    './filtered_matrices_mex/hg19/',
    var_names='gene_symbols',
    cache=True
)

print(f"Initial raw dataset shape: {anndata.shape}") # Should be ~ 68,579 cells x 32,738 genes

# ==========================================
# 2. Quality Control (QC) Metrics
# ==========================================
print("\nCalculating QC metrics...")
# This function automatically counts how many genes each cell has,
# and how many cells each gene is found in. (inplace=True saves it inside anndata)
sc.pp.calculate_qc_metrics(anndata, percent_top=None, log1p=False, inplace=True)

print("\n--- CELL STATISTICS (Genes per cell) ---")
print(anndata.obs['n_genes_by_counts'].describe())

print("\n--- GENE STATISTICS (Cells per gene) ---")
print(anndata.var['n_cells_by_counts'].describe())

# ==========================================
# 3. Pre-Filtering Visualizations
# ==========================================
print("\nGenerating QC plots. Please close the plot windows to continue the script...")

# Plot a violin plot of the genes detected per cell and total counts
sc.pl.violin(anndata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)

# Plot a histogram to visualize the distribution BEFORE we delete any cells
plt.hist(anndata.obs['n_genes_by_counts'], bins=100, color='blue', edgecolor='black')
plt.axvline(x=200, color='red', linestyle='dashed', linewidth=2, label='Min cut-off (200)')
plt.axvline(x=2500, color='green', linestyle='dashed', linewidth=2, label='Max cut-off (2500)')
plt.title('Distribution of Genes per Cell BEFORE Filtering')
plt.xlabel('Number of Genes')
plt.ylabel('Number of Cells')
plt.legend()
plt.show()

# ==========================================
# 4. Biological Filtering
# ==========================================
print("\nFiltering dataset...")
# Filter out "ghost" genes that are expressed in fewer than 3 cells
sc.pp.filter_genes(anndata, min_cells=3)

# Filter out dying/damaged cells with too few genes
sc.pp.filter_cells(anndata, min_genes=200)

# Filter out artificial doublets (two cells stuck together) with abnormally high counts
sc.pp.filter_cells(anndata, max_genes=2500) 

# ==========================================
# 5. Normalization & Log-Transformation
# ==========================================
print("Normalizing and log-transforming data...")
# Normalize every cell to have the same total count (10,000) so they are comparable
sc.pp.normalize_total(anndata, target_sum=1e4)

# Log-transform the data to squash extreme outliers (crucial for neural networks)
sc.pp.log1p(anndata)

print(f"Shape after preprocessing: {anndata.shape}")

# ==========================================
# 6. Save Checkpoint & Set Dimensions
# ==========================================
# Save the exact number of genes remaining so the VQ-VAE knows how big its input layer must be
input_dimension = anndata.shape[1]

# Save the fully processed matrix locally so we can skip this pipeline in the future if needed
print("Saving preprocessed data to local file...")
anndata.write('preprocessed_pbmc.h5ad')
print("Successfully saved as 'preprocessed_pbmc.h5ad'!")

# ==========================================
# 7. Create PyTorch DataLoader
# ==========================================
print("\nSetting up PyTorch DataLoader...")

class scRNADataset(Dataset):
    """
    A custom PyTorch Dataset class to handle Scanpy AnnData objects.
    Converts the sparse matrix into dense PyTorch tensors for neural network ingestion.
    """
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

# Instantiate the dataset
dataset = scRNADataset(anndata)

# Create the dataloader (feeds batches of 256 randomized cells at a time)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

print("DataLoader ready for the VQ-VAE!")