#Import necessary libraries
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Load dataset using scanpy for correct representation
#AnnData; standard container for scRNA data in python
#anndata.X: the main matrix (Cells x Genes)
#anndata.obs: cell barcodes
#anndata.var: genes
print("Loading data ...")
anndata = sc.read_10x_mtx(
    './filtered_matrices_mex/hg19/',
    var_names = 'gene_symbols',
    cache=True
)

print(anndata)
#68,579 cells x 32,738 genes

#Inspection of the data
#qc check before filtering
print("\nCalculating QC metrics...")

# This function automatically counts how many genes each cell has,
# and how many cells each gene is found in. (inplace=True saves it inside anndata)
sc.pp.calculate_qc_metrics(anndata, percent_top=None, log1p=False, inplace=True)

print("\n CELL STATISTICS (Genes per cell)")
# anndata.obs['n_genes_by_counts'] holds the number of genes detected in each cell
print(anndata.obs['n_genes_by_counts'].describe())

print("\n GENE STATISTICS (Cells per gene)")
# anndata.var['n_cells_by_counts'] holds the number of cells each gene was found in
print(anndata.var['n_cells_by_counts'].describe())

# Plot a violin plot of the genes detected per cell
sc.pl.violin(anndata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)

#Filtering of the data based on qc results
#filters out genes that are only expressed in 3 cells max
sc.pp.filter_genes(anndata, min_cells=3)
#filters out cells that have less genes (dead cells)
sc.pp.filter_cells(anndata, min_genes=200)
#filters out cells
sc.pp.filter_cells(anndata, max_genes=2500) # high counts - > removes doublets

# Plot a histogram of the genes per cell
plt.hist(anndata.obs['n_genes_by_counts'], bins=100, color='blue', edgecolor='black')
plt.axvline(x=200, color='red', linestyle='dashed', linewidth=2, label='Min cut-off (200)')
plt.axvline(x=2500, color='green', linestyle='dashed', linewidth=2, label='Max cut-off (2500)')
plt.title('Distribution of Genes per Cell')
plt.xlabel('Number of Genes')
plt.ylabel('Number of Cells')
plt.legend()
plt.show()

#Data Normalisation
# Normalize every cell to have the same total count (10,000)
sc.pp.normalize_total(anndata, target_sum=1e4)

# Log-transform the data (crucial for neural networks)
sc.pp.log1p(anndata)