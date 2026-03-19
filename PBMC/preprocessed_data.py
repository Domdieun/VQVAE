"""
step1_preprocess.py

This script processes raw 10x Genomics single-cell RNA sequencing data (PBMCs).
It performs data loading, QC visualization, biological filtering, normalization,
and log-transformation. Finally, it tests the transformation and saves a clean checkpoint.
It also exports a completely RAW, unfiltered dataset for experimental comparison.
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Load Data
# ==========================================
print("Loading raw data ...")
anndata = sc.read_10x_mtx(
    './filtered_matrices_mex/hg19/',
    var_names='gene_symbols',
    cache=True
)
print(f"Initial raw dataset shape: {anndata.shape}")

# ==========================================
# 2. Quality Control (QC) & Visualizations
# ==========================================
print("\nCalculating QC metrics...")
sc.pp.calculate_qc_metrics(anndata, percent_top=None, log1p=False, inplace=True)

print("\nGenerating QC plots. Please close the windows to continue...")
sc.pl.violin(anndata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)

plt.hist(anndata.obs['n_genes_by_counts'], bins=100, color='blue', edgecolor='black')
plt.axvline(x=200, color='red', linestyle='dashed', linewidth=2, label='Min cut-off (200)')
plt.axvline(x=2500, color='green', linestyle='dashed', linewidth=2, label='Max cut-off (2500)')
plt.title('Distribution of Genes per Cell BEFORE Filtering')
plt.xlabel('Number of Genes')
plt.ylabel('Number of Cells')
plt.legend()
plt.show()

# ==========================================
# 2.5 Save Raw/Unfiltered Checkpoint (For Comparison)
# ==========================================
raw_filename = 'PBMC_68k_UNFILTERED.h5ad'
print(f"\n[!] Saving UNFILTERED raw data to local file: {raw_filename}...")
# This saves the matrix exactly as it is right now (68579 x 32738)
anndata.write(raw_filename)
print("Raw dataset saved!")

# ==========================================
# 3. Biological Filtering
# ==========================================
print("\nFiltering dataset...")
sc.pp.filter_genes(anndata, min_cells=3)
sc.pp.filter_cells(anndata, min_genes=200)
sc.pp.filter_cells(anndata, max_genes=2500)

# ==========================================
# 4. Normalization, Log-Transformation & VERIFICATION
# ==========================================
print("\nNormalizing data to 10,000 counts per cell...")
sc.pp.normalize_total(anndata, target_sum=1e4)

# VERIFICATION 1: Did normalization work?
cell_sums = np.asarray(anndata.X[:3, :].sum(axis=1)).flatten()
print(f"--> Verification - Total counts for first 3 cells: {cell_sums} (Should be ~10000)")

print("\nLog-transforming data...")
sc.pp.log1p(anndata)

# VERIFICATION 2: Did log1p work?
max_val = anndata.X.max()
print(f"--> Verification - Maximum expression value in entire matrix: {max_val:.2f} (Should be < 15)")

print(f"\nShape after all preprocessing: {anndata.shape}")

# ==========================================
# 5. Save Clean Checkpoint
# ==========================================
final_filename = 'PBMC_68k_filtered_normalized.h5ad'
print(f"\nSaving preprocessed data to local file: {final_filename}...")
anndata.write(final_filename)
print("Data pipeline complete!")