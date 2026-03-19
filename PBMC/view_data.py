import scanpy as sc

# 1. Load the file from your PBMC folder
adata = sc.read_h5ad('PBMC_68k_filtered_normalized.h5ad')

# 2. Print the overall summary of what is inside
print("=== Overall Dataset Summary ===")
print(adata)

# 3. Look at the first 5 cells and their metadata (like gene counts)
print("\n=== First 5 Cells (Metadata) ===")
print(adata.obs.head())

# 4. Look at the exact normalized gene expression values for the first 3 cells and 3 genes
print("\n=== Normalized Expression Matrix (3x3 snippet) ===")
print(adata.X[:3, :3].toarray() if hasattr(adata.X, 'toarray') else adata.X[:3, :3])