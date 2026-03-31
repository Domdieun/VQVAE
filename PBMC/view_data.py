import scanpy as sc

print("Loading datasets... This might take a few seconds.\n")

# 1. Load BOTH files from your PBMC folder
adata_clean = sc.read_h5ad('PBMC_68k_filtered_normalized.h5ad')
adata_raw = sc.read_h5ad('PBMC_68k_UNFILTERED.h5ad')


# PART 1: THE RAW, UNFILTERED DATA


print("\nUNFILTERED RAW DATASET")
print("\nOverall Dataset Summary")
print(adata_raw)

print("\n First 5 Cells (Metadata) ")
print(adata_raw.obs.head())

print("\nRaw Expression Matrix (3x3 snippet)")
# Raw data should have integer counts (e.g., 0, 1, 5, etc.)
print(adata_raw.X[:3, :3].toarray() if hasattr(adata_raw.X, 'toarray') else adata_raw.X[:3, :3])



# PART 2: THE CLEANED, NORMALIZED DATA

print("\nCLEANED & NORMALIZED DATASET")

print("\nOverall Dataset Summary")
print(adata_clean)

print("\nFirst 5 Cells (Metadata)")
print(adata_clean.obs.head())

print("\nNormalized Expression Matrix (3x3 snippet)")
# Clean data should have decimals from the log-transformation
print(adata_clean.X[:3, :3].toarray() if hasattr(adata_clean.X, 'toarray') else adata_clean.X[:3, :3])

import scanpy as sc
import pandas as pd

print("\nLoading cleaned dataset...")
adata = sc.read_h5ad('PBMC_68k_filtered_normalized.h5ad')

print("THE DATA MATRIX (20 Cells x 20 Genes)")

# 1. Extract the 20x20 chunk of the matrix
matrix_chunk = adata.X[:20, :20]

# 2. Convert it from a compressed sparse format to a standard grid of numbers
if hasattr(matrix_chunk, 'toarray'):
    matrix_chunk = matrix_chunk.toarray()

# 3. Grab the actual names for the rows (Cells) and columns (Genes)
cell_barcodes = adata.obs_names[:20]
gene_names = adata.var_names[:20]

# 4. Wrap it in a Pandas DataFrame for beautiful terminal printing
df = pd.DataFrame(matrix_chunk, index=cell_barcodes, columns=gene_names)

# Force Pandas to display all 20 columns without hiding them in the middle
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

print(df)

