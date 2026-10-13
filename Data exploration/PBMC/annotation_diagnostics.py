"""
diagnose_pbmc_annotation_matching.py

Diagnose why PBMC cell type annotations are not matching the barcodes
in the PBMC h5ad file.

This script checks:
1. AnnData obs_names barcode format
2. Annotation barcode format
3. Whether barcodes overlap directly
4. Whether barcodes overlap after removing -1 suffix
5. Whether the annotation file has duplicate barcodes
6. Whether duplicate barcodes have conflicting cell type labels

Run:
    python diagnose_pbmc_annotation_matching.py
"""

import os
import pandas as pd
import scanpy as sc


# -----------------------------
# CONFIG
# -----------------------------

H5AD_PATH = "../PBMC_68k_UNFILTERED.h5ad"

ANNOTATION_PATH = "../68k_pbmc_barcodes_annotation.tsv"

ANNOTATION_HAS_HEADER = False
ANNOTATION_SEP = ","
ANNOTATION_NAMES = ["umap_x", "umap_y", "barcode", "cell_type_annotation"]

ANNOTATION_BARCODE_COL = "barcode"
ANNOTATION_CELL_TYPE_COL = "cell_type_annotation"


# -----------------------------
# BARCODE CLEANING FUNCTIONS
# -----------------------------

def clean_barcode_basic(barcode):
    """
    Basic barcode cleaning:
    - convert to string
    - remove surrounding whitespace
    """

    return str(barcode).strip()


def clean_barcode_remove_10x_suffix(barcode):
    """
    More tolerant PBMC barcode cleaning.

    Converts:
        AAACATACAACCAC-1

    into:
        AAACATACAACCAC

    This helps when one file contains the 10x -1 suffix
    and the other file does not.
    """

    barcode = str(barcode).strip()

    if barcode.endswith("-1"):
        barcode = barcode[:-2]

    return barcode


# -----------------------------
# LOADING FUNCTIONS
# -----------------------------

def load_adata():
    """
    Load PBMC h5ad file.
    """

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find h5ad file: {H5AD_PATH}")

    print(f"Loading AnnData from: {H5AD_PATH}")
    adata = sc.read_h5ad(H5AD_PATH)

    print("\nAnnData summary:")
    print(adata)
    print(f"Number of cells: {adata.n_obs}")
    print(f"Number of genes: {adata.n_vars}")

    return adata


def load_annotation_table():
    """
    Load PBMC annotation file.
    """

    if not os.path.exists(ANNOTATION_PATH):
        raise FileNotFoundError(f"Could not find annotation file: {ANNOTATION_PATH}")

    print(f"\nLoading annotation file from: {ANNOTATION_PATH}")

    if ANNOTATION_HAS_HEADER:
        annotation_df = pd.read_csv(
            ANNOTATION_PATH,
            sep=ANNOTATION_SEP,
            low_memory=False,
        )
    else:
        annotation_df = pd.read_csv(
            ANNOTATION_PATH,
            sep=ANNOTATION_SEP,
            header=None,
            names=ANNOTATION_NAMES,
            low_memory=False,
        )

    print("\nAnnotation file summary:")
    print(f"Shape: {annotation_df.shape}")
    print("Columns:")
    print(annotation_df.columns.tolist())

    required_cols = [ANNOTATION_BARCODE_COL, ANNOTATION_CELL_TYPE_COL]

    missing_cols = [
        col for col in required_cols
        if col not in annotation_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Annotation file is missing columns: {missing_cols}")

    return annotation_df


# -----------------------------
# DIAGNOSTICS
# -----------------------------

def print_barcode_examples(adata, annotation_df):
    """
    Print examples of AnnData and annotation barcodes.
    """

    print("\n" + "=" * 80)
    print("BARCODE EXAMPLES")
    print("=" * 80)

    adata_barcodes = pd.Series(adata.obs_names.astype(str))
    annotation_barcodes = annotation_df[ANNOTATION_BARCODE_COL].astype(str)

    print("\nFirst 20 AnnData obs_names:")
    print(adata_barcodes.head(20).tolist())

    print("\nFirst 20 annotation barcodes:")
    print(annotation_barcodes.head(20).tolist())

    print("\nLast 5 AnnData obs_names:")
    print(adata_barcodes.tail(5).tolist())

    print("\nLast 5 annotation barcodes:")
    print(annotation_barcodes.tail(5).tolist())


def diagnose_duplicate_annotations(annotation_df):
    """
    Check whether the annotation file has duplicate barcodes.
    """

    print("\n" + "=" * 80)
    print("DUPLICATE ANNOTATION BARCODE DIAGNOSTIC")
    print("=" * 80)

    df = annotation_df.copy()

    df[ANNOTATION_BARCODE_COL] = (
        df[ANNOTATION_BARCODE_COL]
        .astype(str)
        .map(clean_barcode_basic)
    )

    df[ANNOTATION_CELL_TYPE_COL] = (
        df[ANNOTATION_CELL_TYPE_COL]
        .astype(str)
        .str.strip()
    )

    total_rows = len(df)
    unique_barcodes = df[ANNOTATION_BARCODE_COL].nunique()
    duplicated_rows = df[ANNOTATION_BARCODE_COL].duplicated(keep=False).sum()

    print(f"Total annotation rows: {total_rows}")
    print(f"Unique annotation barcodes: {unique_barcodes}")
    print(f"Rows with duplicated barcodes: {duplicated_rows}")

    if duplicated_rows == 0:
        print("No duplicated annotation barcodes found.")
        return

    duplicated_df = df[
        df[ANNOTATION_BARCODE_COL].duplicated(keep=False)
    ].copy()

    conflict_counts = (
        duplicated_df
        .groupby(ANNOTATION_BARCODE_COL)[ANNOTATION_CELL_TYPE_COL]
        .nunique()
    )

    conflicting_barcodes = conflict_counts[conflict_counts > 1]

    print(f"Duplicated barcode groups: {conflict_counts.shape[0]}")
    print(f"Duplicated barcode groups with conflicting labels: {len(conflicting_barcodes)}")

    if len(conflicting_barcodes) > 0:
        print("\nFirst 10 conflicting duplicated barcodes:")
        print(conflicting_barcodes.head(10))

        first_conflicting_barcode = conflicting_barcodes.index[0]

        print(f"\nExample conflicting rows for barcode: {first_conflicting_barcode}")
        print(
            duplicated_df[
                duplicated_df[ANNOTATION_BARCODE_COL] == first_conflicting_barcode
            ][
                [ANNOTATION_BARCODE_COL, ANNOTATION_CELL_TYPE_COL]
            ]
        )
    else:
        print(
            "Duplicated barcodes exist, but they do not have conflicting labels. "
            "They are likely repeated duplicate rows."
        )


def diagnose_matching_strategy(adata, annotation_df, cleaner, strategy_name):
    """
    Compare AnnData barcodes and annotation barcodes using one cleaning strategy.
    """

    print("\n" + "=" * 80)
    print(f"MATCHING DIAGNOSTIC: {strategy_name}")
    print("=" * 80)

    adata_barcodes = pd.Series(adata.obs_names.astype(str)).map(cleaner)
    annotation_barcodes = (
        annotation_df[ANNOTATION_BARCODE_COL]
        .astype(str)
        .map(cleaner)
    )

    adata_set = set(adata_barcodes)
    annotation_set = set(annotation_barcodes)

    overlap = adata_set.intersection(annotation_set)

    print(f"AnnData cells: {adata.n_obs}")
    print(f"Unique AnnData barcodes after cleaning: {len(adata_set)}")
    print(f"Annotation rows: {len(annotation_df)}")
    print(f"Unique annotation barcodes after cleaning: {len(annotation_set)}")
    print(f"Barcode overlap: {len(overlap)}")

    if adata.n_obs > 0:
        print(f"Fraction of AnnData cells potentially annotated: {len(overlap) / adata.n_obs:.4f}")

    if len(annotation_set) > 0:
        print(f"Fraction of annotation barcodes found in AnnData: {len(overlap) / len(annotation_set):.4f}")

    print("\nFirst 20 overlapping barcodes:")
    print(list(overlap)[:20])

    missing_from_annotation = list(adata_set - annotation_set)
    annotation_not_in_adata = list(annotation_set - adata_set)

    print("\nFirst 20 AnnData barcodes NOT found in annotation:")
    print(missing_from_annotation[:20])

    print("\nFirst 20 annotation barcodes NOT found in AnnData:")
    print(annotation_not_in_adata[:20])


def diagnose_possible_separator_issue():
    """
    Try reading the annotation file with common separators to check whether
    the current separator is correct.
    """

    print("\n" + "=" * 80)
    print("ANNOTATION SEPARATOR DIAGNOSTIC")
    print("=" * 80)

    separators = {
        "comma": ",",
        "tab": "\t",
        "semicolon": ";",
    }

    for name, sep in separators.items():
        try:
            df = pd.read_csv(
                ANNOTATION_PATH,
                sep=sep,
                header=None,
                nrows=5,
            )

            print(f"\nReading with {name} separator {repr(sep)}:")
            print(f"Shape of first 5 rows: {df.shape}")
            print(df.head())

        except Exception as e:
            print(f"\nCould not read with {name} separator {repr(sep)}:")
            print(e)


def suggest_fix(adata, annotation_df):
    """
    Print a practical recommendation based on matching overlap.
    """

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    adata_basic = set(
        pd.Series(adata.obs_names.astype(str)).map(clean_barcode_basic)
    )
    ann_basic = set(
        annotation_df[ANNOTATION_BARCODE_COL]
        .astype(str)
        .map(clean_barcode_basic)
    )

    adata_no_suffix = set(
        pd.Series(adata.obs_names.astype(str)).map(clean_barcode_remove_10x_suffix)
    )
    ann_no_suffix = set(
        annotation_df[ANNOTATION_BARCODE_COL]
        .astype(str)
        .map(clean_barcode_remove_10x_suffix)
    )

    overlap_basic = len(adata_basic.intersection(ann_basic))
    overlap_no_suffix = len(adata_no_suffix.intersection(ann_no_suffix))

    print(f"Direct/basic barcode overlap: {overlap_basic}")
    print(f"Overlap after removing -1 suffix: {overlap_no_suffix}")

    if overlap_no_suffix > overlap_basic:
        print(
            "\nRecommended fix:"
            "\nUse the PBMC barcode cleaner that removes the -1 suffix."
        )
        print(
            """
Replace your get_pbmc_barcode() function with:

def get_pbmc_barcode(barcode):
    barcode = str(barcode).strip()

    if barcode.endswith("-1"):
        barcode = barcode[:-2]

    return barcode
"""
        )

    elif overlap_basic > 0:
        print(
            "\nYour barcodes already overlap using basic cleaning."
            "\nIf annotation UMAP was skipped, check whether the script is reading "
            "the correct annotation file and using the correct barcode column."
        )

    else:
        print(
            "\nNo overlap was found even after removing -1 suffix."
            "\nThis suggests one of these problems:"
            "\n1. The annotation file does not correspond to this h5ad file."
            "\n2. The h5ad file is unfiltered, but the annotation file is for filtered cells."
            "\n3. The annotation file is being read with the wrong separator."
            "\n4. The barcode column is not the column you think it is."
        )


# -----------------------------
# MAIN
# -----------------------------

def main():
    print("PBMC annotation matching diagnostic")
    print("=" * 80)

    adata = load_adata()
    annotation_df = load_annotation_table()

    print_barcode_examples(adata, annotation_df)

    diagnose_possible_separator_issue()

    diagnose_duplicate_annotations(annotation_df)

    diagnose_matching_strategy(
        adata=adata,
        annotation_df=annotation_df,
        cleaner=clean_barcode_basic,
        strategy_name="basic cleaning only",
    )

    diagnose_matching_strategy(
        adata=adata,
        annotation_df=annotation_df,
        cleaner=clean_barcode_remove_10x_suffix,
        strategy_name="remove -1 suffix",
    )

    suggest_fix(adata, annotation_df)


if __name__ == "__main__":
    main()