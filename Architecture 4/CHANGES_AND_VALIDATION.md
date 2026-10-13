# Architecture 4 — thesis alignment and validation

**Reference:** `Thesis Report FINAL.pdf`, treated as authoritative. Section 2.2 (architecture), Section 2.3 (tuning and training), Section 2.4 (downstream analysis), and Section 3.4/Table 1 (Architecture 4 on PBMC).

## Corrections

1. **Architecture 4 model**: Preserved the existing two-layer encoder/decoder, normalized-data MSE loss, squared Euclidean nearest-code assignment, straight-through estimator, gradient-based codebook update, and orthogonality penalty. No math/weighting changes were justified by the report.
2. **PBMC experiment**: Set dataset defaults in Optuna, final training, assignment extraction, and visualization to PBMC. Other optional dataset support remains, but is not part of the reported Architecture 4 result.
3. **Tuning**: Adjusted Optuna trials from 20 to 100 and maximum epochs from 20 to 30; early-stopping patience 8 was already correct. Added ReduceLROnPlateau on total validation loss with factor 0.5 and patience 5, as reported.
4. **Final training**: Aligned epoch limit to 30, early-stopping patience from 10 to 8, and scheduler patience from 3 to 5. The report does not separately spell out a *different* final-training epoch cap, so its described 30-epoch training configuration is used throughout this package.
5. **Assignment extraction**: Batch size from 512 to 256. Added a small JSON summary of whole-dataset codebook utilization (active/total codebook entries and percent) as described in the report.
6. **Annotation metrics**: Excluded missing/NaN labels before converting them to strings; `Unknown` exclusion and barcode matching were retained.
7. **Imports and paths**: Standardized the 7 script filenames and their local imports; normalized PBMC/Ath paths to be resolved inside `VQVAE_PROJECT_ROOT` rather than two directories above it. Left optional Ath annotation filename variants unchanged because Architecture 4 is PBMC-only in the report and the actual Ath filename was not provided.
8. **Annotations**: Added short `# Thesis alignment ...` comments at adjusted functional sections so you can identify each change in the scripts. See `UNIFIED_DIFFS.patch` for every line-level change.

## Explicitly preserved because the report does not provide a distinct A4 override

- Orthogonal regularization weights `[0, 1e-4, 1e-3, 1e-2, 0.05, 0.1]` already match Methods 2.3.
- `num_embeddings = 11` fixed for the PBMC A4 optimization remains unchanged. The report lists baseline codebook search sizes and separate PBMC sweeps, but does not explicitly state that Architecture 4 tuned its codebook size. The code's fixed-K procedure therefore is **not** silently replaced with a different experiment.
- Existing fallback hyperparameter values remain unchanged; the thesis does not enumerate A4's winning Optuna trial hyperparameters.
- Reference ARI/NMI/purity values (0.1967/0.3519/0.5383) are **not hardcoded**. Matching them requires original datasets, saved checkpoint(s), and experimental environment.

## How to use

Place the seven Python files in the same folder. Place the PBMC dataset under `PROJECT_ROOT/PBMC/` with the names already used by the scripts. Set `VQVAE_PROJECT_ROOT` to the project root if auto-discovery does not find it. Run:

```bash
python optuna_vqvae_4.py
python train_model_4.py
python extract_cluster_assignments_4.py
python compare_clusters_to_annotations_4.py
python visualize_cluster_assignments_4.py
```

## Validation

- All seven corrected Python scripts checked with AST parsing / `py_compile`.
- Passed a synthetic-data functional smoke test covering: path/import consistency, PyTorch encoder/quantizer/decoder forward and backward passes, a one-epoch Optuna objective, one-epoch final training, best-checkpoint reload and test evaluation, direct cluster extraction and codebook utilization, PBMC barcode matching, Unknown/missing-label exclusion, ARI/NMI/purity, and heatmap output. Visualization defaults and shared paths were checked; rendering UMAP images was not executed because Scanpy is unavailable in the test environment. The 100-trial Optuna search and 30-epoch final PBMC training were not run.
- A pre-existing edge case remains: a training batch of size one can cause BatchNorm to fail because DataLoader does not drop the final small batch. This was left unchanged as it was not a mismatch explicitly established by the report.
- No dataset-scale rerun or reproduction of thesis metrics is claimed.

## Note

The initial pipeline materializes the full expression matrix into dense NumPy and PyTorch tensors; PBMC 68k can therefore demand substantial CPU memory. This original implementation has not been redesigned because it is a performance constraint, not a demonstrated discrepancy with the report.

## Clean script version
All annotation comments specifically calling out thesis-based adjustments have been removed from the Python scripts. Other ordinary code documentation and the implementation itself remain unchanged. The patch in this archive reflects this clean version.
