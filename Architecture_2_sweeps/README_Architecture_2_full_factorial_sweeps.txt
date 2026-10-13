
Architecture_2 unfiltered raw-count full factorial sweep
=================================================

Purpose
-------
Train the two-hidden-layer Negative-Binomial Architecture_2 model on:

    PBMC/PBMC_68k_UNFILTERED.h5ad

Full grid
---------
Codebook vectors:
    10, 11, 12, 13, 14

Training batch sizes:
    32, 64, 128, 256, 512

Total:
    5 x 5 = 25 unique models

Only codebook size and batch size change.

Fixed PBMC settings
-------------------
hidden_dim:
    768

latent_dim:
    128

commitment_cost:
    1.012229629179095

learning_rate:
    0.0003042077888794354

weight_decay:
    3.3950954367462734e-09

random seed:
    42

maximum epochs:
    30

early-stopping patience:
    8

The architecture, Negative Binomial loss, annotation file, and other logic are
unchanged. The scheduler uses factor 0.5 and patience 5.

Input provenance
----------------
This sweep uses PBMC_68k_UNFILTERED.h5ad, matching the separate Architecture_2
baseline input. The count matrix remains unnormalized and is obtained from the
counts layer, raw.X, or X, in that order. No extra filtering or normalization is
performed by these sweep scripts.

Run outputs are written under an unfiltered-specific directory to avoid reusing
checkpoints produced earlier with PBMC_68k_NB_filtered.h5ad. If you have older
filtered-data checkpoints, do not copy them into the unfiltered run directory:
those models were trained on a different dataset and must be retrained.

Analysis
--------
Cluster assignments are extracted in batches of 256 cells, while encoder latents
for UMAP are extracted in batches of 512 cells. Missing or Unknown reference
labels are excluded from clustering metrics. Codebook usage in the comparison
table is calculated over all assigned cells. Test-set-only usage is retained in
separately identified columns.

Previously trained models
-------------------------
With SKIP_COMPLETED_RUNS=True, completed checkpoints are reused, even if their
historical training patience differs from the corrected patience of 8. Set
SKIP_COMPLETED_RUNS=False to retrain all 25 models under identical settings.

Suggested location
------------------
Place the sweep files together, for example:

    /lustre/BIF/nobackup/park022/VQVAE/Pipeline2_NB/2HL_sweeps/

If automatic project-root detection fails:

    export VQVAE_PROJECT_ROOT=/lustre/BIF/nobackup/park022/VQVAE

Commands
--------
Activate the environment:

    source /lustre/BIF/nobackup/park022/VQVAE/.venv/bin/activate

Train all 25 configurations:

    CUDA_VISIBLE_DEVICES=0 python run_hyperparameter_sweeps_2.py --only train

Extract assignments:

    CUDA_VISIBLE_DEVICES=0 python run_hyperparameter_sweeps_2.py --only extract

Compare against annotations:

    python run_hyperparameter_sweeps_2.py --only compare

Create checkpoint summary:

    python run_hyperparameter_sweeps_2.py --only checkpoints

Run all stages except UMAP:

    CUDA_VISIBLE_DEVICES=0 python run_hyperparameter_sweeps_2.py --skip-visualization

Run every stage:

    CUDA_VISIBLE_DEVICES=0 python run_hyperparameter_sweeps_2.py

Main output directory
---------------------
    PBMC/vqvae_runs/architecture_2_full_factorial_sweeps_unfiltered/

Important combined outputs
--------------------------
    experiment_manifest.csv
    training_summary.csv
    full_factorial_comparison_table.csv
    best_configurations_by_metric.csv
    checkpoint_summary.csv
    ari_grid_heatmap.png
    nmi_grid_heatmap.png
    overall_purity_grid_heatmap.png
    best_val_loss_grid_heatmap.png
    test_total_loss_grid_heatmap.png
    codebook_usage_percent_grid_heatmap.png

Completed runs
--------------
SKIP_COMPLETED_RUNS=True. A run is skipped when both its checkpoint and
run_result.json exist. Test losses are provided for reporting but are not used
to select the best configuration.
