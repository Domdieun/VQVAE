
"""Run the normal Architecture_3 PBMC pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = {
    "check": "check_data_3.py",
    "optuna": "optuna_vqvae_3.py",
    "train": "train_model_3.py",
    "extract": "extract_cluster_assignments_3.py",
    "compare": "compare_clusters_to_annotations_3.py",
    "visualize": "visualize_cluster_assignments_3.py",
    "checkpoint": "checkpoint_reader_3.py",
}


def run_stage(stage):
    script = SCRIPT_DIR / STAGES[stage]

    print(f"\nRunning {stage}: {script}")

    subprocess.run(
        [sys.executable, str(script)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only",
        choices=list(STAGES),
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help=(
            "Use fallback final-training parameters instead of "
            "running Optuna."
        ),
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
    )

    args = parser.parse_args()

    if args.only:
        run_stage(args.only)
        return

    order = ["check"]

    if not args.skip_optuna:
        order.append("optuna")

    order += [
        "train",
        "extract",
        "compare",
        "visualize",
        "checkpoint",
    ]

    for stage in order:
        if (
            args.skip_visualization
            and stage == "visualize"
        ):
            continue

        run_stage(stage)


if __name__ == "__main__":
    main()
