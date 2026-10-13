
"""
run_hyperparameter_sweeps_2.py

Run Architecture_2 full-factorial stages.

Examples:
    python run_hyperparameter_sweeps_2.py
    python run_hyperparameter_sweeps_2.py --only train
    python run_hyperparameter_sweeps_2.py --only extract
    python run_hyperparameter_sweeps_2.py --skip-visualization
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = {
    "train": "train_hyperparameter_sweeps_2.py",
    "extract": "extract_cluster_assignments_sweeps_2.py",
    "compare": "compare_clusters_to_annotations_sweeps_2.py",
    "visualize": "visualize_cluster_assignments_sweeps_2.py",
    "checkpoints": "checkpoint_reader_sweeps_2.py",
}


def run_stage(stage):
    script = SCRIPT_DIR / STAGES[stage]

    print("\n" + "#" * 88)
    print(f"Running stage: {stage}")
    print(f"Script: {script}")
    print("#" * 88)

    subprocess.run([sys.executable, str(script)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=list(STAGES),
        help="Run only one stage.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Run all stages except the expensive UMAP stage.",
    )
    args = parser.parse_args()

    if args.only:
        run_stage(args.only)
        return

    order = ["train", "extract", "compare", "visualize", "checkpoints"]
    for stage in order:
        if args.skip_visualization and stage == "visualize":
            print("Skipping visualization stage.")
            continue
        run_stage(stage)


if __name__ == "__main__":
    main()
