"""
Run the complete Architecture_1 comparison pipeline.

Examples:
    python run_hyperparameter_sweeps_1.py
    python run_hyperparameter_sweeps_1.py --skip-visualization
    python run_hyperparameter_sweeps_1.py --only train
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = {
    "train": "train_hyperparameter_sweeps_1.py",
    "extract": "extract_cluster_assignments_sweeps_1.py",
    "compare": "compare_clusters_to_annotations_sweeps_1.py",
    "visualize": "visualize_cluster_assignments_sweeps_1.py",
    "checkpoints": "checkpoint_reader_sweeps_1.py",
}


def run_stage(stage):
    script = SCRIPT_DIR / STAGES[stage]
    print("\n" + "#" * 80)
    print(f"Running stage: {stage}")
    print("#" * 80)
    subprocess.run([sys.executable, str(script)], check=True, cwd=SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=list(STAGES),
        help="Run only one pipeline stage.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Run training, extraction, comparison, and checkpoint summary only.",
    )
    args = parser.parse_args()

    if args.only:
        run_stage(args.only)
        return

    order = ["train", "extract", "compare"]
    if not args.skip_visualization:
        order.append("visualize")
    order.append("checkpoints")

    for stage in order:
        run_stage(stage)


if __name__ == "__main__":
    main()
