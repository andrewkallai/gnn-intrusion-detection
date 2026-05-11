#!/usr/bin/env python3
"""Orchestrate Module A -> Module B -> Module C for the IDS GCN workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the integrated UNSW IDS GCN workflow.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Folder containing UNSW-NB15_*.csv files.")
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graph_unsw_full_10min_stratified_clean"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/graph_unsw_full_10min_stratified_clean/module_b_results"),
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=Path("data/graph_unsw_full_10min_stratified_clean/module_c_test_viz"),
    )
    parser.add_argument("--window", default="10min")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-module-a", action="store_true", help="Reuse existing graph artifacts.")
    parser.add_argument("--skip-module-b", action="store_true", help="Reuse existing trained model outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.skip_module_a:
        run_step(
            [
                sys.executable,
                "gcn_ids/data_graph.py",
                "--input-files",
                str(args.dataset_dir),
                "--input-glob",
                "UNSW-NB15_*.csv",
                "--output-dir",
                str(args.graph_dir),
                "--window",
                args.window,
                "--split-ratio",
                "0.6,0.2,0.2",
                "--split-strategy",
                "stratified_attack_presence",
                "--node-label-rule",
                "any_malicious_wins",
                "--save-cleaned-flows",
                "--seed",
                str(args.seed),
            ]
        )

    if not args.skip_module_b:
        run_step(
            [
                sys.executable,
                "gcn_ids/learning.py",
                "--graph-dir",
                str(args.graph_dir),
                "--output-dir",
                str(args.results_dir),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
            ]
        )

    run_step(
        [
            sys.executable,
            "gcn_ids/graph_viz.py",
            "--graph-dir",
            str(args.graph_dir),
            "--predictions-csv",
            str(args.results_dir / "test_predictions.csv"),
            "--output-dir",
            str(args.viz_dir),
            "--max-graphs",
            "16",
            "--layout-seed",
            str(args.seed),
        ]
    )


if __name__ == "__main__":
    main()
