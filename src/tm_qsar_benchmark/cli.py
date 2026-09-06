"""Command-line entry point for the consolidated benchmark pipeline.

Examples
--------
Run the classic 8x-clause benchmark (auto-detects CPU/parallel/GPU backend)::

    pixi run python -m tm_qsar_benchmark.cli --clauses 8

Run the 16x-clause variant, forcing the CPU backend::

    pixi run python -m tm_qsar_benchmark.cli --clauses 16 --backend cpu

Fast smoke test (tiny search budget, one dataset, resumable)::

    pixi run python -m tm_qsar_benchmark.cli --clauses 8 \\
        --datasets opioids/MOR_cutoff6.csv --n-outer 1 --n-inner 2 \\
        --n-trials 2 --n-tm-epochs 2 --run-label smoketest --resume
"""
from __future__ import annotations

import argparse
import logging

from tm_qsar_benchmark.config import BenchmarkConfig
from tm_qsar_benchmark.hardware import VALID_BACKENDS, describe
from tm_qsar_benchmark.pipeline import run_benchmark


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tm_qsar_benchmark",
        description="Consolidated TM vs RandomForest vs XGBoost QSAR benchmark pipeline.",
    )
    parser.add_argument("--clauses", type=int, default=8, help="TM clause factor (clauses = 100 trees * factor); default 8.")
    parser.add_argument("--backend", choices=VALID_BACKENDS, default="auto", help="TM backend; 'auto' picks GPU/parallel/CPU based on detected hardware.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset CSVs under --data-dir (default: opioid MOR/DOR/KOR).")
    parser.add_argument("--data-dir", default="data", help="Root directory containing dataset CSVs.")
    parser.add_argument("--output-dir", default="results", help="Directory to write MACRO/MICRO result CSVs into.")
    parser.add_argument("--run-label", default=None, help="Suffix for output filenames (default: the clause factor, e.g. MACRO_TM_Benchmark_8).")
    parser.add_argument("--n-outer", type=int, default=None, help="Outer CV splits (default 5).")
    parser.add_argument("--n-inner", type=int, default=None, help="Inner CV folds per outer split (default 5).")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials per HP search (default 25).")
    parser.add_argument("--n-tm-epochs", type=int, default=None, help="TM training epochs (default 50).")
    parser.add_argument("--resume", action="store_true", help="Skip (dataset, group, split, fold, descriptor, model) combinations already present in the macro output file.")
    parser.add_argument("--describe-hardware", action="store_true", help="Print detected hardware/backend and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging.")
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.describe_hardware:
        print(describe(args.backend))
        return

    config = BenchmarkConfig(
        clause_factor=args.clauses,
        backend=args.backend,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_label=args.run_label,
        resume=args.resume,
    )
    if args.datasets is not None:
        config.dataset_subset = args.datasets
        config.learning_task = ["class"] * len(args.datasets)
        config.smiles_col = ["SMILES"] * len(args.datasets)
        config.prop_col = ["label"] * len(args.datasets)
    if args.n_outer is not None:
        config.n_outer = args.n_outer
    if args.n_inner is not None:
        config.n_inner = args.n_inner
    if args.n_trials is not None:
        config.n_trials = args.n_trials
    if args.n_tm_epochs is not None:
        config.n_tm_epochs = args.n_tm_epochs

    run_benchmark(config)


if __name__ == "__main__":
    main()
