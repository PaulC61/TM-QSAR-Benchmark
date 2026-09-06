#!/usr/bin/env python
"""CLI for training a "final" (100%-of-data refit) TM classifier for any
QSAR classification dataset -- see `tm_qsar_benchmark.final_model` for the
underlying logic.

The only required inputs are the dataset location and its SMILES/property
column names. What happens next depends on whether this dataset already has
a CV performance distribution in `results/`:

* Already benchmarked (e.g. CYP3A4/CYP2D6/MOR/DOR/KOR at the requested
  clause count): reuse those CV results directly -- aggregate the
  best-performing descriptor/hyperparameters/epoch count and refit on all
  the data. Fast (no CV search).
* Never seen before: first run the same repeated 5x5 nested-CV benchmark
  used for every other target in this project (`pixi run benchmark`'s
  pipeline) to build that performance distribution, then aggregate and
  refit exactly as above. This can take a long time (it's a full
  benchmark run) -- use --n-outer/--n-inner/--n-trials/--n-tm-epochs to
  shrink it for a quick/experimental pass.

Examples
--------
Reuse existing CV results for a previously benchmarked target::

    pixi run python scripts/train_final_models.py \\
        --dataset opioids/CYP3A4_cutoff6.csv --n-clauses 1600

Train a final model for a brand new target (SMILES/label columns named
differently), running the full CV benchmark first since it's unseen::

    pixi run python scripts/train_final_models.py \\
        --dataset my_new_target.csv --smiles-col Structure --prop-col Active \\
        --target-name MyNewTarget --n-clauses 1600

Force a fresh CV run even if results already exist (e.g. after changing the
dataset), with a shrunk search budget for a quick pass::

    pixi run python scripts/train_final_models.py \\
        --dataset opioids/CYP3A4_cutoff6.csv --n-clauses 1600 \\
        --force-benchmark --n-outer 1 --n-inner 2 --n-trials 2 --n-tm-epochs 2
"""
from __future__ import annotations

import argparse
import logging

from tm_qsar_benchmark.final_model import train_final_model
from tm_qsar_benchmark.hardware import VALID_BACKENDS, describe


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train_final_models", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset CSV path relative to --data-dir, e.g. opioids/CYP3A4_cutoff6.csv.")
    parser.add_argument("--smiles-col", default="SMILES", help="SMILES column name in the dataset CSV (default: SMILES).")
    parser.add_argument("--prop-col", default="label", help="Classification label column name in the dataset CSV (default: label).")
    parser.add_argument("--target-name", default=None, help="Name used in the output filename/pickle metadata (default: the dataset filename's stem).")
    parser.add_argument("--n-clauses", type=int, default=1600, help="Number of TM clauses (default: 1600).")
    parser.add_argument("--backend", choices=VALID_BACKENDS, default="auto", help="TM backend; 'auto' picks GPU/parallel/CPU based on detected hardware.")
    parser.add_argument("--data-dir", default="data", help="Root directory containing dataset CSVs (default: data).")
    parser.add_argument("--results-dir", default="results", help="Directory to look for/write CV benchmark results in (default: results).")
    parser.add_argument("--output-dir", default="models", help="Directory to write the final model pickle into (default: models).")
    parser.add_argument("--force-benchmark", action="store_true", help="Run the full CV benchmark even if results already exist for this dataset/clause count.")
    parser.add_argument("--n-outer", type=int, default=None, help="Outer CV splits for a fresh benchmark run (default 5).")
    parser.add_argument("--n-inner", type=int, default=None, help="Inner CV folds per outer split for a fresh benchmark run (default 5).")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials per HP search for a fresh benchmark run (default 25).")
    parser.add_argument("--n-tm-epochs", type=int, default=None, help="TM training epochs for a fresh benchmark run (default 50).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    print(describe(args.backend))

    cv_overrides = {
        field: value
        for field, value in (
            ("n_outer", args.n_outer),
            ("n_inner", args.n_inner),
            ("n_trials", args.n_trials),
            ("n_tm_epochs", args.n_tm_epochs),
        )
        if value is not None
    }

    out_path = train_final_model(
        dataset_csv=args.dataset,
        smiles_col=args.smiles_col,
        prop_col=args.prop_col,
        target_name=args.target_name,
        n_clauses=args.n_clauses,
        backend=args.backend,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        force_benchmark=args.force_benchmark,
        cv_overrides=cv_overrides or None,
    )
    print(f"Saved final model -> {out_path}")


if __name__ == "__main__":
    main()
