"""Minimal smoke tests for the consolidated benchmark pipeline.

These exercise the hardware/backend resolution and a tiny end-to-end
benchmark run (1 dataset, 1 outer split, 2 inner folds, 2 HP trials, 2 TM
epochs) so `pixi run test` catches import/wiring regressions without
requiring a GPU or a multi-hour run.
"""
from __future__ import annotations

import csv

import pandas as pd
import pytest

from tm_qsar_benchmark.config import BenchmarkConfig
from tm_qsar_benchmark.hardware import BACKEND_CPU, BACKEND_GPU, BACKEND_PARALLEL, resolve_backend
from tm_qsar_benchmark.pipeline import RESULTS_COLUMNS, run_benchmark


def test_resolve_backend_explicit_choices_pass_through():
    assert resolve_backend(BACKEND_CPU) == BACKEND_CPU
    assert resolve_backend(BACKEND_PARALLEL) == BACKEND_PARALLEL
    assert resolve_backend(BACKEND_GPU) == BACKEND_GPU


def test_resolve_backend_auto_returns_valid_backend():
    resolved = resolve_backend("auto")
    assert resolved in (BACKEND_CPU, BACKEND_PARALLEL, BACKEND_GPU)


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_backend("not-a-real-backend")


@pytest.mark.slow
def test_smoke_benchmark_run_produces_melted_csv(tmp_path):
    """Fast end-to-end run: 1 dataset x 1 outer split x 2 inner folds x
    2 optuna trials x 2 TM epochs, forced onto the CPU backend so this test
    doesn't depend on GPU hardware being present."""
    config = BenchmarkConfig(
        n_clauses=800,
        backend=BACKEND_CPU,
        dataset_subset=["opioids/MOR_cutoff6.csv"],
        learning_task=["class"],
        smiles_col=["SMILES"],
        prop_col=["label"],
        n_outer=1,
        n_inner=2,
        n_trials=2,
        n_tm_epochs=2,
        output_dir=str(tmp_path),
        run_label="pytest_smoketest",
    )

    run_benchmark(config)

    with open(config.macro_out_filename, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) > 0
    assert len(rows[0]) == len(RESULTS_COLUMNS)

    macro_df = pd.read_csv(config.macro_out_filename, names=RESULTS_COLUMNS)
    assert set(macro_df["Model"]) <= {"TsetlinMachine", "RandomForest", "XGBoost"}
    assert set(macro_df["ScoreType"]) <= {"ROC_AUC", "PRC_AUC", "PPV", "NPV"}
