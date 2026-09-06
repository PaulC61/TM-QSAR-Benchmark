"""Run configuration for the consolidated benchmark pipeline.

Defaults reproduce the original `benchmark_8.py` (opioid targets, ECFP +
RDKit2D descriptors, random/butina/scaffold splits, TM/RF/XGBoost). What
used to require copy-pasting a whole new script (`benchmark_16.py` for
`C_FACTOR=16`, `_GPU`/`_para` for a different TM backend, a different
`DATASET_SUBSET` for a different target) is now just a different
`BenchmarkConfig`/CLI flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import useful_rdkit_utils as uru

# Default dataset characterization (opioid targets, from data/opioids/).
DEFAULT_DATASET_SUBSET = [
    "opioids/MOR_cutoff6.csv",
    "opioids/DOR_cutoff6.csv",
    "opioids/KOR_cutoff6.csv",
]
DEFAULT_LEARNING_TASK = ["class", "class", "class"]
DEFAULT_SMILES_COL = ["SMILES", "SMILES", "SMILES"]
# NOTE: the original scripts had a missing comma here ("label"\n"label"),
# which silently concatenated to "labellabel" and left only 2 usable
# entries for 3 datasets. Fixed as part of the consolidation.
DEFAULT_PROP_COL = ["label", "label", "label"]

DEFAULT_GROUP_LST = [
    ("random", uru.get_random_clusters),
    ("butina", uru.get_butina_clusters),
    ("scaffold", uru.get_bemis_murcko_clusters),
]

DEFAULT_DESCRIPTOR_SET = ["ECFP", "RDKit2D"]
DEFAULT_MODEL_LABELS = ["TsetlinMachine", "RandomForest", "XGBoost"]

DEFAULT_FP_SIZE = 2048
DEFAULT_FP_RAD = 2

DEFAULT_N_TREES = 100
DEFAULT_N_TM_EPOCHS = 50
DEFAULT_N_OUTER = 5
DEFAULT_N_INNER = 5
DEFAULT_K_VAL = 2
DEFAULT_N_TRIALS = 25
# Outer splits beyond this index trigger a fresh inner-CV hyperparameter
# search (kept from the original "two-fold CV does not need re-initializing
# for multiple different splits" behaviour).
DEFAULT_PARAM_SEARCH_MIN_SPLIT = 3

# start, end, log, is_int -- by model (TsetlinMachine, RandomForest, XGBoost).
# The TM "T" bound scales with the run's clause count, so it's filled in by
# `resolve_param_grid` rather than hardcoded here.
DEFAULT_PARAM_GRIDS = {
    "TsetlinMachine": {
        "s": (1, 7, False, False),
    },
    "RandomForest": {
        "max_depth": (10, 100, False, True),
        "ccp_alpha": (0.001, 1.0, True, False),
    },
    "XGBoost": {
        "max_depth": (5, 20, False, True),
        "learning_rate": (0.01, 1.0, True, False),
        "min_child_weight": (1, 10, True, False),
        "gamma": (0.01, 1.0, True, False),
        "subsample": (0.01, 1.0, True, False),
        "colsample_bytree": (0.01, 1.0, True, False),
        "colsample_bynode": (0.01, 1.0, True, False),
        "reg_alpha": (0.001, 1.0, True, False),
        "reg_lambda": (0.001, 1.0, True, False),
    },
}


def resolve_param_grid(model_label: str, n_clauses: int) -> dict:
    """Resolve a model's HP search grid for this run's clause count. Only
    the TsetlinMachine grid depends on `n_clauses` (its `T` threshold search
    range is defined relative to the number of clauses, as in the original
    scripts)."""
    grid = dict(DEFAULT_PARAM_GRIDS[model_label])
    if model_label == "TsetlinMachine":
        grid["T"] = (1 * n_clauses, 10 * n_clauses, False, True)
    return grid


@dataclass
class BenchmarkConfig:
    clause_factor: int = 8
    backend: str = "auto"

    dataset_subset: list = field(default_factory=lambda: list(DEFAULT_DATASET_SUBSET))
    learning_task: list = field(default_factory=lambda: list(DEFAULT_LEARNING_TASK))
    smiles_col: list = field(default_factory=lambda: list(DEFAULT_SMILES_COL))
    prop_col: list = field(default_factory=lambda: list(DEFAULT_PROP_COL))
    group_lst: list = field(default_factory=lambda: list(DEFAULT_GROUP_LST))
    descriptor_set: list = field(default_factory=lambda: list(DEFAULT_DESCRIPTOR_SET))
    model_labels: list = field(default_factory=lambda: list(DEFAULT_MODEL_LABELS))

    fp_size: int = DEFAULT_FP_SIZE
    fp_rad: int = DEFAULT_FP_RAD

    n_trees: int = DEFAULT_N_TREES
    n_tm_epochs: int = DEFAULT_N_TM_EPOCHS
    n_outer: int = DEFAULT_N_OUTER
    n_inner: int = DEFAULT_N_INNER
    k_val: int = DEFAULT_K_VAL
    n_trials: int = DEFAULT_N_TRIALS
    param_search_min_split: int = DEFAULT_PARAM_SEARCH_MIN_SPLIT

    data_dir: str = "data"
    output_dir: str = "results"
    run_label: str | None = None
    resume: bool = False
    n_hp_jobs: int = 10

    @property
    def n_clauses(self) -> int:
        return self.n_trees * self.clause_factor

    @property
    def macro_out_filename(self) -> str:
        label = self.run_label or str(self.clause_factor)
        return f"{self.output_dir}/MACRO_TM_Benchmark_{label}"

    @property
    def micro_out_filename(self) -> str:
        label = self.run_label or str(self.clause_factor)
        return f"{self.output_dir}/MICRO_TM_Benchmark_{label}"
