"""Run configuration for the consolidated benchmark pipeline.

Defaults reproduce the original `benchmark_8.py` (opioid targets, ECFP +
RDKit2D descriptors, random/butina/scaffold splits, TM/RF/XGBoost, 800
TM clauses). What used to require copy-pasting a whole new script
(`benchmark_16.py` for 1600 clauses, `_GPU`/`_para` for a different TM
backend, a different `DATASET_SUBSET` for a different target) is now
just a different `BenchmarkConfig`/CLI flag -- in particular, the TM
clause count (`n_clauses`) is a plain integer parameter, not a
qualitative "8x/16x" tag.
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
# Default study design: repeated 5x5 CV (5 outer splits, each a 5-fold
# GroupKFold) with a nested 2-fold inner split for the HP search -- see
# `BenchmarkConfig.n_outer`/`n_inner`/`k_val`. Any run that deviates from
# this (fewer/more splits, or skipping the per-split HP search via
# `param_search_min_split`) must set that explicitly; there is no implicit
# "fast" default.
DEFAULT_N_OUTER = 5
DEFAULT_N_INNER = 5
DEFAULT_K_VAL = 2
DEFAULT_N_TRIALS = 25
# `None` (the default) means every one of the `n_outer` outer splits runs
# its own real nested-CV Optuna HP search -- the full repeated-5x5-CV/
# 2-fold-inner-search design above, with no shortcuts. Set this to an
# explicit outer-split index to opt into a (non-default) speed shortcut:
# splits up to and including that index each run a fresh search, and every
# split after it reuses the most recently searched params instead of
# re-searching. This must be set explicitly (e.g. via `--param-search-
# min-split`) -- it is never silently applied.
DEFAULT_PARAM_SEARCH_MIN_SPLIT = None

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
    n_clauses: int = 800
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
    # Default study design: repeated 5x5 CV (n_outer=5 repeats of an
    # n_inner=5-fold GroupKFold split) with a nested k_val=2-fold split for
    # the HP search. Passing non-default values for these is a deliberate
    # deviation and must be specified explicitly (e.g. via --n-outer/
    # --n-inner/--k-val).
    n_outer: int = DEFAULT_N_OUTER
    n_inner: int = DEFAULT_N_INNER
    k_val: int = DEFAULT_K_VAL
    n_trials: int = DEFAULT_N_TRIALS
    # None (default): every outer split runs its own real HP search --
    # nothing is skipped/reused. Set to an outer-split index to explicitly
    # opt into skipping the search (and reusing the most recent result)
    # for every split after it -- a deliberate, non-default deviation.
    param_search_min_split: int | None = DEFAULT_PARAM_SEARCH_MIN_SPLIT

    data_dir: str = "data"
    output_dir: str = "results"
    run_label: str | None = None
    resume: bool = False
    n_hp_jobs: int = 10

    @property
    def macro_out_filename(self) -> str:
        label = self.run_label or str(self.n_clauses)
        return f"{self.output_dir}/MACRO_TM_Benchmark_{label}"
