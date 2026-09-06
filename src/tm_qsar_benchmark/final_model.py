"""Generalised "final model" training: given just a dataset CSV, its SMILES
column and its property column, produce a ready-to-use TM classifier.

The only inputs that should matter for a new target are the dataset
location and its column names -- everything else (which hyperparameters,
which descriptor, how many training epochs) is *derived*, not guessed:

* If this dataset has already been through the benchmarking pipeline at the
  requested clause count (i.e. there's an existing performance distribution
  for it in `results/`), reuse that: aggregate the CV-tuned descriptor/
  hyperparameters/epoch count from those results and refit on 100% of the
  data, exactly like the CYP3A4/CYP2D6 final models.
* If it hasn't (a genuinely new/unseen target), first run the *same*
  repeated 5x5 nested-CV benchmark (`tm_qsar_benchmark.pipeline.run_benchmark`)
  used for every other target in this project, to produce that performance
  distribution -- then aggregate it and refit, same as above.

Either way you end up with a model trained the same principled way: CV
first (to know which hyperparameters/descriptor to trust), full refit
second (to use every available sample in the final artifact).
"""
from __future__ import annotations

import ast
import logging
import pickle
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from tm_qsar_benchmark.binarizer import Binarizer
from tm_qsar_benchmark.config import BenchmarkConfig
from tm_qsar_benchmark.descriptors import gen_ecfp_arr, gen_rdkit2D_arr, mol_from_smiles
from tm_qsar_benchmark.hardware import resolve_backend
from tm_qsar_benchmark.pipeline import RESULTS_COLUMNS, run_benchmark
from tm_qsar_benchmark.tm_backends import get_backend

log = logging.getLogger(__name__)

DEFAULT_N_CLAUSES = 1600
FP_SIZE = 2048
FP_RAD = 2
RDKIT2D_BINARIZER_RESOLUTION = 10

# Every prior benchmark run (including the original paper's) accumulated its
# results into one of these two files per clause count -- checked (in this
# order) before deciding a dataset needs a fresh CV run. A dataset benchmarked
# under a custom --run-label won't be found here; pass `force_benchmark=True`
# (or point `results_dir` at the right file) if that's your situation.
_LEGACY_ALL_CLAUSES_FILE = "MACRO_TM_Benchmark_ALL_NClauses"


def _read_macro_file(path: Path) -> pd.DataFrame:
    """Read a MACRO_TM_Benchmark_* file, whether it's this pipeline's plain
    header-less melted CSV (`RESULTS_COLUMNS`) or the original paper's
    combined file (which has a header plus extra `ClauseFactor`/`N_Clauses`/
    index columns)."""
    with open(path) as f:
        first_line = f.readline()
    if first_line.startswith(",TargetDataset") or first_line.startswith("TargetDataset"):
        df = pd.read_csv(path)
        return df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return pd.read_csv(path, names=RESULTS_COLUMNS)


def find_existing_cv_results(dataset_name: str, n_clauses: int, results_dir: str = "results") -> pd.DataFrame | None:
    """Look for an already-run performance distribution for `dataset_name`
    (e.g. `"opioids/CYP3A4_cutoff6.csv"`) at `n_clauses`, across the legacy
    combined results file and this pipeline's own per-clause-count output
    file. Returns the matching TsetlinMachine rows, or None if nothing was
    found (i.e. this target needs a fresh benchmark run)."""
    candidates = [
        Path(results_dir) / _LEGACY_ALL_CLAUSES_FILE,
        Path(results_dir) / f"MACRO_TM_Benchmark_{n_clauses}",
    ]
    frames = []
    for path in candidates:
        if not path.exists():
            continue
        df = _read_macro_file(path)
        df = df[(df["TargetDataset"] == dataset_name) & (df["Model"] == "TsetlinMachine")]
        if "N_Clauses" in df.columns:
            df = df[df["N_Clauses"] == n_clauses]
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def run_full_cv_benchmark(
    dataset_csv: str,
    smiles_col: str,
    prop_col: str,
    n_clauses: int,
    backend: str = "auto",
    data_dir: str = "data",
    results_dir: str = "results",
    cv_overrides: dict | None = None,
) -> None:
    """Run the standard repeated 5x5 nested-CV benchmark
    (`tm_qsar_benchmark.pipeline.run_benchmark`) for a single, previously
    unseen dataset -- the same pipeline/defaults (3 descriptors x 3 grouping
    strategies x TM/RandomForest/XGBoost) used for every other target in this
    project. Results are appended to `results/MACRO_TM_Benchmark_{n_clauses}`
    alongside whatever other targets already share that clause count."""
    config = BenchmarkConfig(
        n_clauses=n_clauses,
        backend=backend,
        dataset_subset=[dataset_csv],
        learning_task=["class"],
        smiles_col=[smiles_col],
        prop_col=[prop_col],
        data_dir=data_dir,
        output_dir=results_dir,
        resume=True,
    )
    for field, value in (cv_overrides or {}).items():
        setattr(config, field, value)

    log.info(
        "No existing CV results for %s at n_clauses=%d -- running the full benchmark "
        "(n_outer=%d, n_inner=%d, n_trials=%d, n_tm_epochs=%d) to build one.",
        dataset_csv, n_clauses, config.n_outer, config.n_inner, config.n_trials, config.n_tm_epochs,
    )
    run_benchmark(config)


def _parse_params(value) -> dict:
    return ast.literal_eval(value) if isinstance(value, str) else dict(value)


def aggregate_best_config(tm_results: pd.DataFrame) -> dict:
    """Reduce a dataset's TsetlinMachine CV rows to one final training
    config: the descriptor with the best mean Test ROC_AUC, that
    descriptor's mean CV-tuned (T, s) across folds, and the training epoch
    with the highest mean Test ROC_AUC across folds. This is exactly the
    aggregation used to build the CYP3A4/CYP2D6 final models, generalised to
    any dataset's results."""
    test_df = tm_results[(tm_results["Dataset"] == "Test") & (tm_results["ScoreType"] == "ROC_AUC")]
    if test_df.empty:
        raise ValueError("No Test/ROC_AUC rows to aggregate a final training config from.")

    descriptor = test_df.groupby("Descriptor")["Score"].mean().idxmax()

    desc_df = tm_results[tm_results["Descriptor"] == descriptor]
    per_fold = desc_df.drop_duplicates(subset=["Group", "Split", "Fold"])
    params = per_fold["Params"].apply(_parse_params)
    mean_T = statistics.mean(p["T"] for p in params)
    mean_s = statistics.mean(p["s"] for p in params)

    desc_test_df = test_df[test_df["Descriptor"] == descriptor]
    epoch_means = desc_test_df.groupby("Epochs")["Score"].mean()
    best_epoch = int(epoch_means.idxmax())

    return {
        "descriptor": descriptor,
        "params": {"T": int(round(mean_T)), "s": round(float(mean_s), 4)},
        "n_epochs": best_epoch,
        "cv_mean_test_roc_auc": float(epoch_means.max()),
        "n_cv_folds": len(per_fold),
    }


def _featurize(descriptor: str, mol_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    if descriptor == "ECFP":
        X = gen_ecfp_arr(mol_df=mol_df, mol_col="mol", fp_size=FP_SIZE, fp_radius=FP_RAD, n_threads=-1)
        return X, {"descriptor": "ECFP", "fp_size": FP_SIZE, "fp_rad": FP_RAD}
    elif descriptor == "RDKit2D":
        X_cont = gen_rdkit2D_arr(mol_df=mol_df, mol_col="mol")
        binarizer = Binarizer(resolution=RDKIT2D_BINARIZER_RESOLUTION)
        X = binarizer.fit_transform(X_cont)
        return X, {"descriptor": "RDKit2D", "binarizer": binarizer}
    raise ValueError(f"Unknown descriptor {descriptor!r}")


def train_final_model(
    dataset_csv: str,
    smiles_col: str = "SMILES",
    prop_col: str = "label",
    target_name: str | None = None,
    n_clauses: int = DEFAULT_N_CLAUSES,
    backend: str = "auto",
    data_dir: str = "data",
    results_dir: str = "results",
    output_dir: str = "models",
    force_benchmark: bool = False,
    cv_overrides: dict | None = None,
) -> Path:
    """Train and pickle a final (100%-of-data refit) TM classifier for
    `dataset_csv`, deriving its descriptor/hyperparameters/epoch count from
    that dataset's CV performance distribution -- running the full
    benchmark first if one doesn't already exist. Returns the output pickle
    path."""
    target_name = target_name or Path(dataset_csv).stem

    existing = None if force_benchmark else find_existing_cv_results(dataset_csv, n_clauses, results_dir)
    if existing is None:
        run_full_cv_benchmark(
            dataset_csv, smiles_col, prop_col, n_clauses,
            backend=backend, data_dir=data_dir, results_dir=results_dir, cv_overrides=cv_overrides,
        )
        existing = find_existing_cv_results(dataset_csv, n_clauses, results_dir)
        if existing is None:
            raise RuntimeError(
                f"Benchmark run completed but produced no TsetlinMachine results for {dataset_csv!r} "
                f"at n_clauses={n_clauses} -- check the dataset/column names."
            )
    else:
        log.info(
            "Found existing CV results for %s at n_clauses=%d (%d rows) -- reusing them instead of "
            "re-running the benchmark.", dataset_csv, n_clauses, len(existing),
        )

    agg = aggregate_best_config(existing)

    df = pd.read_csv(f"{data_dir}/{dataset_csv}")
    df["mol"] = df[smiles_col].apply(mol_from_smiles)
    n_dropped = df["mol"].isna().sum()
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    if n_dropped:
        log.warning("%s: dropped %d rows with unparsable SMILES", target_name, n_dropped)

    Y = np.array(df[prop_col]).flatten()
    X, featurizer_meta = _featurize(agg["descriptor"], df)

    backend_name = resolve_backend(backend)
    tm_backend = get_backend(backend_name)
    clf = tm_backend.make_classifier(agg["params"], n_clauses)
    for _ in range(agg["n_epochs"]):
        tm_backend.fit_clf_epoch(clf, X, Y)
    log.info(
        "%s: trained %d epochs on %d molecules (%s backend, %s descriptor, params=%s)",
        target_name, agg["n_epochs"], len(df), backend_name, agg["descriptor"], agg["params"],
    )

    payload = {
        "target": target_name,
        "model": clf,
        "backend": backend_name,
        "n_clauses": n_clauses,
        "n_epochs": agg["n_epochs"],
        "params": agg["params"],
        "n_train_molecules": len(df),
        "cv_mean_test_roc_auc": agg["cv_mean_test_roc_auc"],
        "n_cv_folds": agg["n_cv_folds"],
        "source_dataset": dataset_csv,
        "smiles_col": smiles_col,
        "prop_col": prop_col,
        **featurizer_meta,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"TM_{target_name}_{n_clauses}clauses.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return out_path
