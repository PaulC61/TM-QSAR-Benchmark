#!/usr/bin/env python
"""Train and pickle "final" (full-dataset, no held-out split) TM classifiers
for CYP3A4 and CYP2D6, at 1600 clauses, for downstream use outside this repo
(e.g. handing a ready-to-use model to a colleague).

Hyperparameters (T, s) and the number of training epochs are not guessed:
they're the CV-tuned values already produced by this repo's own benchmark
runs, aggregated from `results/MACRO_TM_Benchmark_ALL_NClauses`
(TsetlinMachine, N_Clauses=1600, Test/ROC_AUC rows) --
mean(T)/mean(s) across every outer-CV fold's inner-search result for the
descriptor that scored best on average, and the training epoch with the
highest mean Test ROC_AUC across folds. That means each model here is
trained with the same clause count/backend/descriptor pipeline as the rest
of the benchmark, just refit on 100% of the data instead of a CV split
(standard practice once CV has told you which hyperparameters to trust).

Usage
-----
    pixi run python scripts/train_final_models.py

Writes one pickle per target to `models/`:
    models/TM_CYP3A4_1600clauses.pkl
    models/TM_CYP2D6_1600clauses.pkl

Each pickle contains a dict with the fitted TM classifier plus everything
needed to featurize new SMILES the same way at inference time (see the
module docstring of `tm_qsar_benchmark.descriptors` for the featurizer
functions themselves).
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from tm_qsar_benchmark.binarizer import Binarizer
from tm_qsar_benchmark.descriptors import gen_ecfp_arr, gen_rdkit2D_arr, mol_from_smiles
from tm_qsar_benchmark.hardware import describe, resolve_backend
from tm_qsar_benchmark.tm_backends import get_backend

log = logging.getLogger(__name__)

N_CLAUSES = 1600
FP_SIZE = 2048
FP_RAD = 2
RDKIT2D_BINARIZER_RESOLUTION = 10

# CV-tuned hyperparameters, aggregated from results/MACRO_TM_Benchmark_ALL_NClauses
# (TsetlinMachine, N_Clauses=1600, Test/ROC_AUC rows) -- see module docstring.
TARGET_CONFIGS = {
    "CYP3A4": {
        "dataset_csv": "data/opioids/CYP3A4_cutoff6.csv",
        "descriptor": "ECFP",
        "params": {"T": 6626, "s": 4.32},
        "n_epochs": 25,
        "cv_mean_test_roc_auc": 0.887,
    },
    "CYP2D6": {
        "dataset_csv": "data/opioids/CYP2D6_cutoff6.csv",
        "descriptor": "RDKit2D",
        "params": {"T": 6800, "s": 3.05},
        "n_epochs": 33,
        "cv_mean_test_roc_auc": 0.631,
    },
}


def _featurize(descriptor: str, mol_df):
    if descriptor == "ECFP":
        X = gen_ecfp_arr(mol_df=mol_df, mol_col="mol", fp_size=FP_SIZE, fp_radius=FP_RAD, n_threads=-1)
        return X, {"descriptor": "ECFP", "fp_size": FP_SIZE, "fp_rad": FP_RAD}
    elif descriptor == "RDKit2D":
        X_cont = gen_rdkit2D_arr(mol_df=mol_df, mol_col="mol")
        binarizer = Binarizer(resolution=RDKIT2D_BINARIZER_RESOLUTION)
        X = binarizer.fit_transform(X_cont)
        return X, {"descriptor": "RDKit2D", "binarizer": binarizer}
    raise ValueError(f"Unknown descriptor {descriptor!r}")


def train_final_model(target: str, backend_name: str, output_dir: Path) -> Path:
    import pandas as pd

    cfg = TARGET_CONFIGS[target]
    df = pd.read_csv(cfg["dataset_csv"])
    df["mol"] = df["SMILES"].apply(mol_from_smiles)
    n_dropped = df["mol"].isna().sum()
    df = df.dropna(subset=["mol"]).reset_index(drop=True)
    if n_dropped:
        log.warning("%s: dropped %d rows with unparsable SMILES", target, n_dropped)

    Y = np.array(df["label"]).flatten()
    X, featurizer_meta = _featurize(cfg["descriptor"], df)

    backend = get_backend(backend_name)
    clf = backend.make_classifier(cfg["params"], N_CLAUSES)
    for epoch in range(cfg["n_epochs"]):
        backend.fit_clf_epoch(clf, X, Y)
    log.info("%s: trained %d epochs on %d molecules (%s backend)", target, cfg["n_epochs"], len(df), backend_name)

    payload = {
        "target": target,
        "model": clf,
        "backend": backend_name,
        "n_clauses": N_CLAUSES,
        "n_epochs": cfg["n_epochs"],
        "params": cfg["params"],
        "n_train_molecules": len(df),
        "cv_mean_test_roc_auc": cfg["cv_mean_test_roc_auc"],
        "source_dataset": cfg["dataset_csv"],
        **featurizer_meta,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"TM_{target}_{N_CLAUSES}clauses.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", choices=list(TARGET_CONFIGS), default=list(TARGET_CONFIGS))
    parser.add_argument("--backend", default="auto", help="TM backend; 'auto' picks GPU/parallel/CPU based on detected hardware.")
    parser.add_argument("--output-dir", default="models", help="Directory to write pickle files into (default: models/).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    backend_name = resolve_backend(args.backend)
    print(describe(args.backend))

    for target in args.targets:
        out_path = train_final_model(target, backend_name, Path(args.output_dir))
        print(f"Saved {target} final model -> {out_path}")


if __name__ == "__main__":
    main()
