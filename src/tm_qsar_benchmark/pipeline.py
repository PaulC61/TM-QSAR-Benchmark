"""The consolidated benchmark pipeline: nested group cross-validation with
Optuna hyperparameter search, comparing a Tsetlin Machine (TM, backend
auto-selected via `tm_qsar_benchmark.hardware`), Random Forest and XGBoost
on a QSAR classification dataset.

"""
from __future__ import annotations

import csv
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from useful_rdkit_utils import GroupKFoldShuffle
from xgboost import XGBClassifier

from tm_qsar_benchmark.binarizer import Binarizer as QuantileBinarizer
from tm_qsar_benchmark.config import BenchmarkConfig, resolve_param_grid
from tm_qsar_benchmark.descriptors import gen_ecfp_arr, gen_rdkit2D_arr, mol_from_smiles
from tm_qsar_benchmark.hardware import resolve_backend
from tm_qsar_benchmark.metrics import write_clf_scores, write_MICRO_clf_scores
from tm_qsar_benchmark.tm_backends import get_backend

log = logging.getLogger(__name__)

RESULTS_COLUMNS = [
    "TargetDataset", "Group", "Split", "Fold", "Descriptor", "HPSearchTime",
    "TrainTime", "InferenceTime", "TotalTime", "Params", "Epochs", "Model",
    "Dataset", "Score", "ScoreType",
]


def _load_explored_configs(macro_out_path: Path) -> set:
    """Optional resume support (config.resume=True): skip
    (dataset, group, split, fold, descriptor, model) combinations already
    present in a prior run's macro results file. Ported from the
    skip-already-explored-configs behaviour that used to exist only in the
    `_GPU`/`_para` scripts, now available uniformly for any backend."""
    if not macro_out_path.exists():
        return set()
    try:
        explored = pd.read_csv(macro_out_path, names=RESULTS_COLUMNS)
    except pd.errors.EmptyDataError:
        return set()
    config_cols = ["TargetDataset", "Group", "Split", "Fold", "Descriptor", "Model"]
    return set(explored.loc[:, config_cols].itertuples(index=False, name=None))


def _gen_descriptor(descriptor: str, train_df, test_df, fp_size: int, fp_rad: int):
    if descriptor == "ECFP":
        return (
            gen_ecfp_arr(mol_df=train_df, mol_col="mol", fp_size=fp_size, fp_radius=fp_rad, n_threads=-1),
            gen_ecfp_arr(mol_df=test_df, mol_col="mol", fp_size=fp_size, fp_radius=fp_rad, n_threads=-1),
        )
    elif descriptor == "RDKit2D":
        train_cont = gen_rdkit2D_arr(mol_df=train_df, mol_col="mol")
        test_cont = gen_rdkit2D_arr(mol_df=test_df, mol_col="mol")
        binarizer = QuantileBinarizer(resolution=10)
        binarizer.fit(train_cont)
        return binarizer.transform(train_cont), binarizer.transform(test_cont)
    raise ValueError(f"Unknown descriptor {descriptor!r}")


def _benchmark_clf_objective(trial, X_train_in, X_val_in, Y_train_in, Y_val_in, model_label, param_grid, config, tm_backend):
    clf_params = {}
    for param, (lo, hi, log_scale, is_int) in param_grid.items():
        if is_int:
            clf_params[param] = trial.suggest_int(param, lo, hi, log=log_scale)
        else:
            clf_params[param] = trial.suggest_float(param, lo, hi, log=log_scale)

    if model_label == "RandomForest":
        clf_params["n_estimators"] = config.n_trees
        clf_model = RandomForestClassifier(**clf_params)
    elif model_label == "XGBoost":
        clf_params["n_estimators"] = config.n_trees
        clf_model = XGBClassifier(**clf_params)
    else:
        clf_model = tm_backend.make_classifier(clf_params, config.n_clauses)

    if model_label != "TsetlinMachine":
        clf_model.fit(X_train_in, Y_train_in)
        Y_val_pred_prob = clf_model.predict_proba(X_val_in)[:, 1]
    else:
        for epoch in range(int(config.n_tm_epochs)):
            tm_backend.fit_clf_epoch(clf_model, X_train_in, Y_train_in)
            Y_val_pred_prob = tm_backend.predict_clf_proba(clf_model, X_val_in, config.n_clauses)

            trial.report(roc_auc_score(y_true=Y_val_in, y_score=Y_val_pred_prob), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return roc_auc_score(y_true=Y_val_in, y_score=Y_val_pred_prob)


def run_benchmark(config: BenchmarkConfig) -> None:
    """Run the full nested-CV benchmark and append melted/long-format rows
    to `config.macro_out_filename` / `config.micro_out_filename`."""
    backend_name = resolve_backend(config.backend)
    tm_backend = get_backend(backend_name)
    log.info("Using TM backend: %s (requested=%s)", backend_name, config.backend)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_path = Path(config.macro_out_filename)
    micro_path = Path(config.micro_out_filename)

    explored_configs = _load_explored_configs(macro_path) if config.resume else set()

    result_file = open(macro_path, "a", newline="")
    result_writer = csv.writer(result_file)
    micro_result_file = open(micro_path, "a", newline="")
    micro_result_writer = csv.writer(micro_result_file)

    try:
        for dataset_indx, dataset_name in enumerate(config.dataset_subset):
            task = config.learning_task[dataset_indx]
            if task != "class":
                raise NotImplementedError(
                    f"Dataset {dataset_name!r} has learning_task={task!r}, but the consolidated "
                    "pipeline currently only implements the classification training/eval path "
                    "(matching every prior benchmark_*.py script, which likewise never trained a "
                    "final regression model despite having regression HP-search code)."
                )

            dataset_df = pd.read_csv(f"{config.data_dir}/{dataset_name}")
            dataset_df["mol"] = dataset_df[config.smiles_col[dataset_indx]].apply(mol_from_smiles)
            dataset_df.dropna(subset=["mol"], inplace=True)

            for group_name, group_func in config.group_lst:
                current_group = np.array(group_func(dataset_df[config.smiles_col[dataset_indx]]))

                split_params = {descriptor: {} for descriptor in config.descriptor_set}
                for split in range(config.n_outer):
                    kf_outer = GroupKFoldShuffle(n_splits=config.n_inner, shuffle=True)
                    for fold, (train_indx, test_indx) in enumerate(kf_outer.split(dataset_df, groups=current_group)):
                        train_df = dataset_df.loc[train_indx]
                        test_df = dataset_df.loc[test_indx]

                        Y_train = np.array(train_df[config.prop_col[dataset_indx]]).flatten()
                        Y_test = np.array(test_df[config.prop_col[dataset_indx]]).flatten()

                        for descriptor in config.descriptor_set:
                            X_train, X_test = _gen_descriptor(
                                descriptor, train_df, test_df, config.fp_size, config.fp_rad
                            )

                            for model_label in config.model_labels:
                                config_tuple = (dataset_name, group_name, split, fold, descriptor, model_label)
                                if config_tuple in explored_configs:
                                    log.info("Skipping already-explored config: %s", config_tuple)
                                    continue

                                param_grid = resolve_param_grid(model_label, config.n_clauses)

                                final_params = {}
                                hp_search_time = 0.0
                                # Two-fold CV need not be re-initialized for every outer split;
                                # only refresh the HP search on later splits, reusing the most
                                # recently searched params for this descriptor otherwise.
                                #
                                # NOTE: the original scripts only ever populated `final_params`
                                # inside this "search now" branch (`if split > 3`, i.e. only the
                                # very last of 5 outer splits) and left the "reuse it on earlier
                                # splits" branch entirely commented out -- so `final_params` was
                                # actually an undefined name until split 4, meaning the script as
                                # committed would raise a `NameError` on the very first
                                # (dataset, group, split=0, fold, descriptor, model) combination it
                                # ever processed. This is fixed here by implementing the reuse
                                # logic the commented-out code was clearly aiming for (falling back
                                # to the HP grid's midpoint before any search has happened yet).
                                if split > config.param_search_min_split:
                                    cv_best_params = []
                                    kf_inner = GroupKFoldShuffle(n_splits=config.k_val, shuffle=False)
                                    for train_val_indx, val_indx in kf_inner.split(
                                        train_df, groups=current_group[train_indx]
                                    ):
                                        train_val_df = train_df.iloc[train_val_indx]
                                        val_df = train_df.iloc[val_indx]

                                        X_train_val, X_val = _gen_descriptor(
                                            descriptor, train_val_df, val_df, config.fp_size, config.fp_rad
                                        )
                                        Y_train_val = np.array(train_val_df[config.prop_col[dataset_indx]]).flatten()
                                        Y_val = np.array(val_df[config.prop_col[dataset_indx]]).flatten()

                                        study = optuna.create_study(
                                            direction="maximize",
                                            pruner=optuna.pruners.MedianPruner(
                                                n_startup_trials=5, n_warmup_steps=int(config.n_tm_epochs * 0.1)
                                            ),
                                        )
                                        hp_start = time.time()
                                        study.optimize(
                                            lambda trial: _benchmark_clf_objective(
                                                trial,
                                                X_train_in=X_train_val,
                                                X_val_in=X_val,
                                                Y_train_in=Y_train_val,
                                                Y_val_in=Y_val,
                                                model_label=model_label,
                                                param_grid=param_grid,
                                                config=config,
                                                tm_backend=tm_backend,
                                            ),
                                            n_trials=config.n_trials,
                                            n_jobs=config.n_hp_jobs if model_label != "TsetlinMachine" else 1,
                                        )
                                        hp_search_time += time.time() - hp_start
                                        cv_best_params.append(study.best_params)

                                    mean_best_params = {}
                                    for param, (_, _, _, is_int) in param_grid.items():
                                        fold_values = [cv_best_params[k][param] for k in range(config.k_val)]
                                        mean_best_params[param] = (
                                            int(statistics.mean(fold_values)) if is_int else statistics.mean(fold_values)
                                        )
                                    split_params[descriptor][split] = mean_best_params
                                    final_params = mean_best_params
                                else:
                                    if split_params[descriptor]:
                                        final_params = dict(next(reversed(split_params[descriptor].values())))
                                    else:
                                        final_params = {
                                            param: (int((lo + hi) / 2) if is_int else (lo + hi) / 2)
                                            for param, (lo, hi, _, is_int) in param_grid.items()
                                        }

                                if model_label != "TsetlinMachine":
                                    final_params["n_estimators"] = config.n_trees
                                    clf_model = (
                                        RandomForestClassifier(**final_params)
                                        if model_label == "RandomForest"
                                        else XGBClassifier(**final_params)
                                    )

                                    train_start = time.time()
                                    clf_model.fit(X_train, Y_train)
                                    train_time = time.time() - train_start

                                    inference_start = time.time()
                                    Y_train_pred_prob = clf_model.predict_proba(X_train)[:, 1]
                                    inference_time = time.time() - inference_start
                                    total_time = hp_search_time + train_time + inference_time

                                    meta_info = [
                                        dataset_name, group_name, split, fold, descriptor,
                                        hp_search_time, train_time, inference_time, total_time,
                                        final_params, config.n_tm_epochs,
                                    ]
                                    write_clf_scores(Y=Y_train, Y_pred=Y_train_pred_prob, meta_info=meta_info, model=model_label, dataset="Train", writer=result_writer)
                                    Y_test_pred_prob = clf_model.predict_proba(X_test)[:, 1]
                                    write_clf_scores(Y=Y_test, Y_pred=Y_test_pred_prob, meta_info=meta_info, model=model_label, dataset="Test", writer=result_writer)

                                    micro_meta = [dataset_name, group_name, split, fold, descriptor, config.n_tm_epochs]
                                    write_MICRO_clf_scores(Y_index=train_indx, Y=Y_train, Y_pred=Y_train_pred_prob, meta_info=micro_meta, model=model_label, dataset="Train", micro_writer=micro_result_writer)
                                    write_MICRO_clf_scores(Y_index=test_indx, Y=Y_test, Y_pred=Y_test_pred_prob, meta_info=micro_meta, model=model_label, dataset="Test", micro_writer=micro_result_writer)
                                else:
                                    clf_model = tm_backend.make_classifier(final_params, config.n_clauses)

                                    tm_train_time = 0.0
                                    tm_inference_time = 0.0
                                    for epoch in range(config.n_tm_epochs):
                                        train_start = time.time()
                                        tm_backend.fit_clf_epoch(clf_model, X_train, Y_train)
                                        tm_train_time += time.time() - train_start

                                        inference_start = time.time()
                                        Y_train_pred_prob = tm_backend.predict_clf_proba(clf_model, X_train, config.n_clauses)
                                        tm_inference_time += time.time() - inference_start
                                        tm_total_time = hp_search_time + tm_train_time + tm_inference_time

                                        meta_info = [
                                            dataset_name, group_name, split, fold, descriptor,
                                            hp_search_time, tm_train_time, tm_inference_time, tm_total_time,
                                            final_params, epoch + 1,
                                        ]
                                        write_clf_scores(Y=Y_train, Y_pred=Y_train_pred_prob, meta_info=meta_info, model=model_label, dataset="Train", writer=result_writer)

                                        Y_test_pred_prob = tm_backend.predict_clf_proba(clf_model, X_test, config.n_clauses)
                                        write_clf_scores(Y=Y_test, Y_pred=Y_test_pred_prob, meta_info=meta_info, model=model_label, dataset="Test", writer=result_writer)

                                        micro_meta = [dataset_name, group_name, split, fold, descriptor, epoch + 1]
                                        write_MICRO_clf_scores(Y_index=train_indx, Y=Y_train, Y_pred=Y_train_pred_prob, meta_info=micro_meta, model=model_label, dataset="Train", micro_writer=micro_result_writer)
                                        write_MICRO_clf_scores(Y_index=test_indx, Y=Y_test, Y_pred=Y_test_pred_prob, meta_info=micro_meta, model=model_label, dataset="Test", micro_writer=micro_result_writer)

                                result_file.flush()
                                micro_result_file.flush()
    finally:
        result_file.close()
        micro_result_file.close()
