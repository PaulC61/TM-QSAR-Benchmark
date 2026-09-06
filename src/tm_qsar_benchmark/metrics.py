"""Classification/regression metrics and the melted/long-format CSV row
writers used by every benchmark run.

The row layout written here (`write_clf_scores`, `write_MICRO_clf_scores`,
`write_reg_scores`) is intentionally preserved byte-for-byte from the
original `benchmark_*.py` scripts: downstream analysis (post-hoc-analysis.ipynb,
model_comparison.py, results/MACRO_TM_Benchmark_* / MICRO_TM_Benchmark_*)
depends on this exact "melted"/long redundant-metadata format.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)

__all__ = [
    "prc_auc_score",
    "ppv_npv_score",
    "write_clf_scores",
    "write_MICRO_clf_scores",
    "oos_r2_score",
    "write_reg_scores",
    "expit",
]


def prc_auc_score(Y, Y_pred):
    """Area under the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(Y, Y_pred)
    return auc(recall, precision)


def ppv_npv_score(Y, Y_pred):
    """Positive/negative predictive value at the ROC-optimal (Youden's J)
    probability cutoff."""
    fpr, tpr, proba = roc_curve(Y, Y_pred)
    optimal_proba_cutoff = sorted(
        list(zip(np.abs(tpr - fpr), proba)), key=lambda i: i[0], reverse=True
    )[0][1]

    hard_Y_pred = [1 if p > optimal_proba_cutoff else 0 for p in Y_pred]

    return (
        precision_score(Y, hard_Y_pred, pos_label=1),
        precision_score(Y, hard_Y_pred, pos_label=0),
    )


def write_clf_scores(Y, Y_pred, meta_info, model, dataset, writer):
    """Write one melted/long-format row per classification metric
    (ROC_AUC, PRC_AUC, PPV, NPV) to `writer` (a `csv.writer`)."""
    meta_info = meta_info + [model, dataset]

    roc_auc = roc_auc_score(Y, Y_pred, multi_class="ovr")
    writer.writerow(meta_info + [roc_auc, "ROC_AUC"])

    prc_auc = prc_auc_score(Y, Y_pred)
    writer.writerow(meta_info + [prc_auc, "PRC_AUC"])

    ppv, npv = ppv_npv_score(Y, Y_pred)
    writer.writerow(meta_info + [ppv, "PPV"])
    writer.writerow(meta_info + [npv, "NPV"])

    return None


def write_MICRO_clf_scores(Y_index, Y, Y_pred, meta_info, model, dataset, micro_writer):
    """Write one melted/long-format row per individual sample prediction."""
    meta_info = meta_info + [model, dataset]
    for sample_indx in range(len(Y_index)):
        sample_pred = meta_info + [Y_index[sample_indx], Y[sample_indx], Y_pred[sample_indx]]
        micro_writer.writerow(sample_pred)
    return None


def oos_r2_score(Y, Y_pred, Y_dummy_pred):
    """Out-of-sample R2 relative to a dummy (mean) predictor."""
    from sklearn.metrics import mean_squared_error

    mse_pred = mean_squared_error(Y, Y_pred)
    mse_dummy = mean_squared_error(Y, Y_dummy_pred)
    return 1 - (mse_pred / mse_dummy)


def write_reg_scores(Y, Y_pred, Y_dummy_pred, meta_info, model, dataset, writer):
    """Write one melted/long-format row per regression metric (RMSE, MAE,
    Pearson_R, R2) to `writer` (a `csv.writer`)."""
    meta_info = meta_info + [model, dataset]

    rmse = root_mean_squared_error(Y, Y_pred)
    writer.writerow(meta_info + [rmse, "RMSE"])

    mae = mean_absolute_error(Y, Y_pred)
    writer.writerow(meta_info + [mae, "MAE"])

    pearson_r = pearsonr(Y, Y_pred)
    writer.writerow(meta_info + [pearson_r, "Pearson_R"])

    oos_r2 = oos_r2_score(Y, Y_pred, Y_dummy_pred)
    writer.writerow(meta_info + [oos_r2, "R2"])

    return None
