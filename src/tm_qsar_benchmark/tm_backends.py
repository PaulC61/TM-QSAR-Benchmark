"""Backend adapters that hide the API differences between the three Tsetlin
Machine implementations this project has historically used on different
hardware:

* `cpu`      - `tmu.models.classification.coalesced_classifier.TMCoalescedClassifier`
               / `tmu.models.regression.vanilla_regressor.TMRegressor`
               (single/multi-threaded CPU, always available).
* `parallel` - `pyTsetlinMachineParallel.tm.MultiClassTsetlinMachine`
               (OpenMP multi-core CPU).
* `gpu`      - `PyTsetlinMachineCUDA.tm.MultiClassTsetlinMachine`
               (CUDA GPU, e.g. H100/H200 servers).

`parallel` and `gpu` share the exact same `MultiClassTsetlinMachine` API
(`fit(X, Y, epochs=1, incremental=True)`, `.transform(X, inverted=False)`,
`.get_state()`, `.T`) so they're implemented once as `_ArrayTMBackend` and
only differ in which package they import the model class from. This is also
where a real bug from the old `benchmark_8_GPU.py` copy is fixed: its
HP-search objective computed `Y_val_css` but then referenced the
similarly-named but undefined `Y_val_ccs`, a copy-paste artifact from the
`_para` script it was cloned from that never actually ran correctly.
"""
from __future__ import annotations

import importlib
from typing import Protocol

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit

from tm_qsar_benchmark.hardware import BACKEND_CPU, BACKEND_GPU, BACKEND_PARALLEL


class TMBackend(Protocol):
    """Common interface the benchmark pipeline drives every TM backend
    through, so the CV/training loop itself stays backend-agnostic."""

    name: str

    def make_classifier(self, params: dict, n_clauses: int): ...

    def make_regressor(self, params: dict, n_clauses: int): ...

    def fit_clf_epoch(self, model, X, Y) -> None: ...

    def predict_clf_proba(self, model, X, n_clauses: int) -> np.ndarray:
        """Positive-class probability, shape (n_samples,)."""
        ...

    def fit_reg_epoch(self, model, X, Y) -> None: ...

    def predict_reg(self, model, X) -> np.ndarray: ...


def _cust_threshold(i):
    j = (i % 2) * (-1)
    return 1 if (j > -1) else j


def _parallel_tm_ccs(tm, X, n_classes=2, n_clauses=1000, n_jobs=22):
    """Class-clause-sum reducer for the array-based (`parallel`/`gpu`) TM
    implementations, ported unchanged from `benchmark_8_para.py`."""
    active_clauses = tm.transform(X, inverted=False)
    weight_state = tm.get_state()
    mask = Parallel(n_jobs=n_jobs)(delayed(_cust_threshold)(i) for i in np.arange(n_clauses))

    ccs = []
    for i_class in range(n_classes):
        clause_class_weights = weight_state[i_class][0] * mask
        active_clause_class_weights = (
            clause_class_weights * active_clauses[:, (i_class * n_clauses):(i_class + 1) * n_clauses]
        )
        class_clause_sum = active_clause_class_weights.sum(axis=1)
        ccs.append(class_clause_sum)
    return np.array(ccs)


class TMCpuBackend:
    """`tmu` (single/multi-threaded CPU) backend -- the always-available
    default when no faster backend is installed/detected."""

    name = BACKEND_CPU

    def make_classifier(self, params: dict, n_clauses: int):
        from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

        clf_params = dict(params)
        clf_params["number_of_clauses"] = n_clauses
        clf_params["weighted_clauses"] = True
        return TMCoalescedClassifier(**clf_params)

    def make_regressor(self, params: dict, n_clauses: int):
        from tmu.models.regression.vanilla_regressor import TMRegressor

        reg_params = dict(params)
        reg_params["number_of_clauses"] = n_clauses
        reg_params["weighted_clauses"] = True
        return TMRegressor(**reg_params)

    def fit_clf_epoch(self, model, X, Y) -> None:
        model.fit(X, Y, incremental=True)

    def predict_clf_proba(self, model, X, n_clauses: int) -> np.ndarray:
        _, class_sums = model.predict(X, return_class_sums=True)
        return expit(class_sums / model.T)[:, 1]

    def fit_reg_epoch(self, model, X, Y) -> None:
        model.fit(X, Y, epochs=1, incremental=True)

    def predict_reg(self, model, X) -> np.ndarray:
        return model.predict(X)


class _ArrayTMBackend:
    """Shared implementation for `parallel` (`pyTsetlinMachineParallel`) and
    `gpu` (`PyTsetlinMachineCUDA`), which expose an identical
    `MultiClassTsetlinMachine` API."""

    name: str
    _module_name: str
    _n_jobs: int

    def _model_cls(self):
        module = importlib.import_module(f"{self._module_name}.tm")
        return module.MultiClassTsetlinMachine

    def make_classifier(self, params: dict, n_clauses: int):
        clf_params = dict(params)
        clf_params["number_of_clauses"] = n_clauses
        clf_params["weighted_clauses"] = True
        return self._model_cls()(**clf_params)

    def make_regressor(self, params: dict, n_clauses: int):
        # Neither pyTsetlinMachineParallel nor PyTsetlinMachineCUDA ship a
        # dedicated regressor; regression runs use the CPU (`tmu`) backend.
        raise NotImplementedError(
            f"{self.name} backend has no regression model; use the cpu backend for regression tasks."
        )

    def fit_clf_epoch(self, model, X, Y) -> None:
        model.fit(X, Y, epochs=1, incremental=True)

    def predict_clf_proba(self, model, X, n_clauses: int) -> np.ndarray:
        ccs = _parallel_tm_ccs(tm=model, X=X, n_clauses=n_clauses, n_jobs=self._n_jobs)
        return expit(ccs / model.T)[1]

    def fit_reg_epoch(self, model, X, Y) -> None:
        raise NotImplementedError(f"{self.name} backend has no regression model.")

    def predict_reg(self, model, X) -> np.ndarray:
        raise NotImplementedError(f"{self.name} backend has no regression model.")


class TMParallelBackend(_ArrayTMBackend):
    """Multi-core CPU backend via `pyTsetlinMachineParallel` (OpenMP)."""

    name = BACKEND_PARALLEL
    _module_name = "pyTsetlinMachineParallel"
    _n_jobs = 22


class TMGpuBackend(_ArrayTMBackend):
    """CUDA GPU backend via `PyTsetlinMachineCUDA`."""

    name = BACKEND_GPU
    _module_name = "PyTsetlinMachineCUDA"
    _n_jobs = 22


_BACKENDS = {
    BACKEND_CPU: TMCpuBackend,
    BACKEND_PARALLEL: TMParallelBackend,
    BACKEND_GPU: TMGpuBackend,
}


def get_backend(name: str) -> TMBackend:
    """Instantiate the TM backend adapter for a resolved (non-"auto")
    backend name (see `tm_qsar_benchmark.hardware.resolve_backend`)."""
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise ValueError(f"Unknown TM backend {name!r}; expected one of {list(_BACKENDS)}") from None
