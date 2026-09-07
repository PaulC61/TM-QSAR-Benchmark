"""Backend adapters that hide the API differences between the Tsetlin
Machine implementations this project has historically used on different
hardware:

* `cpu`      - `tmu.models.classification.coalesced_classifier.TMCoalescedClassifier`
               / `tmu.models.regression.vanilla_regressor.TMRegressor`,
               constructed with `platform="CPU"` (single/multi-threaded CPU,
               always available).
* `gpu`      - the *same* `tmu` classifier/regressor classes, constructed
               with `platform="CUDA"` instead (CUDA GPU, e.g. H100/H200
               servers). `tmu` ships its own CUDA clause bank
               (`tmu.clause_bank.clause_bank_cuda.ClauseBankCUDA`); no
               separate `PyTsetlinMachineCUDA`/`PyCUDATsetlinMachine`
               package is needed, just `pycuda` installed (`pixi install -e
               gpu` on a CUDA-toolchain-equipped machine) -- see
               `hardware.gpu_backend_available`. Because it's the same
               `tmu` model classes as `cpu`, `gpu` also gets regression for
               free (the old separate-package GPU backend never had one).
* `parallel` - `pyTsetlinMachineParallel.tm.MultiClassTsetlinMachine`
               (OpenMP multi-core CPU; classification only -- it doesn't
               ship a regressor).

This is also where a real bug from the old `benchmark_8_GPU.py` copy is
fixed: its HP-search objective computed `Y_val_css` but then referenced the
similarly-named but undefined `Y_val_ccs`, a copy-paste artifact from the
`_para` script it was cloned from that never actually ran correctly.
"""
from __future__ import annotations

import importlib
from typing import Protocol

import numpy as np
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


def _parallel_tm_ccs(tm, X, n_classes=2, n_clauses=1000, n_jobs=22):
    """Class-clause-sum reducer for the array-based `parallel` TM
    implementation, ported from `benchmark_8_para.py`.

    NOTE: the original script computed the (purely deterministic,
    alternating +1/-1) clause polarity `mask` by farming
    ``range(n_clauses)`` out to a fresh 22-worker `joblib.Parallel` *process*
    pool on every single prediction call (i.e. every epoch, for both train
    and test). That's pure overhead for a trivial computation -- on a
    laptop (far fewer cores, much higher process-spawn cost than the
    original many-core server) this made every TM epoch take tens of
    seconds instead of milliseconds. Replaced with the equivalent
    vectorized numpy expression (`n_jobs` is kept as a no-op parameter for
    backend-interface compatibility). `_n_jobs` on the backend is unused
    now but left in place in case a genuinely parallel workload is added
    here later.
    """
    active_clauses = tm.transform(X, inverted=False)
    weight_state = tm.get_state()
    mask = np.where(np.arange(n_clauses) % 2 == 0, 1, -1)

    ccs = []
    for i_class in range(n_classes):
        clause_class_weights = weight_state[i_class][0] * mask
        active_clause_class_weights = (
            clause_class_weights * active_clauses[:, (i_class * n_clauses):(i_class + 1) * n_clauses]
        )
        class_clause_sum = active_clause_class_weights.sum(axis=1)
        ccs.append(class_clause_sum)
    return np.array(ccs)


class _TmuBackend:
    """Shared implementation for `cpu` and `gpu`, which are both just `tmu`
    (`TMCoalescedClassifier` / `TMRegressor`) constructed with a different
    `platform` kwarg ("CPU" vs "CUDA") -- see class docstring at module
    top for why no separate GPU package is needed."""

    name: str
    _platform: str

    def make_classifier(self, params: dict, n_clauses: int):
        from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

        clf_params = dict(params)
        clf_params["number_of_clauses"] = n_clauses
        clf_params["weighted_clauses"] = True
        clf_params["platform"] = self._platform
        return TMCoalescedClassifier(**clf_params)

    def make_regressor(self, params: dict, n_clauses: int):
        from tmu.models.regression.vanilla_regressor import TMRegressor

        reg_params = dict(params)
        reg_params["number_of_clauses"] = n_clauses
        reg_params["weighted_clauses"] = True
        reg_params["platform"] = self._platform
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


class TMCpuBackend(_TmuBackend):
    """`tmu` (single/multi-threaded CPU, `platform="CPU"`) backend -- the
    always-available default when no faster backend is installed/detected."""

    name = BACKEND_CPU
    _platform = "CPU"


class TMGpuBackend(_TmuBackend):
    """`tmu` CUDA GPU backend (`platform="CUDA"`), e.g. H100/H200 servers.
    Requires `pycuda` (opt-in `gpu` pixi feature); see
    `hardware.gpu_backend_available`."""

    name = BACKEND_GPU
    _platform = "CUDA"


class TMParallelBackend:
    """Multi-core CPU backend via `pyTsetlinMachineParallel` (OpenMP).
    Classification only -- this package doesn't ship a regressor."""

    name = BACKEND_PARALLEL
    _module_name = "pyTsetlinMachineParallel"
    _n_jobs = 22

    def _model_cls(self):
        module = importlib.import_module(f"{self._module_name}.tm")
        return module.MultiClassTsetlinMachine

    def make_classifier(self, params: dict, n_clauses: int):
        clf_params = dict(params)
        clf_params["number_of_clauses"] = n_clauses
        clf_params["weighted_clauses"] = True
        return self._model_cls()(**clf_params)

    def make_regressor(self, params: dict, n_clauses: int):
        raise NotImplementedError(
            f"{self.name} backend has no regression model; use the cpu or gpu backend for regression tasks."
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
