"""Hardware detection for auto-selecting a Tsetlin Machine backend.

This project historically kept separate copies of the benchmark script per
machine: a CPU-only ("regular TM library", `tmu`) version, a multi-core CPU
version (`pyTsetlinMachineParallel`), and a GPU version (a separate
`PyTsetlinMachineCUDA`/`PyCUDATsetlinMachine` package), hand-picked depending
on which server it ran on. This module replaces that manual process with
automatic detection so the same consolidated pipeline "just works" on a
laptop with no GPU and on an H100/H200 GPU server alike.

GPU support does *not* need a separate TM package: `tmu` itself ships a CUDA
clause bank (`tmu.clause_bank.clause_bank_cuda.ClauseBankCUDA`) and switches
to it for any model constructed with `platform="CUDA"` -- same
`TMCoalescedClassifier`/`TMRegressor` classes, same fit/predict API, just a
different constructor kwarg (see `tm_backends.py`). The only extra
requirement is `pycuda` (which needs a working CUDA toolchain, so it's an
opt-in pixi feature -- `pixi install -e gpu` -- rather than a default
dependency; see pyproject.toml's `[tool.pixi.feature.gpu]`).

Detection is done by shelling out to `nvidia-smi` (present with the NVIDIA
driver, absent on a Mac or a GPU-less Linux box), mirroring the pattern used
in the sibling `aerleumLitScraperV2` project's `hardware.py`. We additionally
confirm `tmu`'s own CUDA clause bank actually initialized successfully
(i.e. `pycuda` imported cleanly) before selecting the GPU backend, so a
GPU-equipped-but-not-yet-configured machine still falls back to a CPU
backend rather than crashing.
"""
from __future__ import annotations

import functools
import importlib.util
import subprocess

# Backend identifiers, in the order hardware detection prefers them.
BACKEND_GPU = "gpu"
BACKEND_PARALLEL = "parallel"
BACKEND_CPU = "cpu"
VALID_BACKENDS = (BACKEND_GPU, BACKEND_PARALLEL, BACKEND_CPU, "auto")


@functools.lru_cache(maxsize=1)
def has_nvidia_gpu() -> bool:
    """True if `nvidia-smi` reports at least one NVIDIA GPU.

    Returns False (never raises) if `nvidia-smi` isn't installed/found, e.g.
    on a MacBook or a GPU-less Linux box.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(result.stdout.strip())


@functools.lru_cache(maxsize=1)
def gpu_backend_available() -> bool:
    """True if both an NVIDIA GPU is detected *and* `tmu`'s own CUDA clause
    bank successfully imported `pycuda` in this environment (`tmu` is a
    core dependency; `pycuda` is the opt-in `gpu` pixi feature)."""
    if not has_nvidia_gpu():
        return False
    if importlib.util.find_spec("tmu") is None:
        return False
    from tmu.clause_bank.clause_bank_cuda import cuda_installed

    return bool(cuda_installed)


@functools.lru_cache(maxsize=1)
def parallel_backend_available() -> bool:
    """True if the multi-core CPU Tsetlin Machine package
    (`pyTsetlinMachineParallel`) is importable in this environment."""
    return importlib.util.find_spec("pyTsetlinMachineParallel") is not None


def resolve_backend(requested: str = "auto") -> str:
    """Resolve a requested backend name to a concrete, available one.

    `"auto"` picks GPU if a CUDA GPU is detected and `tmu`'s CUDA clause
    bank has `pycuda` available, otherwise the multi-core parallel CPU
    backend if installed, otherwise the plain single-core `tmu` CPU backend
    (always available, it is a core dependency). Any explicit non-"auto"
    value is validated and returned unchanged (explicit user choice wins,
    even if unavailable -- the caller will get an ImportError with a clear
    cause).
    """
    if requested not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {requested!r}; expected one of {VALID_BACKENDS}")

    if requested != "auto":
        return requested

    if gpu_backend_available():
        return BACKEND_GPU
    if parallel_backend_available():
        return BACKEND_PARALLEL
    return BACKEND_CPU


def describe(requested: str = "auto") -> str:
    """Human-readable summary of detected hardware and the resulting
    backend choice, useful for logging/diagnostics."""
    resolved = resolve_backend(requested)
    gpu_note = "GPU detected" if has_nvidia_gpu() else "no GPU detected"
    return f"Hardware: {gpu_note} -> TM backend={resolved!r} (requested={requested!r})"


if __name__ == "__main__":
    print(describe())
