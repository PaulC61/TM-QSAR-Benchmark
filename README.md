# TM-QSAR-Benchmark

Benchmark code for:

> Clarke PFA, Cmelo I, Helin R, Shende MK, Granmo O-C, Fayne D. **"The
> Tsetlin Machine: A 'Third Way' in QSAR Modeling."** *J. Chem. Inf. Model.*
> 2026, 66(11), 6250-6270. DOI: [10.1021/acs.jcim.5c03109](https://doi.org/10.1021/acs.jcim.5c03109),
> PMCID: [PMC13250913](https://pmc.ncbi.nlm.nih.gov/articles/PMC13250913/).

## What this repo is

QSAR (Quantitative Structure-Activity Relationship) models predict a
molecule's biological activity (e.g. whether it inhibits a target protein)
directly from its structure. This repo benchmarks the **Tsetlin Machine
(TM)** -- a comparatively new, logic-based learning algorithm built from
simple "Tsetlin Automata" that learn conjunctive (AND) clauses over
binarized inputs -- against two well-established QSAR baselines,
**Random Forest** and **XGBoost**, across a panel of opioid-receptor-family
classification and regression datasets (MOR, DOR, KOR, CYP2D6, CYP3A4,
MDR1).

The motivation: TMs are interpretable by construction (their clauses are
literally human-readable logical rules) and cheap to run, which could make
them an attractive "third way" between black-box gradient-boosted trees and
classical statistical QSAR methods -- *if* they are competitive on
predictive performance. This repo answers that "is it competitive?"
question empirically; it does **not** cover the clause-level
interpretability analysis from the paper (WAC scores, TM-MPM visualization)
-- that lives in the separate **TM-QSAR-Interpretability** repo.

Headline results from the paper (ECFP descriptors, scaffold-split
evaluation): the TM reaches ROC-AUC ~ 0.87 / PRC-AUC ~ 0.77 on MOR opioid
receptor activity and ROC-AUC ~ 0.92 / PRC-AUC ~ 0.63 on CYP3A4 metabolic
liability, broadly comparable to Random Forest and XGBoost on the same
splits -- see the paper for full per-target/per-split numbers and
statistical comparisons.

## Repo layout

```
src/tm_qsar_benchmark/   Benchmark package: TM vs RandomForest vs XGBoost
data/opioids/            Input datasets (SMILES + labels/regression targets)
dev/                     TM library dependencies, as git submodules:
                         tmu (CPU), pyTsetlinMachineParallel (parallel CPU /
                         GPU via PyCUDA), chembl_structure_pipeline,
                         useful_rdkit_utils
results/                 Melted/long-format benchmark result CSVs (MACRO_*,
                         MICRO_*) -- committed outputs from prior runs
Polaris_examples/        Additional regression-task example/notebook
notebooks/               Tutorial notebook for exploring result CSVs
post-hoc-analysis.ipynb  In-depth statistical comparison of results (Tukey
                         HSD, effect sizes, plots) -- see that notebook for
                         a deeper dive than the tutorial notebook covers
tests/                   Smoke/unit tests for the pipeline
.devcontainer/           Optional Docker devcontainer (see "Devcontainer")
```

## Setup

This project uses [pixi](https://pixi.sh) for environment management. It
works both directly on your machine (laptop or GPU server) and, optionally,
inside a devcontainer -- **the devcontainer is not required.**

### Plain setup (no devcontainer)

```bash
# 1. Clone with submodules (dev/tmu, dev/pyTsetlinMachineParallel, etc.)
git clone --recurse-submodules <this-repo-url>
cd TM-QSAR-Benchmark
# (if you already cloned without --recurse-submodules:)
git submodule update --init --recursive

# 2. Patch a portability issue in the vendored `tmu` submodule's build
#    flags (an x86-only compiler flag that breaks the build on
#    non-x86 platforms, e.g. Apple Silicon). Safe/idempotent, must be
#    re-run any time the submodule is reset/updated.
pixi run fix-submodules

# 3. Install the environment (builds tmu/pyTsetlinMachineParallel from
#    source; requires no manual compiler setup thanks to pixi's
#    c-compiler/cxx-compiler packages).
pixi install
```

Then run a benchmark:

```bash
pixi run describe-hardware   # shows which TM backend will be auto-selected
pixi run benchmark --n-clauses 800    # opioid MOR/DOR/KOR, 800 TM clauses
pixi run benchmark --n-clauses 1600   # same, with 1600 TM clauses
pixi run test                # unit + smoke tests (pytest)
```

Or invoke the CLI directly for full control (dataset selection, backend
override, CV/HP-search sizing, resuming a partial run):

```bash
pixi run python -m tm_qsar_benchmark.cli --help
pixi run python -m tm_qsar_benchmark.cli --n-clauses 800 --backend cpu \
    --datasets opioids/MOR_cutoff6.csv --n-outer 1 --n-inner 2 \
    --n-trials 2 --n-tm-epochs 2 --run-label smoketest -v
```

The TM backend is auto-selected (`--backend auto`, the default) by probing
for an NVIDIA GPU via `nvidia-smi`: a CUDA-capable server picks the
PyCUDA-backed GPU implementation, a machine without a GPU but with
`pyTsetlinMachineParallel` built picks the parallel CPU implementation, and
otherwise it falls back to the base `tmu` CPU implementation. Override with
`--backend cpu|parallel|gpu` to force a specific one.

Results are written as melted/long-format CSVs to `results/` (e.g.
`MACRO_TM_Benchmark_8`, `MICRO_TM_Benchmark_8`) in the same format as
pre-existing files already committed there -- see `notebooks/` for a
tutorial on loading and visualizing them.

### Devcontainer (optional)

`.devcontainer/` provides an optional, reproducible Docker environment.
`.devcontainer/build` probes for `nvidia-smi` and renders
`.devcontainer/devcontainer.json` from `devcontainer.json.template`,
picking a plain or CUDA-enabled pixi base image accordingly (override with
`GPU=0`/`GPU=1`/`PIXI_IMAGE_TAG` env vars if auto-detection guesses wrong).
It works unmodified on a laptop (CPU-only pixi image) and on a CUDA GPU
server (CUDA pixi image + `--gpus` passthrough). To use it:

```bash
bash .devcontainer/build   # generates .devcontainer/devcontainer.json
# then "Reopen in Container" in VS Code, or use the devcontainer CLI
```

The container's `postStartCommand` runs the same submodule-init /
fix-submodules / `pixi install` steps as the plain setup above, so the
devcontainer and bare-metal setups stay in sync. If you don't use
devcontainers at all, just ignore this directory entirely.

## Interpretability

TM clause/rule interpretability analysis (WAC scores, TM-MPM) is
intentionally **out of scope** for this repo -- see the separate
**TM-QSAR-Interpretability** repo for that functionality.
