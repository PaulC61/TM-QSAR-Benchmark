# Final TM models

Ready-to-use Tsetlin Machine classifiers, refit on 100% of their training
data (not a CV split) at 1600 clauses.

| File | Target | Descriptor | Params | Epochs | CV mean Test ROC_AUC (1600 clauses) |
|---|---|---|---|---|---|
| `TM_CYP3A4_1600clauses.pkl` | CYP3A4 inhibition (`data/opioids/CYP3A4_cutoff6.csv`) | ECFP (2048 bits, radius 2) | T=6626, s=4.32 | 25 | 0.887 |
| `TM_CYP2D6_1600clauses.pkl` | CYP2D6 inhibition (`data/opioids/CYP2D6_cutoff6.csv`) | RDKit2D (quantile-binarized) | T=6800, s=3.05 | 33 | 0.631 |

Hyperparameters (T, s), descriptor choice and epoch count are never
guessed or hand-picked -- they're aggregated from a dataset's own CV
performance distribution (whichever descriptor has the best mean Test
ROC_AUC, that descriptor's mean best-fold `(T, s)`, and the training epoch
with the highest mean Test ROC_AUC across folds), then a final model is
refit on 100% of the data using that config. This is generalised to any
classification target via `scripts/train_final_models.py`, which only
needs a dataset's CSV path plus its SMILES/label column names:

```bash
# Reuses this repo's existing CV results for CYP3A4/CYP2D6 (fast: no CV
# search, just a refit).
pixi run python scripts/train_final_models.py \
    --dataset opioids/CYP3A4_cutoff6.csv --target-name CYP3A4 --n-clauses 1600
pixi run python scripts/train_final_models.py \
    --dataset opioids/CYP2D6_cutoff6.csv --target-name CYP2D6 --n-clauses 1600

# A brand new/never-benchmarked target: runs the same repeated 5x5
# nested-CV benchmark used for every other target in this project first
# (to get a performance distribution), then aggregates and refits exactly
# as above. Only needed once per (dataset, clause count) -- later calls
# reuse the results this produces.
pixi run python scripts/train_final_models.py \
    --dataset my_new_target.csv --smiles-col Structure --prop-col Active \
    --target-name MyNewTarget --n-clauses 1600
```

See `scripts/train_final_models.py --help` for the full set of flags
(custom data/results/output directories, forcing a fresh CV run even for
an already-benchmarked target, and shrinking the CV search budget for a
quick/experimental pass via `--n-outer`/`--n-inner`/`--n-trials`/
`--n-tm-epochs`).

## Loading a model


Each pickle is a plain `dict` (not just the bare model), produced with
whichever TM backend was auto-detected on the training machine (see the
`backend` key) so predictions are reproducible with the matching package:

```python
import pickle

with open("models/TM_CYP3A4_1600clauses.pkl", "rb") as f:
    payload = pickle.load(f)

model = payload["model"]           # fitted TM classifier
print(payload["descriptor"])       # "ECFP" or "RDKit2D"
print(payload["params"], payload["n_clauses"], payload["n_epochs"])
```

Featurize new SMILES the same way before calling `model.predict(...)`:

- **ECFP** targets (e.g. CYP3A4): use `payload["fp_size"]`/`payload["fp_rad"]`
  with `tm_qsar_benchmark.descriptors.gen_ecfp_arr`.
- **RDKit2D** targets (e.g. CYP2D6): compute RDKit2D descriptors with
  `tm_qsar_benchmark.descriptors.gen_rdkit2D_arr`, then binarize with the
  *fitted* `payload["binarizer"]` (`.transform(...)`, not `.fit_transform(...)`
  -- it must reuse the thresholds fit on the training data).

These are classification-performance benchmark models only; they do not
include the clause-level interpretability tooling from the separate
TM-QSAR-Interpretability repo.
