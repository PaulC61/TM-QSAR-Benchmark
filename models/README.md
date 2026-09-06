# Final TM models

Ready-to-use Tsetlin Machine classifiers, refit on 100% of their training
data (not a CV split) at 1600 clauses.

| File | Target | Descriptor | Params | Epochs | CV mean Test ROC_AUC (1600 clauses) |
|---|---|---|---|---|---|
| `TM_CYP3A4_1600clauses.pkl` | CYP3A4 inhibition (`data/opioids/CYP3A4_cutoff6.csv`) | ECFP (2048 bits, radius 2) | T=6626, s=4.32 | 25 | 0.887 |
| `TM_CYP2D6_1600clauses.pkl` | CYP2D6 inhibition (`data/opioids/CYP2D6_cutoff6.csv`) | RDKit2D (quantile-binarized) | T=6800, s=3.05 | 33 | 0.631 |

Hyperparameters (T, s), descriptor choice and epoch count were not
re-searched -- they're aggregated from this repo's own CV results in
`results/MACRO_TM_Benchmark_ALL_NClauses` (TsetlinMachine rows,
`N_Clauses == 1600`): the descriptor with the best mean Test ROC_AUC per
target, that descriptor's mean best-fold `(T, s)`, and the training epoch
with the highest mean Test ROC_AUC across folds. Regenerate with:

```bash
pixi run python scripts/train_final_models.py -v
```

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
