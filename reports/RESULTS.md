# Recorded experiment results

Local run: **4 October 2026**, Python 3.12.14 on Windows. Direct package versions are in [environment.json](environment.json) and pinned in `requirements.txt`. Machine-readable results are in [metrics.json](metrics.json).

## Cleveland classification experiment

The dataset contains 303 rows. The fixed stratified split assigns 227 rows to training and 76 to the holdout. Six missing input cells are imputed inside the training pipelines. The target is 0 for no disease and 1 for original labels greater than zero.

The pruning grid was `[0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08]`. Five shuffled stratified folds, seed 42, selected `ccp_alpha=0.01` using mean balanced accuracy. Its training cross-validation balanced accuracy was **0.8064**, with fold standard deviation **0.0535**. That standard deviation is not a confidence interval.

| Model | Holdout accuracy | Holdout balanced accuracy | Holdout F1 |
|---|---:|---:|---:|
| Majority-class baseline | 0.5395 | 0.5000 | 0.0000 |
| Logistic regression baseline | 0.8553 | 0.8554 | 0.8451 |
| Pruned decision tree | 0.7632 | 0.7596 | 0.7353 |

Balanced accuracy averages recall across classes. F1 treats the disease-present label as positive. For the pruned tree, the confusion matrix is `[[33, 8], [10, 25]]`, with true classes as rows and predicted classes as columns in order `[0, 1]`.

**The logistic regression baseline performed better than the pruned tree on this split.** Pruning provides a controlled model-selection exercise; it does not establish that a tree is the best model for the dataset.

## Limits of the evidence

- The original notebook already explored test performance. Reorganising evaluation does not create an independent external validation dataset. Treat these as educational results.
- One small historical holdout is sensitive to the split. Do not repeatedly tune against these recorded test scores.
- No external population validation, clinical assessment or production deployment was performed.
- Toy-data notebooks demonstrate mechanics; their training accuracy is not evidence of real-world predictive quality.
- The regression notebook reports **MAE**, not MSE. The NumPy logistic routine reports training log loss and convergence status without calling a separating line optimal.

## Reproduce

From the repository root, using the installed environment:

```bash
python -m pytest -q
python scripts/run_notebooks.py
python scripts/evaluate.py
```

Do not update this table by hand to show a better outcome. If the protocol changes, explain the reason, rerun it and regenerate the measured values.
