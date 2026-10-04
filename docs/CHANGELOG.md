

## 4 October 2026

- Preserved all 11 notebook filenames and the original dataset file.
- Rewrote classification-tree evaluation around a predeclared pruning grid, training-only preprocessing, stratified cross-validation and two baselines; retained a warning that the historical holdout had already been inspected.
- Rewrote the regression comparison so MAE and model settings match their labels.
- Extracted stable logistic loss, gradients and perceptron learning into small tested NumPy helpers. Replaced unsupported optimality claims with measured loss and an explicit stopping criterion.
- Added held-out predictions to the random-classification split example, correcting the mapping between score sign and its 0/1 labels.
- Corrected sigmoid boundary comments and class labels; refreshed notebook outputs with fresh kernels.
- Added installation instructions, dataset attribution, contribution/provenance notes, tests, a notebook runner, evaluation results and a GitHub Actions workflow.


