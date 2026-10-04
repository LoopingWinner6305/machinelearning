"""Binary logistic loss and a perceptron, implemented with NumPy.

These are educational routines, not general estimator replacements.
"""
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    exp_neg_abs = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exp_neg_abs), exp_neg_abs / (1 + exp_neg_abs))

def binary_log_loss_from_logits(y, logits):
    y, logits = np.asarray(y, dtype=float), np.asarray(logits, dtype=float)
    if y.shape != logits.shape or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError('Matching nonempty arrays and binary labels are required')
    return float(np.mean(np.logaddexp(0, logits) - y * logits))

def logistic_gradient(X, y, weights):
    return X.T @ (sigmoid(X @ weights) - y) / len(y)

def fit_logistic(X, y, learning_rate=0.05, max_iterations=2000, tolerance=1e-6):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or len(y) == 0:
        raise ValueError('Expected a nonempty feature matrix and aligned target vector')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(X).all():
        raise ValueError('Expected finite features and binary labels')
    if learning_rate <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError('Training controls must be positive')
    design = np.column_stack([np.ones(len(X)), X])
    weights = np.zeros(design.shape[1])
    history = []
    converged = False
    for iteration in range(max_iterations):
        gradient = logistic_gradient(design, y, weights)
        weights -= learning_rate * gradient
        loss = binary_log_loss_from_logits(y, design @ weights)
        current_gradient = logistic_gradient(design, y, weights)
        norm = float(np.linalg.norm(current_gradient))
        history.append({'iteration': iteration + 1, 'loss': loss, 'gradient_norm': norm})
        if norm < tolerance:
            converged = True
            break
    return weights, history, converged

def predict_logistic(X, weights):
    design = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    return (sigmoid(design @ weights) >= 0.5).astype(int)

def fit_perceptron(X, y, max_epochs=100):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or len(y) == 0 or not np.isin(y, [-1, 1]).all():
        raise ValueError('Expected nonempty aligned data and labels -1 or +1')
    if max_epochs < 1:
        raise ValueError('max_epochs must be positive')
    weights = np.zeros(X.shape[1]); bias = 0.0; history = []
    for _ in range(max_epochs):
        updates = 0
        for xi, yi in zip(X, y):
            if yi * (xi @ weights + bias) <= 0:
                weights += yi * xi; bias += yi; updates += 1
        history.append(updates)
        if updates == 0:
            break
    return weights, bias, history
