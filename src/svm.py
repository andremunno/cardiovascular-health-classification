"""Linear SVM trained with mini-batch SGD (baseline for logistic regression)."""

import numpy as np


def linear_svm_sgd(
    y,
    X,
    lambda_=1e-4,
    epochs=30,
    batch_size=256,
    learning_rate=0.01,
    positive_weight=None,
    seed=38,
):
    """Minimize mean weighted hinge loss c_i * max(0, 1 - y_i (x_i w + b)) + lambda_ * ||w||^2.

    Positives get weight c = positive_weight (default: n_negative / n_positive),
    negatives c = 1: without it, on imbalanced data the hinge loss is minimized
    by pushing every row to the negative side. Same optimizer as
    logistic_regression_sgd: shuffled mini-batches, learning rate divided by
    sqrt(epoch + 1). Labels in {-1, 1}; the intercept b is not regularized.

    Returns:
        w (D,), b (scalar); the score of a row x is x @ w + b
    """
    if positive_weight is None:
        positive_weight = np.sum(y == -1) / np.sum(y == 1)
    weight = np.where(y == 1, positive_weight, 1.0)
    rng = np.random.default_rng(seed)
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(epochs):
        rate = learning_rate / np.sqrt(epoch + 1)
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            rows = order[start : start + batch_size]
            Xb, cy = X[rows], weight[rows] * y[rows]
            active = y[rows] * (Xb @ w + b) < 1  # rows inside the margin
            w -= rate * (-(Xb[active].T @ cy[active]) / len(rows) + 2 * lambda_ * w)
            b -= rate * (-cy[active].sum() / len(rows))
    return w, b
