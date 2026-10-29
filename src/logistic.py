"""Logistic regression trained with mini-batch SGD (the final model)."""

import numpy as np

from implementations import compute_logistic_gradient


def logistic_regression_sgd(
    y, X, lambda_=0.0, epochs=30, batch_size=256, learning_rate=0.05, seed=38
):
    """Minimize the logistic loss of implementations.py + lambda_ * ||w||^2.

    Each epoch visits the rows in a new random order, in mini-batches; the
    learning rate is divided by sqrt(epoch + 1) to reduce oscillations near
    the optimum. Labels in {-1, 1} or {0, 1}; the intercept b is not regularized.

    Returns:
        w (D,), b (scalar); P(y = 1 | x) = sigmoid(x @ w + b)
    """
    target = (y == 1).astype(float)
    tx = lambda rows: np.c_[np.ones(len(rows)), X[rows]]  # bias column + features
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1] + 1)  # w[0] is the intercept
    for epoch in range(epochs):
        rate = learning_rate / np.sqrt(epoch + 1)
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            rows = order[start : start + batch_size]
            gradient = compute_logistic_gradient(target[rows], tx(rows), w)
            gradient[1:] += 2 * lambda_ * w[1:]
            w -= rate * gradient
    return w[1:], w[0]
