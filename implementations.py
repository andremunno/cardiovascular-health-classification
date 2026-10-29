"""The six methods required by the project, and the losses/gradients they use.

Shapes: y (N,), tx (N, D), w (D,). Every method returns (w, loss), where loss
is computed at the returned w and does not include the regularization term.
"""

import numpy as np

# ---- Losses and gradients ----


def compute_mse(y, tx, w):
    """Mean squared error, with the factor 1/2 of the lecture notes."""
    e = y - tx @ w
    return e @ e / (2 * len(y))


def compute_gradient(y, tx, w):
    """Gradient of compute_mse with respect to w."""
    e = y - tx @ w
    return -tx.T @ e / len(y)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Yield num_batches mini-batches (y_batch, tx_batch) of batch_size consecutive rows.

    With shuffle, each batch starts at a random position (batches may overlap);
    without it, the batches are taken in order, cycling over the data.
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)
    max_batches = data_size // batch_size
    remainder = data_size - max_batches * batch_size
    if shuffle:
        starts = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            starts += np.random.randint(remainder + 1, size=num_batches)
    else:
        starts = np.array([i % max_batches for i in range(num_batches)]) * batch_size
    for start in starts:
        yield y[start : start + batch_size], tx[start : start + batch_size]


def sigmoid(t):
    """Numerically stable sigmoid 1 / (1 + exp(-t))."""
    return np.exp(-np.logaddexp(0, -np.asarray(t, dtype=float)))


def compute_logistic_loss(y, tx, w):
    """Mean negative log-likelihood of the logistic model; y in {0, 1}."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of compute_logistic_loss with respect to w; y in {0, 1}."""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ---- Required methods ----


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with gradient descent on the MSE."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with stochastic gradient descent (batch size 1) on the MSE."""
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            w = w - gamma * compute_gradient(y_batch, tx_batch, w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares with the normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression: minimizes MSE + lambda_ * ||w||^2 (normal equations)."""
    A = tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w = np.linalg.solve(A, tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with gradient descent; y in {0, 1}."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression with gradient descent on loss + lambda_ * ||w||^2; y in {0, 1}.

    The returned loss does not include the penalty.
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * (compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w)
    return w, compute_logistic_loss(y, tx, w)
