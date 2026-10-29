"""Checks of the stable sigmoid and of the SGD logistic regression."""

import unittest

import numpy as np

from implementations import sigmoid
from src.logistic import logistic_regression_sgd


def objective(w, b, X, y01, lambda_):
    z = X @ w + b
    return np.mean(np.logaddexp(0, z) - y01 * z) + lambda_ * w @ w


class LogisticTests(unittest.TestCase):
    def test_stable_extreme_probabilities(self):
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_allclose(
                sigmoid(np.array([-10000, 0, 10000])), [0, 0.5, 1]
            )

    def test_one_full_batch_step_is_a_gradient_step(self):
        X = np.array([[1.0, 2.0], [-2.0, 1.0], [0.0, -1.0]])
        y = np.array([1, -1, -1])
        w, b = logistic_regression_sgd(
            y, X, lambda_=0.3, epochs=1, batch_size=10, learning_rate=0.2
        )

        # Numerical gradient at the starting point (b = 0, w = 0); theta = [b, w1, w2]
        def f(theta):
            return objective(theta[1:], theta[0], X, y == 1, 0.3)

        eps = 1e-6
        grad = np.array([(f(e) - f(-e)) / (2 * eps) for e in np.eye(3) * eps])
        np.testing.assert_allclose(np.r_[b, w], -0.2 * grad, atol=1e-8)

    def test_learns_a_separable_problem_with_both_label_codings(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(301, 3))  # not divisible by the batch size
        y = np.where(X[:, 0] - X[:, 1] > 0.4, 1, -1)
        w, b = logistic_regression_sgd(
            y, X, epochs=40, batch_size=32, learning_rate=0.2
        )
        w01, _ = logistic_regression_sgd(
            (y == 1).astype(int), X, epochs=40, batch_size=32, learning_rate=0.2
        )
        np.testing.assert_array_equal(w, w01)
        self.assertGreater(np.mean(np.where(X @ w + b > 0, 1, -1) == y), 0.95)


if __name__ == "__main__":
    unittest.main()
