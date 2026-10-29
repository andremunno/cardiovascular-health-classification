"""Checks of the cross-validation helpers and of the linear SVM baseline."""

import unittest

import numpy as np

from src.svm import linear_svm_sgd
from src.validation import evaluate_fold, stratified_k_fold


class ValidationTests(unittest.TestCase):
    def test_folds_are_disjoint_complete_and_stratified(self):
        y = np.r_[np.ones(100), -np.ones(900)]
        folds = stratified_k_fold(y, 5, seed=38)
        self.assertEqual(len(folds), 5)
        all_val = np.concatenate([va for _, va in folds])
        np.testing.assert_array_equal(np.sort(all_val), np.arange(len(y)))
        for tr, va in folds:
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            self.assertEqual(len(tr) + len(va), len(y))
            self.assertEqual(np.mean(y[va] == 1), 0.1)

    def test_threshold_is_chosen_on_training_part(self):
        X_tr, y_tr = np.array([[0.1], [0.9], [0.2], [0.8]]), np.array([-1, 1, -1, 1])
        X_va, y_va = np.array([[0.85], [0.3]]), np.array([1, -1])
        r = evaluate_fold(lambda X, y: (lambda Z: Z[:, 0]), X_tr, y_tr, X_va, y_va)
        self.assertEqual(r["threshold"], 0.8)
        self.assertEqual(r["f1"], 1.0)
        self.assertEqual(
            evaluate_fold(
                lambda X, y: (lambda Z: Z[:, 0]), X_tr, y_tr, X_va, y_va, np.inf
            )["f1"],
            0.0,
        )


class SVMTests(unittest.TestCase):
    def test_learns_a_separable_problem(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(400, 2))
        y = np.where(X[:, 0] + X[:, 1] > 0, 1.0, -1.0)
        w, b = linear_svm_sgd(
            y, X, lambda_=1e-4, epochs=30, batch_size=32, learning_rate=0.1
        )
        self.assertGreater(np.mean(np.sign(X @ w + b) == y), 0.95)


if __name__ == "__main__":
    unittest.main()
