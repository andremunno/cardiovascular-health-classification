"""Stratified k-fold cross-validation of the full pipeline."""

import numpy as np

from src.config import SEED
from src.metrics import auc, best_threshold, f1_score
from src.preprocessing import apply_preprocess, fit_preprocess


def stratified_k_fold(y, k, seed=SEED):
    """Split the row indices into k folds with the same share of positives.

    Returns:
        list of k (train_idx, val_idx) pairs
    """
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == label))
        for f, part in enumerate(np.array_split(rows, k)):
            folds[f].extend(part)
    folds = [np.sort(f) for f in folds]
    return [
        (np.concatenate([folds[g] for g in range(k) if g != f]), folds[f])
        for f in range(k)
    ]


def evaluate_fold(train_fn, X_tr, y_tr, X_va, y_va, threshold=None):
    """Train on one fold and score the held-out part.

    train_fn(X, y) returns a function X -> scores (higher = more likely positive).
    If threshold is None, it is chosen to maximize F1 on the training part.

    Returns:
        dict with f1, auc and the threshold used
    """
    score = train_fn(X_tr, y_tr)
    if threshold is None:
        _, threshold = best_threshold(y_tr, score(X_tr))
    s = score(X_va)
    pred = np.where(s >= threshold, 1, -1)
    return {
        "f1": float(f1_score(y_va, pred)),
        "auc": auc(y_va, s),
        "threshold": float(threshold),
    }


def summarize(fold_results):
    """Mean and standard deviation over folds of each metric."""
    out = {"folds": fold_results}
    for metric in ("f1", "auc"):
        values = np.array([r[metric] for r in fold_results])
        out[metric + "_mean"], out[metric + "_std"] = float(values.mean()), float(
            values.std()
        )
    return out


def cross_validate(X_raw, y, names, meta, max_group, train_fn, k=5, seed=SEED):
    """k-fold CV: preprocessing, model and threshold are fitted on the training part of each fold.

    Returns:
        dict with the per-fold results and the mean/std of F1 and AUC
    """
    results = []
    for tr, va in stratified_k_fold(y, k, seed):
        state = fit_preprocess(X_raw[tr], meta, names, max_group)
        X_tr, _ = apply_preprocess(X_raw[tr], meta, names, state)
        X_va, _ = apply_preprocess(X_raw[va], meta, names, state)
        results.append(evaluate_fold(train_fn, X_tr, y[tr], X_va, y[va]))
    return summarize(results)
