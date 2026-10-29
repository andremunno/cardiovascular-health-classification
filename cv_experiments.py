"""All cross-validation experiments of the report. Run: python cv_experiments.py

Stratified 5-fold CV. In every fold, preprocessing, interaction choice, model
and F1 threshold are fitted on the training part only.
  1. Feature groups: logistic regression on groups 1, <=2, ..., <=5.
  2. Threshold: tuned for F1 vs fixed 0.5 (final model).
  3. Regularization: lambda grid (final model).
  4. Non-linear terms: top-5 Group 1 interactions, and age^2 (suggested by the EDA).
  5. Baselines on the final features: always negative, ridge regression,
     logistic regression with full-batch GD, linear SVM.
  6. Alternative recodings (e.g. age categories) added to the final features.
Results go to experiments/results/cv_results.json (about 1.5 hours).
"""

import json

import numpy as np

from implementations import logistic_regression, ridge_regression, sigmoid
from src.config import RESULTS_DIR, SEED
from src.logistic import logistic_regression_sgd
from src.preprocessing import (
    apply_preprocess,
    fit_preprocess,
    load_data,
    load_feature_names,
    load_metadata,
)
from src.svm import linear_svm_sgd
from src.validation import evaluate_fold, stratified_k_fold, summarize

K = 5
FINAL_GROUP = 5
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2]


# Each model: train(X, y) returns a function X -> scores


def logistic_sgd(lambda_=0.0):
    def train(X, y):
        w, b = logistic_regression_sgd(y, X, lambda_, seed=SEED)
        return lambda Z: sigmoid(Z @ w + b)

    return train


def with_bias(X):
    return np.c_[np.ones(len(X), dtype=X.dtype), X]


def ridge(X, y):
    w, _ = ridge_regression(y.astype(float), with_bias(X).astype(float), 1e-4)
    return lambda Z: with_bias(Z) @ w


def logistic_gd(X, y):
    tx = with_bias(X)
    w, _ = logistic_regression(
        (y == 1).astype(float), tx, np.zeros(tx.shape[1]), 300, 0.5
    )
    return lambda Z: with_bias(Z) @ w


def svm(X, y):
    w, b = linear_svm_sgd(y.astype(float), X, seed=SEED)
    return lambda Z: Z @ w + b


def always_negative(X, y):
    return lambda Z: np.zeros(len(Z))


# Extra terms for the non-linear experiments


def source_variable(column, meta):
    """Original variable of an encoded column, e.g. DIABETE3__1 -> DIABETE3."""
    while column not in meta:
        column = column.rsplit("__", 1)[0]
    return column


def top_interactions(X, y, features, meta, score, k=5):
    """The k Group 1 column pairs whose product would most reduce the logistic loss.

    Ranked by |gradient of the loss| with respect to each product, at the linear model.
    """
    var = [source_variable(f, meta) for f in features]
    g1 = [
        i
        for i, f in enumerate(features)
        if meta[var[i]]["group"] == "1" and not f.endswith("__missing")
    ]
    pairs = [(i, j) for a, i in enumerate(g1) for j in g1[a + 1 :] if var[i] != var[j]]
    P = np.column_stack([X[:, i] * X[:, j] for i, j in pairs])
    P = (P - P.mean(0)) / (P.std(0) + 1e-12)
    residual = score(X) - (y == 1)
    return [pairs[t] for t in np.argsort(-np.abs(P.T @ residual))[:k]]


def add_terms(X_tr, X_va, features, pairs, age_squared):
    """Append the pair products (and age^2), standardized with training statistics."""

    def terms(X):
        cols = [X[:, i] * X[:, j] for i, j in pairs]
        if age_squared:
            cols.append(X[:, features.index("_AGE80")] ** 2)
        return np.column_stack(cols)

    T_tr, T_va = terms(X_tr), terms(X_va)
    mean, std = T_tr.mean(0), T_tr.std(0) + 1e-12
    return np.c_[X_tr, (T_tr - mean) / std], np.c_[X_va, (T_va - mean) / std]


def main():
    raw, _, y, _, _ = load_data()
    names, meta = load_feature_names(), load_metadata()
    results = {}

    for f, (tr, va) in enumerate(stratified_k_fold(y, K, SEED)):
        for group in range(1, FINAL_GROUP + 1):
            state = fit_preprocess(raw[tr], meta, names, group)
            X_tr, features = apply_preprocess(raw[tr], meta, names, state)
            X_va, _ = apply_preprocess(raw[va], meta, names, state)

            def run(label, train, X_a=X_tr, X_b=X_va, threshold=None):
                results.setdefault(label, []).append(
                    evaluate_fold(train, X_a, y[tr], X_b, y[va], threshold)
                )

            if group < FINAL_GROUP:
                run(f"1. groups <={group}", logistic_sgd())
                continue

            # Final features: all the other experiments share this preprocessing
            final = logistic_sgd()(X_tr, y[tr])
            run(f"1. groups <={group} (final)", lambda X, t: final)
            run("2. final, threshold 0.5", lambda X, t: final, threshold=0.5)
            for lam in LAMBDAS:
                run(f"3. final, lambda={lam:g}", logistic_sgd(lam))
            pairs = top_interactions(X_tr, y[tr], features, meta, final)
            for label, age_squared in (
                ("4. final + 5 interactions", False),
                ("4. final + 5 interactions + age^2", True),
            ):
                A_tr, A_va = add_terms(X_tr, X_va, features, pairs, age_squared)
                run(label, logistic_sgd(), A_tr, A_va)
            run("5. always negative", always_negative, threshold=np.inf)
            run("5. ridge regression", ridge)
            run("5. logistic, full-batch GD", logistic_gd)
            run("5. linear SVM", svm)

            state = fit_preprocess(raw[tr], meta, names, group, alternatives=True)
            A_tr, _ = apply_preprocess(raw[tr], meta, names, state)
            A_va, _ = apply_preprocess(raw[va], meta, names, state)
            run("6. final + alternatives", logistic_sgd(), A_tr, A_va)
        print(
            f"fold {f + 1}/{K} done; interactions: {[(features[i], features[j]) for i, j in pairs]}",
            flush=True,
        )

    # Paired comparison with the final model, fold by fold
    reference = np.array(
        [r["f1"] for r in results[f"1. groups <={FINAL_GROUP} (final)"]]
    )
    summary = {}
    print(
        f"\n{'experiment':36} {'F1':>15} {'AUC':>15} {'dF1 vs final':>16} {'folds better':>12}"
    )
    for label in sorted(results):
        s = summarize(results[label])
        d = np.array([r["f1"] for r in results[label]]) - reference
        s["df1_mean"], s["df1_std"], s["folds_better"] = (
            float(d.mean()),
            float(d.std()),
            int(np.sum(d > 0)),
        )
        summary[label] = s
        print(
            f"{label:36} {s['f1_mean']:.4f} ± {s['f1_std']:.4f} {s['auc_mean']:.4f} ± {s['auc_std']:.4f} "
            f"{s['df1_mean']:+.4f} ± {s['df1_std']:.4f} {s['folds_better']:>8}/{K}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "cv_results.json").write_text(
        json.dumps({"folds": K, "seed": SEED, "results": summary}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
