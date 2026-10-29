"""Train the final model on all labeled rows and write the submission.

Final model (chosen by 5-fold CV, see cv_experiments.py): feature groups 1-5,
logistic regression with mini-batch SGD, lambda = 0, decision threshold that
maximizes F1 on the training predictions.
"""

import numpy as np

from helpers import create_csv_submission
from implementations import sigmoid
from src.config import SUBMISSION_PATH
from src.logistic import logistic_regression_sgd
from src.metrics import best_threshold
from src.preprocessing import (
    apply_preprocess,
    fit_preprocess,
    load_data,
    load_feature_names,
    load_metadata,
)

MAX_GROUP = 5


def main():
    x_train, x_test, y_train, _, test_ids = load_data()
    names, meta = load_feature_names(), load_metadata()

    state = fit_preprocess(x_train, meta, names, MAX_GROUP)
    X_train, features = apply_preprocess(x_train, meta, names, state)
    X_test, _ = apply_preprocess(x_test, meta, names, state)
    print(f"{len(y_train)} training rows, {len(features)} encoded features")

    w, b = logistic_regression_sgd(y_train, X_train)
    f1, threshold = best_threshold(y_train, sigmoid(X_train @ w + b))
    print(f"Threshold {threshold:.3f} (training F1 {f1:.3f})")

    predictions = np.where(sigmoid(X_test @ w + b) >= threshold, 1, -1)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(test_ids, predictions, SUBMISSION_PATH)
    print(
        f"Submission saved to {SUBMISSION_PATH} ({np.mean(predictions == 1):.1%} predicted positive)"
    )


if __name__ == "__main__":
    main()
