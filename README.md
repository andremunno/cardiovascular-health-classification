# Cardiovascular Health Classification

A reproducible machine learning pipeline for classifying self-reported myocardial infarction or coronary heart disease (**MICHD**) from **BRFSS 2015** survey responses.

Models and preprocessing are implemented with **NumPy**, including survey-code decoding, categorical encoding, missingness indicators, logistic regression and stratified cross-validation.

## Results

The final model achieves **F1 0.430 ± 0.004** in stratified five-fold cross-validation and **0.444** on the final AIcrowd submission.

| Model | CV F1, mean ± std | Mean ROC AUC |
| --- | ---: | ---: |
| **Logistic regression — mini-batch SGD, final** | **0.430 ± 0.004** | **0.861** |
| Logistic regression — full-batch GD | 0.430 ± 0.004 | 0.861 |
| Linear SVM — class-weighted | 0.427 ± 0.004 | 0.860 |
| Ridge regression | 0.419 ± 0.003 | 0.854 |
| Always-negative baseline | 0.000 ± 0.000 | 0.500 |

The positive class accounts for **8.83%** of training observations, so positive-class F1 is the primary metric. For the final logistic model, optimizing the threshold on training predictions raises held-out CV F1 from **0.241** at threshold 0.5 to **0.430**. Scores and AUC remain unchanged.

Interaction and alternative-recoding variants reach approximately **0.432** CV F1. Their gains are small and vary across folds; the final pipeline retains the simpler feature representation.

## Quick start

### 1. Install dependencies

Training, inference and the repository tests require **Python 3 and NumPy**:

```sh
python -m pip install numpy
```

The exploratory notebook additionally requires Matplotlib and a Jupyter environment.

### 2. Add the data

Use the three CSV files supplied for the course competition. The raw dataset is not included in the repository.

```text
data/
  raw/
    x_train.csv
    y_train.csv
    x_test.csv
  metadata/
    variables_BRFSS_2015.csv
```

The provided split contains **328,135 labelled training observations**, **109,379 unlabelled test observations** and **321 original features**.

The first run creates `data/raw/cache.npz` to speed up subsequent loads. Delete this cache if you replace the input CSV files or interrupt cache creation.

### 3. Train and generate predictions

Run from the repository root:

```sh
python run.py
```

This fits preprocessing and the final model on all labelled observations and writes:

```text
data/submissions/submission.csv
```

The CSV preserves the test IDs and returns labels in `{-1, 1}`.

## Pipeline

1. **Select features:** compare five cumulative relevance groups and exclude target-revealing variables, administrative fields and alternative recodings by default.
2. **Clean survey codes:** distinguish missing answers from genuine zeros and apply variable-specific recodings and scale factors.
3. **Decode structured values:** convert coded frequencies, durations and dates; preserve special responses through separate indicators.
4. **Encode and impute:** encode binary and nominal categories, pool categories observed fewer than 50 times, median-impute numerical variables and add missingness indicators.
5. **Standardize:** center and scale every encoded column, including binary indicators.
6. **Fit and classify:** train logistic regression and select a decision threshold maximizing training F1.

During cross-validation, preprocessing statistics, interaction choices, model parameters and thresholds are fitted exclusively on the training portion of each fold and then applied to the held-out observations.

## Final model configuration

| Setting | Value |
| --- | --- |
| Features | Cumulative groups 1–5; 691 encoded columns when fitted on all training observations |
| Model | Logistic regression |
| Optimizer | Mini-batch stochastic gradient descent |
| Epochs | 30 |
| Batch size | 256 |
| Learning rate | `0.05 / sqrt(epoch + 1)` |
| L2 penalty | `lambda = 0`; experiments use the convention `lambda * ||w||²` |
| Initialization | Zero coefficients and intercept |
| Random seed | 38 |
| Decision threshold | Recomputed to maximize training F1; approximately 0.208 in the final run |

## Reproduce the experiments

```sh
python cv_experiments.py
```

The script runs stratified five-fold CV for cumulative feature groups, regularization strengths, threshold selection, interaction terms, squared age, alternative recodings and baseline models. Summaries are written to:

```text
experiments/results/cv_results.json
```

The complete experiment suite can take around 1.5 hours, depending on the machine.

## Tests

```sh
python -m unittest discover -s tests -v
```

The repository contains 13 tests covering survey-code cleaning and decoding, encoding, metrics, fold construction, threshold isolation, logistic optimization and the linear SVM.

## Repository structure

```text
implementations.py    Six required regression and optimization methods
run.py                Final training and submission generation
cv_experiments.py     Cross-validation experiments and baselines
helpers.py            Dataset loading and submission utilities
src/
  preprocessing.py    Feature selection and fitted preprocessing pipeline
  missing.py          Survey-code cleaning and median imputation
  brfss.py            BRFSS-specific decoding
  encoding.py         Categorical and numerical feature encoding
  logistic.py         Mini-batch logistic regression
  svm.py              Class-weighted linear SVM
  validation.py       Stratified folds and fold evaluation
  metrics.py          F1, accuracy, threshold selection and ROC AUC
  config.py           Paths and reproducibility settings
notebooks/            Exploratory data analysis
tests/                Repository tests
data/metadata/        Variable definitions and preprocessing rules
```

## Scope and limitations

The target is a **previously reported diagnosis**, rather than future disease onset. The data are self-reported and cross-sectional, so associations may reflect confounding or consequences of disease. Feature relevance groups encode prior judgments.

Thresholds are selected on in-sample training scores. The same CV folds also guide model selection, so local performance estimates may be optimistic. Nested validation and separate preprocessing ablations would strengthen the evaluation. This is an educational classification project, not a clinically validated decision system.

## Authors

- Andreas Caeyman
- Andrea Munno
- Lorenzo Redighieri

Developed for **Machine Learning (CS-433), EPFL — Project 1**.

## Data source

[CDC — 2015 BRFSS Survey Data and Documentation](https://www.cdc.gov/brfss/annual_data/annual_2015.html). Experiments use the course-provided subset and split.

## License

MIT. See [LICENSE](LICENSE).
