# EPFL Machine Learning Project 1

Prediction of myocardial infarction or coronary heart disease (MICHD) from the
BRFSS 2015 survey. Only Python 3 and NumPy are needed.

## Team
- Andrea Munno
- Lorenzo Redighieri
- Andreas Caeyman

## Data

The dataset is not in the repository. Put the three CSV files directly in `data/raw/`:

```text
data/
  raw/        x_train.csv, y_train.csv, x_test.csv
  metadata/   variables_BRFSS_2015.csv   (in the repository)
```

The first run caches the CSVs in `data/raw/cache.npz` (about 100 s).
If the run is interrupted during this step, delete `cache.npz` and start again.

## Usage

```sh
python run.py              # final model -> data/submissions/submission.csv
python cv_experiments.py   # all cross-validation experiments of the report (~1.5 h)
python -m unittest discover -s tests
```

## Structure

| File | Content |
| --- | --- |
| `implementations.py` | The six required methods |
| `run.py` | Trains the final model and writes the submission |
| `cv_experiments.py` | 5-fold CV: feature groups, threshold, lambda, interactions, baselines |
| `helpers.py` | Course helpers (`load_csv_data`, `create_csv_submission`) |
| `src/preprocessing.py` | Data loading, feature selection, full pipeline |
| `src/missing.py` | Metadata cleaning and median imputation |
| `src/brfss.py` | BRFSS storage formats (prefix codes, dates, special answers) |
| `src/encoding.py` | Binary / one-hot / numeric encoding |
| `src/logistic.py` | Logistic regression with mini-batch SGD (final model) |
| `src/svm.py` | Linear SVM (baseline) |
| `src/metrics.py` | F1, accuracy, best threshold, AUC |
| `src/validation.py` | Stratified k-fold cross-validation |
| `data/metadata/variables_BRFSS_2015.csv` | Per-variable type, missing/zero codes, recodings, group and use |

## Preprocessing

```
select_features -> clean -> decode -> encode -> standardize
```

1. **select_features**: variables with `use == include` and `group <= max_group` (groups 1 to 5).
2. **clean**: per variable, from the metadata: `na_values` to NaN, `zero_values` to 0, `recode`, `scale`.
3. **decode**: prefix-coded frequencies and durations to one unit, MMYYYY dates to a month index,
   special answers (e.g. `LASTSMK2 = 8`, never smoked regularly) to a separate 0/1 flag.
4. **encode**: binary variables to one 0/1 column; nominal variables to one-hot without the most
   frequent level, rare levels (< 50 in training) pooled in `__other`; ordinal and numeric variables
   median-imputed; a `__missing` flag for every column with missing values.
5. **standardize**: every encoded column to mean 0 and standard deviation 1.

Everything learned from data (levels, medians, means, standard deviations) is fitted on the
training rows only, with `fit_preprocess`, and applied to any rows with `apply_preprocess`.
In cross-validation this happens inside each fold.

## Final model

Feature groups 1-5 (about 690 encoded columns), logistic regression trained with mini-batch SGD
(batch 256, 30 epochs, learning rate 0.05 / sqrt(epoch + 1), lambda = 0), decision threshold
that maximizes F1 on the training predictions. 5-fold CV: F1 0.430 ± 0.004, AUC 0.861.
