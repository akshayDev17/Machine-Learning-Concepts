# Feature-engineering pipeline — one script per step, switched on and off in `config.yml`

The notebook in [`../lab/`](../lab/README.md) teaches each step on its own. This folder runs **all** of them as one
pipeline on the customer-churn table, so you can switch steps on and off and see what each is worth. The theory for
every step is in [`../README.md`](../README.md) (sections 3–21).

## What is where

| File | What it does | Default | Main parameters in `config.yml` |
|---|---|---|---|
| `01_load_data.py` | read the CSV | **required** | `path`, `target` |
| `02_split_data.py` | stratified train/test split + CV folds on train | **required** | `test_size`, `stratify`, `cv_folds` |
| `03_drop_useless_columns.py` | drop ID-like, constant and duplicate columns | on | `id_unique_ratio`, `drop_constant`, `drop_duplicates` |
| `04_remove_leakage.py` | drop listed leaky columns; flag suspiciously predictive ones | on | `columns`, `warn_auc_above` |
| `05_fix_data_types.py` | money text → numbers, yes/no → 1/0, tidy spellings, impossible values → missing | on | `money_columns`, `yes_no_columns`, `tidy_columns`, `valid_ranges` |
| `06_make_date_features.py` | dates → days since, month (and weekday) as sin/cos | on | `date_columns`, `reference_date`, `cyclical_month`, `cyclical_weekday` |
| `07_make_domain_features.py` | ratios, differences, group statistics, keyword flags from text | on | `ratios`, `differences`, `group_stats`, `keyword_flags` |
| `08_impute_missing.py` | fill gaps (median / "missing"), add "was missing" flags | on | `numeric_strategy`, `text_strategy`, `add_missing_flags`, `flag_min_rate` |
| `09_cap_outliers.py` | clip continuous columns to learned limits | on | `method` (quantile / iqr), `lower_quantile`, `upper_quantile`, `iqr_factor` |
| `10_transform_skewed.py` | un-skew long-tailed columns | on | `method` (yeo-johnson / box-cox / log1p), `skew_threshold` |
| `11_group_rare_categories.py` | rare categories → "other" | on | `min_share`, `other_label` |
| `12_encode_categoricals.py` | ordinal / one-hot / target or frequency encoding | on | `ordinal`, `max_onehot_levels`, `high_cardinality` (target / frequency / onehot), `target_cv_folds` |
| `13_add_interactions.py` | products of column pairs, centred squares | on | `products`, `squares` |
| `14_bin_numeric.py` | quantile / equal-width bins as 0/1 columns | on | `columns`, `n_bins`, `strategy`, `keep_original` |
| `15_drop_correlated.py` | drop near-copies (\|r\| above a threshold) | on | `threshold` |
| `16_reduce_dimensions.py` | compress a block of columns (the survey) with PCA or LDA | on | `method` (pca / lda), `columns_prefix`, `n_components` |
| `17_select_features.py` | keep useful columns: lasso, mutual information or RFE | on | `method` (l1 / mutual_info / rfe), `C`, `k` |
| `18_scale_numeric.py` | standard / robust / min-max scaling of continuous columns | on | `method` |
| `19_train_and_evaluate.py` | train the model, score it | **required** | `model` (logistic_regression / knn / random_forest / hist_gradient_boosting), `C`, `n_neighbors` |
| `run_pipeline.py` | the runner: reads the config, runs the enabled stages, prints scores | | |
| `column_kinds.py` | three tiny helpers that sort columns into text / numeric / continuous | | |
| `config.yml` | the control panel: stage order, on/off switches, parameters | | |

## How to run it (M4 Mac)

From the repository root, once (keep the virtual environment outside the repository; on this machine pyenv's
Python 3.13.3 is broken, so 3.12.8 is used explicitly — any other working Python ≥ 3.10 is fine):

```bash
~/.pyenv/versions/3.12.8/bin/python -m venv ~/venvs/feature-eng
```

```bash
source ~/venvs/feature-eng/bin/activate
```

```bash
pip install -r src/Feature-Eng/lab/requirements.txt
```

Then run the pipeline (about 4 seconds):

```bash
python src/Feature-Eng/scripts/run_pipeline.py --config src/Feature-Eng/scripts/config.yml
```

The output lists every stage, what it learned on the training rows, and the scores (shortened here):

```text
Stages (top to bottom):
  [on ] load_data                (required) 3000 rows x 35 columns
  [on ] split_data               (required) train 2400 / test 600, 5 CV folds on train
  [on ] drop_useless_columns     dropped 3: customer_id (id-like ...), country (constant), monthly_fee (duplicate of monthly_charges)
  [on ] remove_leakage           dropped retention_call_made (AUC 0.93 on its own); suspicious columns still present: none
  ...
  [on ] group_rare_categories    city 119->16 levels
  [on ] encode_categoricals      ordinal: ['plan']; one-hot: ['region', 'payment_method', 'device']; target-encoded: ['city']
  ...
  [on ] train_and_evaluate       (required) logistic_regression on 29 features

CV on train (5 folds): ROC-AUC 0.895 ± 0.011   accuracy 0.807 ± 0.016
Test (used once)    : ROC-AUC 0.893           accuracy 0.788
```

If you see `customer_churn.csv not found`, create the table first: `python src/Feature-Eng/data/make_customer_churn.py`.

## Toggle a stage and compare

**In the config (the builder way):** open `config.yml`, change a stage's `enabled: true` to `enabled: false`, save,
re-run the same command. For example, switching off `make_date_features` and `impute_missing`:

```text
  [off] make_date_features       disabled in config
  [off] impute_missing           disabled in config
  note: filled 575 remaining gaps with 0 (logistic_regression can't handle missing values)
CV on train (5 folds): ROC-AUC 0.851 ± 0.004   accuracy 0.761 ± 0.012
Test (used once)    : ROC-AUC 0.849           accuracy 0.780
```

0.895 → 0.851: those two stages are worth about 0.04 of ROC-AUC. Notice the runner's `note`: with imputation off,
the model still needs complete numbers, so the leftover gaps are filled with 0 and reported. Likewise, with
`encode_categoricals` off the text columns are dropped (and reported), and stages whose input column no longer exists
or isn't numeric skip that feature and say so — every combination of switches runs.

**From the command line (for quick experiments, the config file is not changed):**

```bash
python src/Feature-Eng/scripts/run_pipeline.py --disable make_date_features,impute_missing
```

**Try a different model:** set `model: knn` (or `random_forest`, `hist_gradient_boosting`) under
`train_and_evaluate`. k-NN with all stages scores 0.850; with `scale_numeric` and `transform_skewed` disabled it drops
to 0.680 — distance-based models need scaling.

**Measure every stage at once (ablation):** one run with everything enabled, then one run per optional stage switched
off (about 20 seconds):

```bash
python src/Feature-Eng/scripts/run_pipeline.py --ablation
```

```text
run                                     CV ROC-AUC   change  test ROC-AUC
(all enabled stages)                  0.895 ± 0.011   +0.000         0.893
without remove_leakage                0.984 ± 0.003   +0.089         0.989
without make_date_features            0.872 ± 0.008   -0.023         0.864
without impute_missing                0.881 ± 0.012   -0.014         0.881
without encode_categoricals           0.875 ± 0.005   -0.020         0.869
...
```

Read it like README section 21: negative change = the stage helps; changes within about ±0.01 are noise; and the
leakage stage "costs" 0.089 because the leak it removes made the score fake.

**Save a run summary** (stages, what each did, scores) as JSON: add `--summary-out my_run.json`.

## Required stages

`load_data`, `split_data` and `train_and_evaluate` are marked `required: true`. If you disable one, the runner stops
and explains:

```text
Stage 'split_data' is required and can't be disabled: without a held-out test part and CV folds there is no honest
score; every stage must learn from training rows only. Set it back to `enabled: true`.
```

## How a run avoids leakage

1. Load the table, split off a stratified 20 % test part (locked away), cut the training part into 5 folds.
2. **For every fold:** fit each enabled stage on the fold's training rows, apply it to the fold's validation rows,
   train the model, score. Every stage is re-fitted inside every fold, so nothing learned from validation rows
   (medians, caps, category rates, chosen columns, PCA directions, ...) leaks into the CV score.
3. Fit every stage on the whole training part, apply to the test part, train, score once on test.

## How a stage is written (to add your own)

Every optional stage is a file `NN_<name>.py` with two plain functions — explicit input, explicit output, no globals,
no classes:

```python
def fit_transform(train, target, params):
    """Learn whatever the stage needs from the training rows. Return (new_train, state)."""


def transform(frame, state):
    """Apply what was learned to any table (validation or test rows). Return the new table."""
```

* `train` / `frame` are pandas DataFrames (the target column stays in them); `target` is its name; `params` is the
  stage's `params` block from `config.yml` plus `random_state`.
* `state` is a plain dict of what was learned (fill values, limits, kept columns, a fitted scikit-learn object, ...)
  plus a one-line `summary` string that the runner prints.
* `fit_transform` returns the *transformed training rows*, not just the state, because a few steps must treat
  training rows specially: target encoding gives each training row a value computed from the other folds
  (cross-fitting), never from its own label.

To add a stage: create `20_my_stage.py` (any unused two-digit number) with those two functions, and add
`- name: my_stage`, `enabled: true`, `params: {...}` to `config.yml` at the position where it should run — the order
in the config decides when it runs, the number only keeps the folder tidy. The runner finds the file by its name.
