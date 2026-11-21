# Bagging Lab — run it, look at it, understand it

This folder is the hands-on half of the bagging notes. The theory is in [`../README.md`](../README.md); every claim
made there is shown here with code, numbers and plots — once with scikit-learn, and once rebuilt from scratch in NumPy
and checked against scikit-learn.

## What is where

| File | What it is |
|---|---|
| [`bagging_lab.ipynb`](bagging_lab.ipynb) | **The library lab.** 19 labs in 3 parts, using scikit-learn's `BaggingRegressor` / `BaggingClassifier`. Already executed: you can read every result on GitHub without installing anything. |
| [`bagging_from_scratch.ipynb`](bagging_from_scratch.ipynb) | **Bagging in plain NumPy:** a weighted CART tree (regression and Gini), bootstrap counts, averaging, OOB — reproducing scikit-learn tree by tree. Already executed. |
| `data/good_fit.csv` | Wiggly curve + noise, 300 rows (`x1, y`): the data bagging is good at. Written by `bagging_lab.ipynb`. |
| `data/bad_fit.csv` | Straight line + noise, 200 rows (`x1, y`); the notebooks train on `x1 < 6` and test on `x1 ≥ 6` (extrapolation). Written by `bagging_lab.ipynb`. |
| [`requirements.txt`](requirements.txt) | The Python packages the notebooks need. |
| `README.md` | This guide. |

All data is generated inside the library notebook (or comes from scikit-learn's `make_*` functions) — no internet
needed. The from-scratch notebook reads the two CSVs, so run `bagging_lab.ipynb` first if you delete them. All results
(printed numbers and plots) appear directly **under the code cell that produced them**.

## How to run it

From the repository root:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r "src/Ensemble Learning/02-bagging/lab/requirements.txt"
```

Then, in **VS Code** (with the Python and Jupyter extensions installed):

1. Open `src/Ensemble Learning/02-bagging/lab/bagging_lab.ipynb` (or `bagging_from_scratch.ipynb`).
2. Click **Select Kernel** (top right) → **Python Environments** → the `.venv` you just created.
3. Run cells one by one with **Shift + Enter**, or everything with **Run All**. Measured runtime on an Apple-silicon
   MacBook Pro: about **21 s** for `bagging_lab.ipynb` and about **7 s** for `bagging_from_scratch.ipynb`.

* **Always run the first code cell (imports) first.** After that, run top-to-bottom: a lab only reuses variables from
  earlier labs.
* Prefer the browser instead of VS Code? `pip install jupyterlab`, then
  `jupyter lab "src/Ensemble Learning/02-bagging/lab/bagging_lab.ipynb"`.
* Just want to read? Open the notebooks on GitHub; the outputs are saved in them.

## How each lab is laid out

Every lab (and every from-scratch section) has the same four parts, so you always know where to look:

1. **Heading + Goal** (markdown) — which question from the README this lab answers.
2. **Code cell(s)** — short, commented, nothing hidden in helper files. Lines marked `# <- try changing this` are
   good knobs to experiment with.
3. **Output** — printed numbers and/or plots directly underneath.
4. **"How to read this" box** (a quoted markdown block) — what you should notice in the output and what it proves.
   Try to guess before you read it.

The from-scratch notebook adds **Implementation hurdle** boxes: real problems you hit when writing bagging yourself,
each demonstrated with a timing, a counter-example or a failing comparison. Both notebooks end with a **Summary** table.

## Which lab answers which question

**`bagging_lab.ipynb`**

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **Part A — How bagging works** | | | |
| 1 | Does the hand-worked example really give 3.5 at x = 3? | 5 | the SSE table per sample; stump thresholds 2.5 / 3.5 / 4.0; OOB predictions |
| 2 | Why ≈ 63.2 % distinct rows? What does `estimators_samples_` hold? | 2.3, 4.1 | formula vs simulation; 750 draws but ≈ 473 distinct; 87.5 % left out with 100 draws |
| 3 | What does "a deep tree has high variance" look like? | 1, 2.2 | three different step curves; the blue disagreement band; the smooth red average |
| 4 | How are the trees' predictions combined? | 3, 4.7 | difference 0.0 for mean vs `predict`; soft vs hard votes; SVC vote fractions |
| 5 | Why can't more trees remove all variance? | 4.2, 4.3 | curves flattening onto the dotted floors ρσ² |
| 6 | What exactly is an OOB prediction? | 2.5, 4.6 | difference 0.0 vs `oob_prediction_`; the 3-tree warning and the 0-filled rows |
| **Part B — Why bagging is useful** | | | |
| 7 | Does bagging cut variance and leave bias alone? | 2.2, 4.3, 4.4 | the bias²/variance table; σ² = 0.127, ρ = 0.454, floor 0.058 |
| 8 | How many trees? Can too many over-fit? | 6.2, 6.4 | OOB and test curves flattening; chosen B = 96 |
| 9 | Is OOB as good as cross-validation? | 4.6, 6.4 | OOB 0.153 vs CV 0.164 vs test 0.185; one fit vs five |
| 10 | Bagging or pasting? How many rows per tree? | 3, 6.2 | CV curves vs `max_samples`; pasting all rows = one tree |
| 11 | Is each bagged tree better than one full tree? | 4.4, 7 | accuracies 0.777 / 0.828 / 0.848; the correct `Counter` reading; per-class histograms |
| 12 | How correlated are the trees' mistakes? | 4.3, 6.5 | the two heat-maps; mean correlation 0.40 / 0.75; the MSE-vs-correlation pairs |
| 13 | Why do 750-row trees beat 100-row trees? What is `oob_decision_function_`? | 4.1, 4.6 | distinct rows 473 vs 94; OOB 0.849 vs 0.817 |
| 14 | Random subspaces vs random patches? | 3, 8.6 | CV MSE per configuration; duplicated features `[0 0 2 2 …]` |
| 15 | Which features matter to a bag? | 6.6 | averaged importances vs permutation importance; noise features ≈ 0 |
| **Part C — Where bagging fails** | | | |
| 16 | What if the base model is stable or biased? | 4.5, 8.1, 8.2 | bagged line = line; bagged stumps still a staircase |
| 17 | What happens outside the training range? (bad-fit data) | 8.3 | the flat orange line after x = 6; MSE 4.73 vs 0.300 |
| 18 | Does bagging help on tiny datasets? | 8.4 | bag/tree error ratio 0.83 at n = 10 vs 0.48 at n = 500; overlap ≈ 0.40 for all n |
| 19 | What does it cost? | 8.7 | time and nodes growing with B; one tree's top levels |

**`bagging_from_scratch.ipynb`**

| Section | Question | README section | What to look at |
|---|---|---|---|
| 1 | How does a tree find its best split, fast? | 2.1 | naive vs sorted timing curve; same thresholds; SSE = Σwy² − proxy |
| 2 | Is our tree identical to scikit-learn's? | 2.1 | max difference 0.0; tied x, midpoint and float32 demos |
| 3 | Bootstrap counts vs duplicated rows? | 2.3, 4.1 | 0.6335 distinct; identical trees with `min_samples_leaf=1`, different with 5 (in both libraries) |
| 4 | The hand example in NumPy? | 5 | the same thresholds, averages and OOB predictions |
| 5 | Our bag vs `BaggingRegressor`, tree by tree? | 3 | per-tree and ensemble difference 0.0 |
| 6 | Our OOB vs `oob_score_`? Rows without OOB trees? | 4.6 | R² equal to 12 decimals; 0.4582 vs 0.6451 |
| 7 | Classification with a Gini tree? | 2.1, 4.7 | exact in 1-D; 197 feature ties in 2-D, same predicted classes |
| 8 | How does the cost grow with B? | 8.7 | nodes and time roughly ∝ B |
| 9 | Both CSVs, side by side with scikit-learn? | 8.3 | identical metrics with the same draws; close metrics with our own draws |

## How to interpret the results — quick guide

* **MSE** (mean squared error, regression) — the average squared gap between prediction and truth; lower is better.
  On the good-fit data the noise alone gives 0.35² = 0.1225, so no model can average much below that on new data.
* **R²** (`oob_score_` for regressors) — 1 − MSE / variance of y: 1 is perfect, 0 is "always predict the mean",
  negative is worse than that.
* **Accuracy** (`oob_score_` for classifiers) — share of rows classified correctly; with 3 balanced classes 0.33 is
  guessing.
* **OOB vs test** — both estimate error on new data; they use different rows, so expect them to differ by about as
  much as cross-validation folds differ from each other (Lab 9).
* **Bias² / variance** (Labs 7, 18) — measured by refitting on many fresh training sets; tiny bias² values (≈ 0.002)
  are within measurement noise.
* **Correlation heat-maps** (Lab 12) — 1 on the diagonal (each tree with itself); off-diagonal values near 1 would mean
  duplicate trees; values well above 0 mean the trees still share most of their mistakes.
* **`estimators_samples_`** — the row numbers *drawn* for each tree (with repeats). `np.unique` gives the distinct rows,
  `np.bincount(..., minlength=n)` the counts the tree was trained with.

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions and is fine
(timings always differ). If a lab errors, re-run the imports cell and the labs above it — most errors mean a variable
from an earlier lab is missing.

## <font color="red">TODO: several targets at once (multi-output regression, multi-label classification)</font>

Not done yet. This lab should show how bagging handles a target with several columns. Sources are in the README's
Further reading ("Several targets at once").
The problem and the library limits are in [section 14 of the folder README](../../README.md#14-several-targets-at-once-multi-output-regression-and-multi-label-classification).

- [ ] `BaggingRegressor` on a 2-D target builds **one** ensemble whose trees predict every target (leaf values of
  shape (nodes, k, 1)). Compare it with `MultiOutputRegressor(BaggingRegressor())` (k ensembles) on:
  - per-target test error;
  - number of trees, nodes and fit time;
  - targets that share a signal vs targets that depend on different features.
- [ ] `BaggingClassifier` rejects a 2-D target (`y should be a 1d array`). Show `MultiOutputClassifier(BaggingClassifier())`,
  and in the from-scratch notebook bag multi-output `DecisionTreeClassifier`s by hand (they accept a 2-D target).
- [ ] `BaggingRegressor(oob_score=True)` crashes on a 2-D target in scikit-learn 1.9.1 (`operands could not be broadcast
  together`, from `_set_oob_score`); `RandomForestRegressor(oob_score=True)` does not. Show the error and compute the
  per-target OOB score by hand from `estimators_samples_`.
