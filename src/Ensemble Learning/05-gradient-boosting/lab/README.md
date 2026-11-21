# Gradient Boosting Lab — run it, look at it, understand it

This folder is the hands-on half of the gradient-boosting notes. The theory is in [`../README.md`](../README.md);
every claim made there is shown here with code, numbers and plots — once with scikit-learn, and once rebuilt from
scratch in NumPy and checked against scikit-learn to floating-point precision.

## What is where

| File | What it is |
|---|---|
| [`gradient_boosting_lab.ipynb`](gradient_boosting_lab.ipynb) | **The library lab.** 18 labs in 3 parts (how it works / why it is useful / where it fails), using scikit-learn. Already executed: you can read every result on GitHub without installing anything. It also generates the two datasets below. |
| [`gradient_boosting_from_scratch.ipynb`](gradient_boosting_from_scratch.ipynb) | **The from-scratch notebook.** Gradient boosting written with NumPy only (tree, losses, boosting loop, binary and multi-class, subsampling, early stopping), each step checked against the README's hand examples, with the implementation hurdles demonstrated, and a final comparison with scikit-learn on both CSVs. |
| [`data/good_fit.csv`](data/good_fit.csv) | Friedman #1: 2,000 rows, features `x1`–`x10`, target `y` — non-linear effects and an interaction (gradient boosting's home ground). |
| [`data/bad_fit.csv`](data/bad_fit.csv) | 480 months (`t`, `month`, `y`) with an upward trend and a yearly season — the task is to predict the future, which trees cannot do. |
| [`requirements.txt`](requirements.txt) | The Python packages the notebooks need. |
| `README.md` | This guide. |

All data is generated inside the library notebook (no internet needed); the from-scratch notebook reads the two CSVs,
so **run `gradient_boosting_lab.ipynb` first** if the `data/` files are missing. Results (printed numbers and plots)
appear directly **under the code cell that produced them**.

## How to run it

From the repository root:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r "src/Ensemble Learning/05-gradient-boosting/lab/requirements.txt"
```

Then, in **VS Code** (with the Python and Jupyter extensions installed):

1. Open `src/Ensemble Learning/05-gradient-boosting/lab/gradient_boosting_lab.ipynb`.
2. Click **Select Kernel** (top right) → **Python Environments** → the `.venv` you just created.
3. Run cells one by one with **Shift + Enter**, or everything with **Run All** (about 70 seconds in total on an M4
   MacBook Pro; Lab 18 alone takes about half of that).
4. Then open `gradient_boosting_from_scratch.ipynb` the same way and **Run All** (about 25 seconds).

* **Always run the first code cell (imports) first.** After that, labs can be run in order; a lab only reuses
  variables from earlier cells, so top-to-bottom always works.
* Prefer the browser instead of VS Code? `pip install jupyterlab`, then
  `jupyter lab "src/Ensemble Learning/05-gradient-boosting/lab/gradient_boosting_lab.ipynb"`.
* Just want to read? Open the notebooks on GitHub; the outputs are saved in them.

## How each lab is laid out

Every lab (and every from-scratch section) has the same parts, so you always know where to look:

1. **Heading + Goal** (markdown) — which question from the README this lab answers.
2. **Code cell(s)** — short, commented, nothing hidden in helper files. Lines marked `# <- try changing this` are
   good knobs to experiment with.
3. **Output** — printed numbers and/or plots directly underneath.
4. **"How to read this" box** (a quoted markdown block) — what you should notice in the output and what it proves.
   Try to guess before you read it.

In the from-scratch notebook, boxes titled **Implementation hurdle** show a problem you only meet when writing the
code yourself (and demonstrate it). Both notebooks end with a **Summary** table.

## Which lab answers which question

**`gradient_boosting_lab.ipynb`**

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **Part A — How it works** | | | |
| 1 | Does the hand example really give splits 3.5 / 4.5 / 5.5 and MSE 10.667 → 0.814? | 5.1 | printed rounds; four step-function panels |
| 2 | Are the lecture screenshots right? | 5.2 | the reprinted table; tree-1 leaves 1.6 / −2.667 / 0.178; tree-2 leaves 0.577 / −2.991 / 1.050; query p = 0.249 |
| 3 | Is the residual really a negative gradient? | 2.3, 4.2 | numeric slope = formula for 4 losses; loss and pseudo-residual curves |
| 4 | How does the fit grow round by round? | 3 | curves after 1, 5, 20, 200 trees; training MSE crossing the noise level |
| 5 | Binary classification step by step: log-odds and Newton leaves | 4.4 | p(class 1) panels for rounds 0–4; hand loop = scikit-learn; log-loss 0.461943067988696 |
| 6 | K classes: K trees per round? | 4.6 | `estimators_` shapes (50, 3) vs (4, 1); scores sum to 0; (K−1)/K leaf check |
| **Part B — Why it is useful** | | | |
| 7 | Does it beat linear, one tree and a random forest? | 7 | test MSE 2.07 vs 3.65 / 7.06 / 7.35 |
| 8 | How many trees? | 6.4 | training error always falls; validation minimum at 499 trees (depth 3) |
| 9 | Low learning rate → more trees? | 4.5, 6.2 | best number of trees 16 → 54 → 188 → 499 → 617 as η shrinks |
| 10 | Early stopping, and what `validation_fraction` really does | 6.2 | `n_estimators_ = 152`; `validation_fraction` alone changes nothing |
| 11 | Does `subsample` redraw each round? | 4.5, 6.2 | 600 of 1200 rows each round, about half shared; best MSE 1.577 vs 1.837 |
| **Part C — Where it fails** | | | |
| 12 | Can it forecast a trend? (bad-fit data) | 8.1 | flat red forecast; future RMSE 9.52 vs 1.63 (linear) and 1.76 (trend + boosting) |
| 13 | Does it learn the population from a narrow sample? | 8.2 | flat lines outside [π−1, π+1] |
| 14 | Can stumps learn XOR? | 8.3 | accuracy 0.577 (depth 1) vs 0.997 (depth 2) |
| 15 | What do outliers do with squared loss? | 8.4 | spiky red curve; RMSE 1.317 / 0.115 / 0.540 |
| 16 | Noisy labels and too many rounds? | 8.5 | U-shaped validation log-loss (best at 9 trees); early stopping 0.529 vs 1.078 |
| 17 | Linear target; what do the residuals say? | 8.6 | RMSE 6.11 / 10.15 vs 0.20; residuals correlated with features 1, 7, 8, 9 |
| 18 | Cost, large n, tuning sensitivity | 8.7 | time ∝ rounds; ≈ 19 s vs a few seconds at n = 50,000; R² heat map 0.61–0.92 |

**`gradient_boosting_from_scratch.ipynb`**

| Section | Question | README section | What to look at |
|---|---|---|---|
| 1.1 | How is a split scored, and why is Friedman's criterion the same? | 4.7 | identical thresholds; timing table (≈ 200× at n = 2000) |
| 1.2 | Does our tree equal scikit-learn's? | 2.4 | max difference 0.0 on 100,000 new rows; float32 and tie hurdles |
| 2 | Are the leaf line searches right? | 4.1, 4.3 | leaf value vs brute force (Huber: one-step approximation); which median |
| 3 | Does the loop reproduce the hand example? What if η goes inside the argmin? | 3, 4.5, 5.1 | F = 2.65 … 8.875; "inside" = η 1 (MAE 1.724 vs 1.046) |
| 4 | Binary: log-odds, Newton leaves, stability, the old tutorial's bugs | 4.4, 5.2 | screenshot leaf values; `nan`/`inf` from naive formulas; the 3 bugs triggered and fixed |
| 5 | Multi-class with softmax | 4.6 | stable softmax; probabilities equal to scikit-learn's to 3e-16; class order |
| 6 | Subsampling | 4.5 | same rows as scikit-learn every round; the tied-x exception |
| 7 | Early stopping | 6.2 | same stopping round (168 and 220) |
| 8 | Histogram binning | 8.7 | same threshold, much faster per node |
| 9 | Own implementation vs scikit-learn on both CSVs | all | metric table; largest difference 7e-14 |

## How to interpret the results — quick guide

* **MSE / RMSE** (regression) — average squared error / its square root (typical error in the units of y). Lower is
  better; the good-fit data has noise variance 1, so a test MSE of 1 would be perfect.
* **R²** — share of the variance of y explained (1 = perfect, 0 = no better than predicting the mean, negative = worse).
* **Log-loss** (classification) — how much probability the model gave to the true class, averaged on a log scale.
  Lower is better; it punishes confident mistakes, so it reveals over-confidence that accuracy hides (Lab 16).
* **Accuracy** — share of correct class predictions; 0.5 is coin-flip for two balanced classes.
* **Training vs validation curves** — training error always falls with more trees; the **validation** curve's
  minimum is the number of trees to keep. The test set is used once, at the end.
* **`estimators_` shape** — (rounds, 1) for regression and binary, (rounds, K) for K classes.
* **max |difference|** (from-scratch notebook) — the largest disagreement between our predictions/probabilities and
  scikit-learn's; ~1e-14 means "identical up to the last digits of floating-point arithmetic".

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions and is fine;
the timings depend on your machine. The from-scratch notebook's *exact* agreement was verified with scikit-learn 1.9.1
(older versions build the trees with a slightly different split criterion, so tiny differences can appear there). If a
lab errors, re-run the imports cell and the cells above it — most errors mean a variable from an earlier cell is
missing.

## <font color="red">TODO: several targets at once (multi-output regression, multi-label classification)</font>

Not done yet. This lab should show how gradient boosting handles a target with several columns. In scikit-learn
1.9.1, `GradientBoosting*` and `HistGradientBoosting*` reject a 2-D target. Sources are in the README's
Further reading ("Several targets at once").
The problem and the library limits are in [section 14 of the folder README](../../README.md#14-several-targets-at-once-multi-output-regression-and-multi-label-classification).

- [ ] Show the error, then `MultiOutputRegressor(GradientBoostingRegressor())`: one boosted model per target.
- [ ] From scratch, the two designs:
  - one tree per target per round, each fitted to that target's residuals;
  - one multi-output tree per round, fitted to the (n, k) residual matrix (scikit-learn's `DecisionTreeRegressor`
    accepts it).

  Compare per-target error, number of trees and time on related vs unrelated targets.
- [ ] Multi-label with log-loss: the residuals are y − p per label. Build the same two designs.
- [ ] Optional, if installed: CatBoost `MultiRMSE` / `MultiLogloss` (one model for all targets). LightGBM has no native
  multi-output support.
