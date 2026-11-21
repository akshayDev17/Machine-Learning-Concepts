# Random Forest Lab — run it, look at it, understand it

This folder is the hands-on half of the random-forest notes. The theory is in [`../README.md`](../README.md); every
claim made there is shown here with code, numbers and plots — once with scikit-learn, once built from scratch in NumPy
and checked against scikit-learn.

## What is where

| File | What it is |
|---|---|
| [`random_forest_lab.ipynb`](random_forest_lab.ipynb) | **The library lab.** 16 labs in 3 parts (how it works, why it is useful, where it fails), using scikit-learn. Already executed: you can read every result on GitHub without installing anything. |
| [`random_forest_from_scratch.ipynb`](random_forest_from_scratch.ipynb) | **The from-scratch lab.** A random forest written with NumPy functions (impurity, split search, per-split feature sampling, tree, bootstrap, OOB, MDI), compared tree by tree with scikit-learn. Also executed. |
| [`data/good_fit.csv`](data/good_fit.csv) | "31 thermometers": 1500 rows, 31 noisy measurements of one hidden signal, regression target. The data a random forest is made for. Written by the first notebook. |
| [`data/bad_fit.csv`](data/bad_fit.csv) | "3 needles in 150": 800 rows, 150 features of which only 3 matter, binary target. The data where the default forest struggles. Written by the first notebook. |
| [`requirements.txt`](requirements.txt) | The Python packages the notebooks need. |
| `README.md` | This guide. |

All data are generated inside `random_forest_lab.ipynb` (the generator functions are visible in its "two datasets"
section) — no internet needed. Results (printed numbers and plots) appear directly **under the code cell that
produced them**. Run `random_forest_lab.ipynb` first if the `data/` folder is missing; the from-scratch notebook
reads the two CSVs.

## How to run it

From the repository root:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r "src/Ensemble Learning/03-random-forest/lab/requirements.txt"
```

Then, in **VS Code** (with the Python and Jupyter extensions installed):

1. Open `src/Ensemble Learning/03-random-forest/lab/random_forest_lab.ipynb`.
2. Click **Select Kernel** (top right) → **Python Environments** → the `.venv` you just created.
3. Run cells one by one with **Shift + Enter**, or everything with **Run All** — about **1 minute** for
   `random_forest_lab.ipynb` and about **30 seconds** for `random_forest_from_scratch.ipynb` (measured on an
   Apple-silicon MacBook Pro).

* **Always run the first code cell (imports) first.** After that, labs can be run in order; a lab only reuses
  variables from earlier cells, so top-to-bottom always works.
* Prefer the browser instead of VS Code? `pip install jupyterlab`, then
  `jupyter lab "src/Ensemble Learning/03-random-forest/lab/random_forest_lab.ipynb"`.
* Just want to read? Open the notebooks on GitHub; the outputs are saved in them.

## How each lab is laid out

Every lab has the same four parts, so you always know where to look:

1. **Heading + Goal** (markdown) — which question from the README this lab answers.
2. **Code cell(s)** — short, commented, nothing hidden in helper files. Lines marked `# <- try changing this` are
   good knobs to experiment with.
3. **Output** — printed numbers and/or plots directly underneath.
4. **"How to read this" box** (a quoted markdown block) — what you should notice in the output and what it proves.
   Try to guess before you read it.

The from-scratch notebook follows the theory in order: each section restates one step (with the README section),
implements it as a small function, and checks it — on the README's 8-row example first. Boxes titled
**"Implementation hurdle"** explain a real difficulty and demonstrate it (a timing, a failing first attempt, …).
Both notebooks end with a **Summary** table.

## Which lab answers which question

**`random_forest_lab.ipynb`**

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **Part A — How it works** | | | |
| 1 | Do the hand-worked numbers come out (3.208 / 1.535; MDI 0.6 / 0.4)? | 5 | printed formula vs simulation; scikit-learn's `feature_importances_` |
| 2 | What is stored in a fitted tree, and how is MDI computed correctly? | 4.6, 6.3 | `tree_` arrays; correct MDI = scikit-learn to 2e-17, old version off by 0.08 |
| 3 | How does per-split sampling differ from bagging and from per-tree sampling? Is the default regressor a forest? | 4.2, 4.5 | "root split on x0" column; test MSE 0.299 vs 0.343 vs 0.352; `True` for default = bagging |
| 4 | How does `max_features` trade correlation ρ against single-tree error σ²? | 4.1, 4.2 | table where formula = measured MSE; three curves; two heatmaps |
| **Part B — Why it is useful** | | | |
| 5 | Does a forest beat one tree and bagging? Are its trees "weak"? | 1, 7 | test MSE 0.703 / 0.342 / 0.299; histogram of single-tree errors |
| 6 | Does averaging reduce bias or variance? | 2.4, 4.3 | bias² and variance columns over 30 training sets |
| 7 | Can more trees over-fit? What does tree size do? | 4.4 | flat error-vs-trees curve; train and test MSE vs `min_samples_leaf` |
| 8 | Is the OOB score trustworthy? | 4.4 | OOB 0.935 vs CV 0.933 vs test 0.940 |
| 9 | How do I tune without touching the test set? | 6.5 | CV and OOB choose the same setting; test used once |
| 10 | Is the median of the trees better than the mean? | 9 | mean vs median columns, clean vs corrupted targets |
| **Part C — Where it fails** | | | |
| 11 | What if only 3 of 150 features matter? | 8.1 | 22.3 % chance; accuracy vs `max_features`; table of fixes |
| 12 | What if the target is a straight line? | 8.2 | MSE 7382 vs 0.01; staircase plot |
| 13 | Can a forest extrapolate? | 8.3 | flat prediction beyond x = 5 |
| 14 | Can I trust `feature_importances_`? | 4.6, 8.4 | noise feature ranked first by MDI, ≈ 0 by permutation |
| 15 | What happens with correlated features? | 8.5 | 0.829 vs 0.168; single vs group permutation |
| 16 | How big and slow are deep forests? | 6.4, 8.6 | MB and ms per `min_samples_leaf` |

**`random_forest_from_scratch.ipynb`**

| Section | Question | README section | What to look at |
|---|---|---|---|
| 1 | What exactly does the model see? | — | `float32` conversion; target is a multiple of 1/64 |
| 2 | How is a node's impurity computed? | 2.2 | 0.46875 and 0.375 |
| 3 | How is the best split found, and why is it fast? | 2.3, 5 | the README's two splits; naive vs sort + cumsum timing |
| 4 | What if all drawn features are constant? | 3 | never-empty candidate list; 3.85 draws on average |
| 5 | How is one tree grown? Why weights, not copies? | 3, 6.4 | the README's tree; 191 vs 315 nodes with `min_samples_leaf=5` |
| 6 | Bootstrap, average, OOB | 2.6, 4.4 | 63.2 % distinct rows; OOB R² 0.932 |
| 7 | MDI, correctly and as in the old notes | 4.6, 5 | 0.6 / 0.4 vs 0.444 / 0.556 |
| 8 | How do cost and memory scale? | 6.4 | split time vs m; nodes ≈ 2 × distinct rows |
| 9 | Do we grow *exactly* scikit-learn's trees? | 4.2 | 0/100 identical (ties!) → replay scikit-learn's feature order → 100/100, difference 0 |
| 10 | Is our random forest statistically the same as scikit-learn's? | 3, 4.4 | metrics next to a "scikit-learn with another seed" yardstick; replayed: 100/100 and 95/100 identical |

## How to interpret the results — quick guide

* **Test MSE** (regression) — average squared error on the held-out rows; lower is better. On the good-fit data about
  0.18 of it is pure noise that no model can remove, so 0.30 is close to the best possible.
* **R²** — 1 − MSE / variance of the target: 1 = perfect, 0 = no better than predicting the average.
* **Accuracy** (classification) — share of correctly classified rows; 0.5 is a coin flip on the bad-fit data.
* **OOB score** — R² or accuracy on the training rows, each predicted only by the trees that did not see it. Treat it
  like a cross-validation score.
* **ρ and σ²** (Lab 4) — how alike two trees' errors are, and how big one tree's error is; the forest's error is about
  ρσ².
* **MDI (`feature_importances_`)** — sums to 1; computed on training data and biased (Lab 14). **Permutation
  importance** — drop in the held-out score when a feature is shuffled; ≈ 0 means "not used", negative values are
  noise.
* **"identical trees k/100"** (from-scratch notebook) — same features and thresholds at every node; "max |prediction
  diff| 0" means bit-for-bit equal predictions.

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions or the
machine and is fine; timings will differ more. If a lab errors, re-run the imports cell and the cells above it — most
errors mean a variable from an earlier cell is missing.

## <font color="red">TODO: several targets at once (multi-output regression, multi-label classification)</font>

Not done yet. This lab should show how a random forest handles a target with several columns: one forest for all
targets by default, not one per target. Sources are in the README's Further reading ("Several targets at once").
The problem and the library limits are in [section 14 of the folder README](../../README.md#14-several-targets-at-once-multi-output-regression-and-multi-label-classification).

- [ ] One forest on a 2-D target vs one forest per target (`MultiOutputRegressor`). Report per-target test error,
  number of trees and nodes, and fit time, on three kinds of targets:
  - targets that share one signal;
  - targets that use the same features through different functions;
  - targets that depend on different features.
- [ ] Target scale: multiply one target by 100 and watch it take over the shared splits, then standardise the targets
  and compare.
- [ ] Multi-label: `predict_proba` returns a list of L arrays. Show per-label F1 for rare vs common labels, since a rare
  label has a small Gini and little say in the averaged split score.
- [ ] From scratch: score a split by the per-target squared-error drops added up, and check that it picks
  scikit-learn's split.
