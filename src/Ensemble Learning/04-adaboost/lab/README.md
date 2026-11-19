# AdaBoost Lab — run it, look at it, understand it

This folder is the hands-on half of the AdaBoost notes. The theory is in [`../README.md`](../README.md); every claim made
there is shown here with code, numbers and plots — once with scikit-learn, and once rebuilt from scratch in NumPy and checked
against scikit-learn number for number.

## What is where

| File | What it is |
|---|---|
| [`adaboost_lab.ipynb`](adaboost_lab.ipynb) | **The library lab.** scikit-learn's `AdaBoostClassifier`, 15 labs in 3 parts (how it works, why it is useful, where it fails). Already executed: you can read every result on GitHub without installing anything. |
| [`adaboost_from_scratch.ipynb`](adaboost_from_scratch.ipynb) | **AdaBoost in NumPy**, one README step per section (weighted stump, textbook AdaBoost, SAMME, decision function), with the implementation hurdles demonstrated and a final exact comparison with scikit-learn on both datasets. Already executed. |
| [`data/good_fit.csv`](data/good_fit.csv) | 1000 points, columns `x1,x2,y`: class 1 inside a circle, no label noise — the data AdaBoost handles well. Written by the lab notebook. |
| [`data/bad_fit.csv`](data/bad_fit.csv) | The same 1000 points with exactly 20 % of the labels flipped — the data where AdaBoost struggles. Written by the lab notebook. |
| [`requirements.txt`](requirements.txt) | The Python packages the notebooks need. |
| `README.md` | This guide. |

All other data is generated inside the notebooks or bundled with scikit-learn (`load_digits`), so no internet is needed, and
all results (printed numbers and plots) appear directly **under the code cell that produced them**.

## How to run it

From the repository root:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r "src/Ensemble Learning/04-adaboost/lab/requirements.txt"
```

Then, in **VS Code** (with the Python and Jupyter extensions installed):

1. Open `src/Ensemble Learning/04-adaboost/lab/adaboost_lab.ipynb` (or `adaboost_from_scratch.ipynb`).
2. Click **Select Kernel** (top right) → **Python Environments** → the `.venv` you just created.
3. Run cells one by one with **Shift + Enter**, or everything with **Run All**. Measured on an M-series MacBook Pro:
   `adaboost_lab.ipynb` about **75 seconds**, `adaboost_from_scratch.ipynb` about **12 seconds**.

* **Always run the first code cell (imports) first.** After that, run top to bottom; a lab only reuses variables from earlier
  cells. The lab notebook (re)writes the two CSVs in `data/`; the from-scratch notebook only reads them (they are committed, so
  it also works on its own).
* Prefer the browser instead of VS Code? `pip install jupyterlab`, then
  `jupyter lab "src/Ensemble Learning/04-adaboost/lab/adaboost_lab.ipynb"`.
* Just want to read? Open the notebooks on GitHub; the outputs are saved in them.

## How each lab is laid out

Every lab has the same four parts, so you always know where to look:

1. **Heading + Goal** (markdown) — which question from the README this lab answers.
2. **Code cell(s)** — short, commented, nothing hidden in helper files (the few helpers are defined in the imports cell).
   Lines marked `# <- try changing this` are good knobs to experiment with.
3. **Output** — printed numbers and/or plots directly underneath.
4. **“How to read this” box** (a quoted markdown block) — what you should notice in the output and what it proves.
   Try to guess before you read it.

The from-scratch notebook uses the same idea per section: the README step and its formula → the code → a **check**
(assertions or a printed comparison with scikit-learn), plus **“Implementation hurdle”** boxes where something non-obvious
had to be solved. Both notebooks end with a **Summary** table.

## Which lab answers which question

**`adaboost_lab.ipynb`**

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **Part A — How AdaBoost works** | | | |
| 1 | Does the hand example really give ε = 1/8, 1/7, 5/24 and get all 8 points right? Why is scikit-learn's α twice as big? | 5, 4.6 | the printed rounds; `estimator_weights_` = ln 7, ln 6, ln(19/5); the 4-panel plot |
| 2 | Where does the weight go? | 3, 4.3 | point sizes; share of weight near the circle 20 % → 74 % (round 300) |
| 3 | How do straight cuts make a circle? | 3, 7 | staircase boundary; test error 0.330 → 0.033 |
| 4 | Is the training error really under ∏ 2√(ε(1−ε))? | 4.4 | bound = exponential loss (gap 2.2e-16); training error 0 from round 446 |
| 5 | Why can test error fall after training error is 0? | 4.5 | test error 0.0663 → 0.0635 after round 169; margin curves moving right at the low end |
| **Part B — Why AdaBoost is useful** | | | |
| 6 | Do boosted stumps beat a single stump / tree / forest? | 7, 6.4 | CV picks 400 rounds; test accuracy 0.670 vs 0.970 |
| 7 | Stumps or deeper trees? | 6.2 | depth 3 learns faster; best CV errors 0.021–0.031 |
| 8 | How do `learning_rate` and `n_estimators` interact? | 6.2 | the CV-error heat map |
| 9 | What changes with three classes? | 4.6 | 396 of 400 rounds with ε > 0.5 still get α > 0; stumps never predict the centre class |
| **Part C — Where AdaBoost fails** | | | |
| 10 | What do flipped labels do? (the bad-fit data) | 8.1 | weight share on flipped points; depth-3 test error rising 0.233 → 0.280; the fixes table |
| 11 | Are outliers a problem? | 8.2 | feature outliers 0.970 vs label outliers 0.920; 21 % of the weight on 10 points |
| 12 | What if the base tree is too strong? | 8.3 | 1 of 50 trees fitted, ε = 0 |
| 13 | What if it is too weak? | 8.4 | the XOR `ValueError`; digits 0.848 (stumps) vs 0.987 (depth 4) |
| 14 | Can I trust `predict_proba`? | 8.5 | the vertical blue reliability curve; probabilities only 0.322–0.609 |
| 15 | Can training use all my cores? | 8.6 | AdaBoost time ∝ rounds; forest roughly 3–4× faster with `n_jobs=-1` |

**`adaboost_from_scratch.ipynb`**

| Section | Question | README section | What to look at |
|---|---|---|---|
| 1 | How does a weighted stump pick its cut, exactly like scikit-learn? | 2.2–2.4 | 50/50 identical stumps vs `DecisionTreeClassifier(max_depth=1)` |
| 1, hurdle 1 | Why sort + cumulative sums? | 2.4 | ≈ 18 ms (naive) vs ≈ 0.1 ms per stump |
| 1, hurdle 2 | What about repeated values and float32? | 2.2 | 107 / 99 impossible cut positions; a test point exactly on a cut |
| 1, hurdle 3 | Is the Gini stump the minimum-error stump? | 2.4, 4.3 | different in 373 of 400 rounds |
| 2 | Does the code reproduce the hand example? | 3, 5 | assertions to 1e-15 |
| 3 | Is SAMME's 2× α really the same model? Edge cases? | 4.6 | identical weights; ε = 0 and XOR behave like scikit-learn |
| 3, hurdle 5 | Why normalise and clip the weights? | 4.4, 4.6 | overflow at round 16 574; unclipped run drifts from round 1612 |
| 4 | How is `decision_function` computed? | 4.6, 6.3 | identical to scikit-learn |
| 5 | Final comparison on both CSVs | all | 100 % identical predictions, differences 0.0, our fit ≈ 4× faster |

## How to interpret the results — quick guide

* **`estimator_errors_` (ε)** — the weighted error of each round's learner *on the weights it was trained with*. It usually
  starts lowish and creeps towards 0.5 (two classes) as the easy structure is used up. Values of exactly 1 at the end mean
  “this slot was never used” (boosting stopped early).
* **`estimator_weights_` (α)** — each learner's say. In scikit-learn, for two classes, 2× the textbook α (README §4.6). A 0
  at the end means an unused slot.
* **Train / test / CV error curves** — the main tool for choosing `n_estimators`. Choose with CV or a validation split,
  never by looking at the test curve.
* **Weight shares** (Labs 2, 10, 11) — where AdaBoost is putting its attention. A large share on points you know are mislabeled
  is the warning sign of 8.1.
* **Margins** (Lab 5) — y·F(x)/Σα between −1 and 1; negative = wrong. Growing small margins = growing confidence.
* **`predict_proba`** (Lab 14) — usable for ranking, not as honest probabilities unless calibrated.
* **Accuracy numbers** — fraction of test points classified correctly; on the bad-fit data about 18 % of the test labels are
  themselves flipped, so ~0.82 is the best any model can score against them.

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions (or, for timings,
your machine) and is fine. The from-scratch vs scikit-learn comparison should still show 100 % identical predictions. If a lab
errors, re-run the imports cell and the cells above it — most errors mean a variable from an earlier cell is missing.
