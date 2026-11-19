# `src/Ensemble Learning/` — evaluation and restructuring plan

This file records **what was in this folder before the rewrite**, how well each file served a reader who is
**extremely new to ML**, which statements were wrong, and how the content was reorganised. It is phase 1 of a
two-phase job (audit, then rewrite). The rewrite follows the standard that `src/PCA/` set: a theory README per
technique plus a `lab/` folder with executed notebooks.

The repo-root `AUDIT-src-factual-consistency.md` already listed many errors in these notebooks. Every one of those
findings is folded in below (marked **[audit]**), along with new ones found during this review (**[new]**). The
audit file was not edited.

---

## 1. Inventory: what the folder contained

| # | File (before) | Size | Cells | Technique(s) it promises | State |
|---|---|---:|---:|---|---|
| 1 | `Ensemble Learning.ipynb` | 11 KB | 6 | Overview of boosting, AdaBoost, gradient boosting, XGBoost | Notes only; one unexecuted code cell; one empty cell |
| 2 | `Intro to Ensembles - Voting.ipynb` | 4 KB | 5 | Voting (hard/soft), mode/mean/weighted mean | Notes + one unexecuted code cell with no data; broken image |
| 3 | `Bagging.ipynb` | 782 KB | 33 | Bagging, pasting, OOB, random subspaces/patches | Executed; **uncommitted user edits** (audit fixes); cell 15 output cleared |
| 4 | `RandomForest.ipynb` | 420 KB | 29 | Random forest (classifier, regressor), MDI, Optuna tuning | Executed; two cells error if re-run; needs a download |
| 5 | `GradientBoosting.ipynb` | 125 KB | 23 | Gradient boosting: regression, binary and multi-class classification | Long derivations; empty cells and seven empty sub-headings |
| 6 | `gradient_boosting_classification.ipynb` | **32.5 MB** | 39 | Gradient boosting binary classification, from scratch | Third-party tutorial copy; plotly outputs make it huge |
| 7 | `XGBoost.ipynb` | 9 KB | 3 | XGBoost regressor (and classifier) | Theory notes; **uncommitted user edits**; only code cell is a `ModuleNotFoundError` |
| 8 | `Advantages and Disadvantages of Ensemble Learning.ipynb` | 0.7 KB | 1 | Pros and cons of ensembles | Disadvantages only |
| 9 | `ABTesting.ipynb` | 0.4 KB | 1 | A/B testing prerequisites | Three-bullet stub; **not an ensemble topic** |
| 10 | `gradientBoostingClassifier_demo1.png` | 0.8 MB | – | Table of a 2-round gradient-boosting classifier (log-odds, probabilities, residuals, leaf ids) | Used by file 5 |
| 11 | `gradientBoostingClassifier_demo2.png` | 1.0 MB | – | Hand-annotated prediction path through F₀ + two trees → −0.59 → probability | Used by file 5 |
| – | `../elephant-ensemble.png` (lives in `src/`) | 0.2 KB | – | “Blind men and the elephant” analogy | Linked from file 2 with a wrong relative path |

### Techniques covered (the list the rewrite is organised around)

1. **Voting** (hard, soft, weighted; mean/median for regression) — files 1, 2. *Stacking* is not in the old
   content; it is added only as the fix for voting's main failure (learned weights), not as a full technique.
2. **Bagging** (incl. pasting, out-of-bag evaluation, random subspaces, random patches) — file 3.
3. **Random forest** — file 4 (+ the bagging/RF comparison in file 3).
4. **AdaBoost** — file 1 (cell 3).
5. **Gradient boosting** (regression, binary and multi-class classification) — files 1 (cell 4), 5, 6, images 10–11.
6. **XGBoost** — files 1 (cell 5), 7.

---

## 2. Per-file verdicts

Verdict scale: **Keep** (sound, needs polish) · **Rewrite** (ideas worth keeping, presentation/accuracy not
newbie-safe) · **Superseded** (content fully absorbed elsewhere) · **Out of scope**.

### 2.1 `Ensemble Learning.ipynb` — **Superseded** (split across 4 new READMEs + the folder README)

*Promises:* an overview of boosting and its three main algorithms.
*Could a newbie get a complete grasp?* No. It assumes the reader already knows decision trees, loss functions,
residuals, gradients, bias and variance; none are defined anywhere in the folder.

Factual errors:

- 🔴 **[audit] cell 4:** “DT2 … whose dependent feature is actually the outputs from DT1”. → Tree *m* is fitted to
  the **new residuals** y − F<sub>m−1</sub>(x), not to the previous tree's outputs.
- 🟠 **[audit] cell 4:** “since gradient is taken w.r.t. the residuals, it's called gradient boosting”. → The gradient
  is taken w.r.t. the **current prediction** F(x); the pseudo-residual is −∂L/∂F.
- 🟠 **[audit] cell 4:** “low bias but high variance, since for the test data the predictions will largely remain the
  same”. → Backwards: predictions that stay the same across data sets are *low* variance. The real reason for shrinkage
  is that a full-size step over-fits the training residuals.
- 🟠 **[audit] cell 4:** “the loss function is not a typical loss function like MSE …” → For squared loss the
  residual **is** the (negative) gradient of the ordinary MSE loss; nothing exotic.
- 🟠 **[audit] cell 4:** the line-search step calls F<sub>m−1</sub> “the new dependent feature” and omits the
  per-leaf γ<sub>jm</sub>.
- 🟡 **[audit] cell 4:** “MSE (minimum squared error)”. → *Mean* squared error.
- 🟡 **[audit] cell 1:** “The aggregate result of multiple models is always less noisy” and boosting “has shown better
  predictive accuracy than bagging” overclaim; bullet 2 merges AdaBoost's *re-weighting* with gradient boosting's
  *residual fitting*; “both of these ensemble techniques use voting and combine models of the same type” — boosting
  uses a **weighted sum**, not a vote.
- 🟡 **[audit] cell 3:** “`n_estimators` is usually set to 500”, “often hundreds, if not thousands”. → scikit-learn's
  default is **50**; with stumps a few hundred is typical, and more rounds can over-fit noisy labels.
- 🟡 **[new] cell 3:** the stage value P = ½ ln((1−TE)/TE) is given without *why* (it is the exact minimiser of the
  exponential loss); no multi-class version (SAMME adds ln(K−1)); no statement that TE must be < 0.5; the final
  prediction rule (sign of the weighted vote) is never written.
- 🟡 **[new] cell 5:** XGBoost section is marketing bullets (“two goals: speed, performance”), no content.

Kept and re-explained: the AdaBoost step list (equal weights 1/N, stumps, total error, stage value, up-/down-weighting,
normalising, termination), the AdaBoost data-preparation advice (quality data, outliers, noisy labels — now
demonstrated in a lab), the gradient-boosting step list and pseudo-algorithm, and the bagging-vs-boosting contrast.

### 2.2 `Intro to Ensembles - Voting.ipynb` — **Rewrite** → `01-voting/`

*Promises:* voting ensembles. *Newbie grasp?* No — it shows a `VotingClassifier` that is never fitted on any data,
and never says *why* voting should work or when it doesn't.

- 🟠 **[audit] cell 0:** hard voting “model is selected … by a simple majority vote”. → No model is selected; the final
  **class** is the most common predicted label (the mode).
- 🟠 **[audit] cells 1, 3:** “almost always better”, “generally higher than the individual classifiers”. → Only when
  members beat chance **and** their errors are weakly correlated (Condorcet's jury theorem). A strong model voted
  down by two correlated weak ones gets worse.
- 🟡 **[audit] cell 4:** the mode is a poor aggregator for regression (continuous predictions rarely tie); use the
  mean or median.
- 🟠 **[audit] broken image:** `elephant-ensemble.png` is in `src/`, not next to the notebook.
- 🟡 **[new]** soft voting is described without its requirement that probabilities be **comparable / calibrated**;
  `SVC()` in the example has no `predict_proba` unless `probability=True`, so it could not be used for soft voting.
- 🟡 **[new]** the `VotingRegressor` link points to the 1.5 docs; the first line links an external blog as the
  content (“Please follow this”).

Kept: the elephant analogy and image, hard vs soft voting, mode / mean / weighted mean, the diversity advice.

### 2.3 `Bagging.ipynb` — **Rewrite** → `02-bagging/` (the strongest file in the folder)

*Promises:* bagging for classification and regression, pasting, OOB, random subspaces and patches.
*Newbie grasp?* Partly. It is the most complete notebook and already contains the user's corrections (variance
formula ρσ² + (1−ρ)σ²/B, “less correlated, not uncorrelated”). Gaps: decision trees and “high variance” are never
explained; the 63 % / 37 % numbers are delegated to a ChatGPT link; the feature-importance section stops at “3.”;
there is no bias/variance demonstration and no “where bagging fails”.

- ✅ **[audit] cells 15–16, 1, 7, 8, 27, 28** — already fixed in the working tree (per-tree predictions; ρ floor;
  correlation ≠ similar MSE). Cell 15's output was cleared and never re-run.
- 🟠 **[audit] cell 12:** “it is really all samples are being used” — `estimators_samples_` lists 750 **draws**
  per tree; only ≈ 63.2 % of them are unique rows.
- 🟠 **[audit] cell 18:** OOB samples exist “when `max_samples != n_samples`”. → They exist whenever
  `bootstrap=True`, even with `max_samples = n`; and OOB predictions **are** aggregated (over the trees that did not
  see the row) — that is how `oob_score_` is computed.
- 🟠 **[audit] cell 23:** “only 63 % of the instances are included” is about **one tree**, not the ensemble, and
  holds for `max_samples = n`; with `max_samples=100` out of 750 each tree leaves out (1 − 1/750)<sup>100</sup> ≈
  **87.5 %**. “Bagging is much better than pasting” overclaims — the two are usually close.
- 🟠 **[audit] cells 29–30:** the model called “random subspaces” uses `max_samples=0.6`, so it also samples rows
  (that is *random patches*), and `bootstrap_features=True` draws features **with** replacement (duplicates).
- 🟡 **[audit] cell 1:** sampling without replacement is called “subsetting”; the standard name is **pasting**.
- 🟡 **[audit / new] cells 5–6:** `Counter((dt > base).all(axis=1))` counts trees that the single tree beats on **all
  three classes**; `False: 16` means “at least one class where this tree ties or wins”, not “16 trees were as precise
  or more precise”.
- 🟡 **[audit] cell 17:** the recall panel plots column 1 twice (labelled class-1 and class-2).
- 🟡 **[new] cell 14:** “Hmm… strange” — the 750-row bagger beating the 100-row bagger is expected: each tree sees
  ~474 unique rows instead of ~94; this is a bias effect, not a mystery.
- 🟡 **[new]** several model variants are compared on the **test** set (cells 4, 13, 30) — mild model selection on
  test. In the rewrite, choices are made with OOB/CV and the test set is used once per comparison.

Kept: everything — bootstrap vs pasting, diversity argument, ρ formula, small-data warning, OOB score and
`oob_decision_function_`, the per-tree-vs-single-tree comparison, the error-correlation heatmaps and the >0.9 check,
random patches/subspaces, averaged feature importance, `BaggingRegressor`.

### 2.4 `RandomForest.ipynb` — **Rewrite** → `03-random-forest/`

*Promises:* random forest classifier/regressor, feature importance, tuning. *Newbie grasp?* No: per-split feature
sampling is described in one bullet, *why* it helps is never derived, and two cells crash on a fresh run.

- 🟠 **[audit] cells 0, 15:** in scikit-learn ≥ 1.1 `RandomForestRegressor` defaults to `max_features=1.0`, i.e. **no**
  feature sampling — the default regressor *is* bagged trees. (Cells 15 vs 16 show it: MSE 7365 vs 7363.)
- 🟠 **[audit] cell 5:** RF is a “lower-bias” model. → Averaging lowers **variance**; bias stays about that of one deep
  tree (slightly higher, because of the feature restriction).
- 🟠 **[audit] cell 9:** the custom MDI uses `feature_importance_arr[feat_idx-1]` (off by one) and sums each split
  node's **raw impurity** instead of its **weighted impurity decrease** — hence it disagrees with
  `feature_importances_` (e.g. 0.035 vs 0.075 for feature 0).
- 🟠 **[audit] cell 23:** Optuna maximises R² on `x_test` (test set used for tuning); the comment says Hyperband but
  no pruner is configured; the run took 758 s.
- 🟡 **[audit] cell 0:** “weak learner” — RF trees are deep, strong, high-variance learners.
- 🟡 **[audit] cell 26:** between-tree spread is not the ensemble's variance (and `rf_reg` there has `max_features=0.8`,
  so the comparison is bagging vs almost-bagging).
- 🟠 **[new] cell 17:** the source is the incomplete expression `rf_reg.est` (raises `AttributeError` if re-run).
- 🟠 **[new] cell 21:** `X.isna()` on a NumPy array raises `AttributeError`.
- 🟠 **[new] cells 19–20:** `fetch_california_housing` downloads data (not runnable offline).
- 🟡 **[new] cell 15:** MSE 7365 on `make_regression` (a **linear** target) is never commented on — it is a textbook
  case of where forests are weak (see the new README §8).
- 🟡 **[audit] cells 27–28:** a DBSCAN visualiser link and a ChatGPT link stand in for content.

Kept: the bagging-vs-RF difference (per-split vs per-tree feature sampling), pruning-related advice, the base-tree vs
forest comparison, error-correlation heatmaps, MDI (now correct) and the “median instead of mean” idea (now a lab).

### 2.5 `GradientBoosting.ipynb` — **Rewrite** → `05-gradient-boosting/`

*Promises:* GBM regression, binary and multi-class classification. *Newbie grasp?* Partly — the derivations are
real and careful, but they jump straight into calculus with no picture, contain the slips below, and the
multi-class and failure sections are empty headings.

- 🟠 **[audit] cells 1, 17:** “had we used the η multiplier [inside the argmin] it would've still not made any
  difference”. → Putting η inside the argmin makes the optimiser undo it (γ grows by 1/η). η is applied **after**
  solving for γ; that is what makes it shrinkage.
- 🟠 **[audit] cells 8–9:** “validation score” is computed on the **test** set.
- 🟡 **[audit] cell 18:** “F₀(x) = p₊ = ȳ”, then log-odds. → F₀ **is** the log-odds ln(p₊/(1−p₊)); p₊ is what you
  get back through the sigmoid. Plus an intermediate sign slip.
- 🟡 **[audit] cells 1–2:** with L = (1/n)Σ(y−ŷ)² the negative gradient is (2/n)(y−ŷ), not y−ŷ; the clean
  statement uses ½(y−ŷ)² per sample. Cell 1 also has `--2(y_i - …)` and mixes indices *i*/*j*.
- 🟡 **[audit] cell 2:** `validation_fraction` only matters when `n_iter_no_change` is set; `subsample` draws a new
  subset **at every iteration, without replacement** (this answers the notebook's own red question).
- 🟡 **[audit] cell 1:** scikit-learn's default base tree has `max_depth=3`, not a stump.
- 🟡 **[audit] cell 10:** imbalance and drift are not gradient-boosting-specific failures; “limited data” is not a
  classic failure either (a small, regularised GBM is a standard baseline).
- 🟡 **[new] cell 10:** “y = x1 XOR x2” fails only for **stumps** (an additive model cannot express an interaction);
  depth ≥ 2 trees learn XOR easily.
- 🟡 **[new] cell 2:** red questions left open — “why `friedman_mse`?”, “low learning rate → more models? Test
  this!” — answered and tested in the rewrite.
- 🟡 **[new] cells 7–8:** `fetch_california_housing` needs a download; `tqdm.notebook` is an extra dependency.
- **Unfinished [audit]:** empty cells 11, 13, 20; seven empty multi-class sub-headings; “Disadvantages: memory
  complexity” only; DBSCAN visualiser link.

Kept: the full regression derivation (F₀ = mean, leaf value = mean residual), the binary “vanilla” derivation in
probability space and *why it breaks down* at m ≥ 2, the second-order (Newton) leaf value Σr / Σp(1−p), the
log-odds rewrite, softmax invariance and the K-class prior derivation, sklearn's K-trees-per-round note, the
residual-correlation diagnostic, the learning-rate experiment, both demo images, the XOR / sin(x₁+x₂) /
“population vs sample” failure ideas.

### 2.6 `gradient_boosting_classification.ipynb` — **Superseded** → `05-gradient-boosting/lab/…_from_scratch.ipynb`

*Promises:* gradient-boosting classification explained with an example, math and code.

- 🟠 **[audit] attribution:** a near-verbatim copy of Tomonori Masui's *Gradient Boosting Algorithm — Part 2:
  Classification* (Towards Data Science) without explicit credit. The rewrite credits it under Further reading.
- 🟠 **[audit] cell 33:** `fit` calls `tree.apply(x)` on the **global** `x`, not the `X` argument; `predict_proba`
  does `Fm = self.F0; Fm += …`, mutating `self.F0` in place, so a second call returns different numbers; `self.F0`
  is stored with the **training** length, so predicting on a different-sized `X` fails.
- 🟡 **[audit] cell 12:** “usually 8 to 32 terminal nodes” — ESL recommends about 4–8.
- 🟡 **[new]** plotly 3-D outputs make the file 32.5 MB; GitHub cannot render it.

Kept: the circles example and the idea of watching residuals and predictions update round by round (2-D plots
instead of 3-D), the exact log-loss agreement with scikit-learn as a correctness test (now with the bugs fixed).

### 2.7 `XGBoost.ipynb` — **Rewrite** → `06-xgboost/`

*Promises:* XGBoost regressor and classifier. *Newbie grasp?* No — the regressor notes (already corrected by the
user: SS = (Σr)²/(N+λ), w* = Σr/(N+λ), gain with ½ and −γ, λ = L2 and α = L1) are good but have no derivation,
no classifier, no code that runs.

- ✅ **[audit]** similarity score, gain, λ/α corrections — present in the working tree.
- 🟡 **[audit]** the 2nd-order Taylor expansion is listed under “optimised for speed”; it is about *accuracy and
  generality* (a Newton step that works for any twice-differentiable loss).
- 🟡 **[audit]** the exact (pre-sorted “block”) method and the histogram method are described as one thing.
- 🟡 **[audit]** “zeros are skipped as missing” is only true for **sparse** input; in a dense array 0 is a value.
- 🟡 **[audit]** “faster inference time than GBM” is unsupported.
- 🟡 **[new]** “λ high → all 3 terms small” is right in spirit but the reason (λ shrinks every leaf weight towards 0
  and lowers every gain) is not spelled out; min_child_weight, eta, subsampling and early stopping are missing.
- **Unfinished [audit]:** empty classifier section, two empty “7.” items, code cell only `ModuleNotFoundError`.

Kept: all of it, including the histogram worked example (now with the gain actually computed: 0.434 for λ = 1,
γ = 0) and the missing-value default-direction explanation.

### 2.8 `Advantages and Disadvantages of Ensemble Learning.ipynb` — **Superseded** → folder `README.md` §6

- 🟡 **[audit]** “not good for real-time applications” and “very difficult to draw any business insights” are too
  strong: a few hundred shallow trees predict in microseconds, and TreeSHAP / permutation importance explain them.
- **Missing [audit]:** there is no Advantages section. Added.

### 2.9 `ABTesting.ipynb` — **Out of scope, left in place**

A three-bullet stub about experiment design; it has nothing to do with ensembles **[audit]**. The task's constraints
allow touching only this folder, so it was **not moved or edited**. Recommendation: move it next to the statistics /
hypothesis-testing notes and port the content listed in the audit (primary metric + guardrails, MDE/power before
launch, no peeking, novelty burn-in, Simpson's-paradox check).

### 2.10 Images

- `gradientBoostingClassifier_demo1.png`, `gradientBoostingClassifier_demo2.png` — screenshots from a video lecture
  (a placement dataset with `cgpa`, `iq`; learning rate 1). Correct and useful; **moved** to
  `05-gradient-boosting/images/` and re-explained number by number in that README.
- `src/elephant-ensemble.png` — **copied** (not moved: `src/` is outside this task's scope) to
  `01-voting/images/elephant-ensemble.png`, fixing the broken link.

---

## 3. Cross-cutting gaps for a beginner

1. **No decision-tree primer.** Every technique here is “many trees”; nothing in the folder says what a tree is,
   how it picks a split, or why a deep tree has high variance.
2. **No bias–variance primer.** “Low bias, high variance” is used in every notebook and defined in none (the audit
   lists bias/variance swaps as recurring anti-pattern 5).
3. **No “where it does NOT fit” for any technique** (GBM's cell 10 is a stub).
4. **No hand-worked examples** with small numbers, except the GBM screenshots.
5. **No from-scratch code** except the copied GBM classifier (which has bugs).
6. **Downloads and extra dependencies:** California housing (RF, GBM), seaborn, plotly, tqdm, optuna.
7. **External links as content:** ChatGPT shares (Bagging, RF), DBSCAN visualiser links (RF, GBM), “Please follow
   this” (Voting).
8. **No reading order** and no index for the folder.

### Leakage sweep (audit anti-patterns 2 and 3)

| File | Finding | Status in the rewrite |
|---|---|---|
| `RandomForest.ipynb` cell 23 | hyper-parameters tuned on the test set | tuning with cross-validation / OOB; test used once |
| `GradientBoosting.ipynb` cells 8–9 | test set reported as “validation score” | explicit train / validation / test split; early stopping on validation |
| `Bagging.ipynb` cells 4, 13, 30 | several configurations compared on test | choices via OOB; test used once per comparison |
| (all) | fit-on-everything preprocessing | not applicable: no scalers/imputers are used; where a scaler is needed (voting members) it sits in a `Pipeline` |

---

## 4. Restructuring plan

One folder per technique, in reading order; each with the PCA-style theory README and a `lab/` folder. A new
folder-level `README.md` holds the shared foundations (what an ensemble is, decision trees, bias and variance,
choosing a technique, advantages and disadvantages).

```
src/Ensemble Learning/
├── README.md                      ← start here: foundations, map, chooser, pros & cons
├── EVALUATION.md                  ← this file
├── 01-voting/            README.md, images/, lab/
├── 02-bagging/           README.md, images/, lab/
├── 03-random-forest/     README.md, images/, lab/
├── 04-adaboost/          README.md, images/, lab/
├── 05-gradient-boosting/ README.md, images/, lab/
├── 06-xgboost/           README.md, images/, lab/
├── ABTesting.ipynb                ← untouched (out of scope, see 2.9)
└── _archive/                      ← the superseded originals, kept for review
```

Every `lab/` folder has the same shape:

| File | What |
|---|---|
| `<technique>_lab.ipynb` | library implementation (scikit-learn / xgboost); executed, outputs saved |
| `<technique>_from_scratch.ipynb` | NumPy-only implementation, one section per theory step, compared with the library on the same data |
| `data/good_fit.csv`, `data/bad_fit.csv` | the generated “this technique shines” and “this technique is the wrong tool” datasets (generator code is in the lab notebook) |
| `README.md` | the lab guide (PCA lab-README format) |
| `requirements.txt` | `ipykernel`-based, no `jupyterlab` |

### Why `_archive/` instead of deleting

`Bagging.ipynb` and `XGBoost.ipynb` hold **uncommitted** edits. Deleting them would destroy those edits, so all
superseded originals were moved (not deleted) into `_archive/`. Their content is fully absorbed into the new files;
`_archive/` can be deleted once reviewed. Nothing was committed.

### Old path → new path

| Old path | New path | What happened |
|---|---|---|
| `Ensemble Learning.ipynb` | `_archive/Ensemble Learning.ipynb` | content split into `README.md`, `04-adaboost/README.md`, `05-gradient-boosting/README.md`, `06-xgboost/README.md` |
| `Intro to Ensembles - Voting.ipynb` | `_archive/Intro to Ensembles - Voting.ipynb` | rewritten as `01-voting/` |
| `Bagging.ipynb` (uncommitted edits) | `_archive/Bagging.ipynb` | rewritten as `02-bagging/` |
| `RandomForest.ipynb` | `_archive/RandomForest.ipynb` | rewritten as `03-random-forest/` |
| `GradientBoosting.ipynb` | `_archive/GradientBoosting.ipynb` | rewritten as `05-gradient-boosting/` |
| `gradient_boosting_classification.ipynb` | `_archive/gradient_boosting_classification.ipynb` | superseded by `05-gradient-boosting/lab/gradient_boosting_from_scratch.ipynb` |
| `XGBoost.ipynb` (uncommitted edits) | `_archive/XGBoost.ipynb` | rewritten as `06-xgboost/` |
| `Advantages and Disadvantages of Ensemble Learning.ipynb` | `_archive/Advantages and Disadvantages of Ensemble Learning.ipynb` | rewritten as `README.md` §6 |
| `gradientBoostingClassifier_demo1.png` | `05-gradient-boosting/images/gradientBoostingClassifier_demo1.png` | moved |
| `gradientBoostingClassifier_demo2.png` | `05-gradient-boosting/images/gradientBoostingClassifier_demo2.png` | moved |
| `../elephant-ensemble.png` | `01-voting/images/elephant-ensemble.png` | copied (original left in `src/`) |
| `ABTesting.ipynb` | `ABTesting.ipynb` | untouched |

### Per-technique plan: good-fit and bad-fit datasets

| Technique | Good-fit dataset (pattern it handles well) | Bad-fit dataset (where it is the wrong tool) | From-scratch ↔ library check |
|---|---|---|---|
| Voting | 2-D classes whose boundary is part straight, part curved, part clustered, so a linear model, naive Bayes and k-NN each fail in *different* regions | a task only one member can solve (others fail *together*): the majority outvotes the one good model | own vote rule on the library members = `VotingClassifier`, prediction for prediction |
| Bagging | noisy, wiggly 1-D/2-D regression: a deep tree over-fits, the average smooths | a straight-line trend evaluated beyond the training range (trees cannot extrapolate; bagging a stable linear model changes nothing) | own CART tree + bootstrap counts = `BaggingRegressor` tree by tree |
| Random forest | many informative features plus one dominant feature that makes bagged trees all split alike | a few relevant features hidden among hundreds of irrelevant ones (most per-split samples contain only noise) | own forest vs `RandomForest*` on accuracy/MSE and importances; exact match in the bagging limit |
| AdaBoost | low-noise data whose boundary needs many axis-aligned cuts | the same data with flipped labels: weights pile onto mislabeled points | own SAMME + weighted-Gini stumps = `AdaBoostClassifier` exactly |
| Gradient boosting | tabular data with non-linear effects and interactions (Friedman #1 style) | a target with a time trend, evaluated in the future (piecewise-constant trees cannot extrapolate) | own GBM = `GradientBoosting*` to float precision (regression, binary, multi-class) |
| XGBoost | larger tabular data with informative missing values and interactions | p ≫ n with a sparse linear signal (a regularised linear model wins) | own second-order booster = `xgboost` with `tree_method="exact"` to float32 precision; histogram variant compared on metrics |

Each README's “where it does NOT fit” table also covers further failure modes, each with its own lab (outliers,
label noise, extrapolation, interactions with stumps, misleading impurity importances, uncalibrated probabilities, …).

---

## 5. Outcome

The rewrite is done: every technique folder above exists with its theory README, images, and a `lab/` holding the
library notebook, the from-scratch notebook, both datasets, a lab README and `requirements.txt`. Each README's
**`Appendix - Assumptions made`** records that technique's assumptions and what was kept or corrected. Verification
(runtimes, warnings, from-scratch-vs-library agreement) is in [§6](#6-verification-results).

## 6. Verification results

Every notebook was executed top to bottom with `nbclient` (Python 3.13.5, NumPy 2.5.4, SciPy 1.18.1, scikit-learn
1.9.1, Matplotlib 3.11.2, xgboost 3.4.1) and saved with outputs. Then each saved notebook was **re-executed from a
blank copy** and its text outputs were diffed against the saved ones. Every figure was opened and checked for layout.

### 6.1 Execution

| Notebook | Size | Figures | Re-run time | Errors | stderr / warnings | Output differences on re-run |
|---|---:|---:|---:|---|---|---|
| `01-voting/lab/voting_lab.ipynb` | 0.65 MB | 9 | 7 s | 0 | 0 | none |
| `01-voting/lab/voting_from_scratch.ipynb` | 0.07 MB | 1 | 3 s | 0 | 0 | none |
| `02-bagging/lab/bagging_lab.ipynb` | 1.13 MB | 14 | 22 s | 0 | 0 | 1 cell, timing column only |
| `02-bagging/lab/bagging_from_scratch.ipynb` | 0.08 MB | 1 | 7 s | 0 | 0 | 3 cells, timing columns only |
| `03-random-forest/lab/random_forest_lab.ipynb` | 0.60 MB | 10 | 50 s | 0 | 0 | 1 cell, timing column only |
| `03-random-forest/lab/random_forest_from_scratch.ipynb` | 0.07 MB | 0 | 31 s | 0 | 0 | 2 cells, timing columns only |
| `04-adaboost/lab/adaboost_lab.ipynb` | 1.16 MB | 12 | 72 s | 0 | 0 | none |
| `04-adaboost/lab/adaboost_from_scratch.ipynb` | 0.06 MB | 0 | 11 s | 0 | 0 | none |
| `05-gradient-boosting/lab/gradient_boosting_lab.ipynb` | 1.02 MB | 14 | 62 s | 0 | 0 | none |
| `05-gradient-boosting/lab/gradient_boosting_from_scratch.ipynb` | 0.09 MB | 0 | 23 s | 0 | 0 | 1 cell, timing columns only |
| `06-xgboost/lab/xgboost_lab.ipynb` | 0.49 MB | 10 | 45 s | 0 | 0 | none |
| `06-xgboost/lab/xgboost_from_scratch.ipynb` | 0.09 MB | 1 | 40 s | 0 | 0 | 2 cells, timing columns only |

All twelve notebooks together run in about **6 minutes** on an M4 MacBook Pro. Wall-clock timings (fit/predict
seconds) naturally change between runs, so the READMEs quote them as approximate; **every other number** reproduced
exactly. One warning is raised on purpose — the `SVC(probability=True)` deprecation in voting Lab 4 — and it is
caught and printed as part of that lab's lesson; nothing else warns. All README links and image paths
resolve, every theory README ends with `## Appendix - Assumptions made`, and no `requirements.txt` lists
`jupyterlab`.

### 6.2 From-scratch (NumPy) vs library, same data and hyper-parameters

| Technique | Library reference | Agreement achieved |
|---|---|---|
| Voting | `VotingClassifier` | Own vote rules on the library's members: 100 % identical predictions (hard and soft, both CSVs). Own members: GaussianNB ≤ 7.8e-16, k-NN exact, logistic regression ≤ 1.2e-4 (3.4e-9 against a tighter solver tolerance). Whole from-scratch ensemble: 100 % identical predictions. |
| Bagging | `BaggingRegressor` / `BaggingClassifier` | With scikit-learn's bootstrap draws: per-tree and ensemble predictions identical (max diff 0.0), OOB R² equal to 12 decimals. Classification on 2-D data: 100 % identical classes (probabilities ≤ 0.08 apart because of exact feature ties broken in random order). |
| Random forest | `RandomForestRegressor` / `RandomForestClassifier` | Bagging limit: 100/100 trees identical, importances ≤ 2.8e-17. With feature sampling, replaying scikit-learn's per-node random draws: 100/100 (good fit) and 95/100 (bad fit) trees identical; the 5 differences start at exact arithmetic ties. With own random numbers: metrics within the spread of a different scikit-learn seed. |
| AdaBoost | `AdaBoostClassifier` (SAMME, stumps) | Bit-for-bit: 100 % identical predictions, every stump identical, `estimator_weights_` / `estimator_errors_` / `decision_function` differences exactly 0.0 (two- and three-class). Textbook ½α version gives the same predictions. |
| Gradient boosting | `GradientBoostingRegressor` / `GradientBoostingClassifier` | 13 settings (squared, absolute, Huber, binary, 3-class, `subsample=0.5`, both CSVs + circles): max difference 7.1e-14; same early-stopping round; reproduces the old notebook's log-loss 0.461943067988696. |
| XGBoost | `xgboost`, `tree_method="exact"` | Bit-for-bit on both CSVs (squared error, regularised, logistic): max prediction difference 0.0; same early-stopping iteration. Histogram variant vs `hist`: RMSE 1.447 vs 1.442 (metrics only). |

### 6.3 Findings during the rewrite that changed the old content

* **Gradient boosting screenshot (`gradientBoostingClassifier_demo2.png`):** round 0 and tree 1 check out, but the
  hand-written tree-2 leaf values (1.09, −2.94, 1.56) cannot come from the leaf formula Σr / Σp(1−p); recomputing them
  from the table's own residuals gives 0.577, −2.991, 1.050 (scikit-learn agrees). The query point's log-odds become
  −1.106 (p = 0.249) instead of −0.59 (p ≈ 0.36); the predicted class is unchanged. The image is kept and the README
  shows both.
* **xgboost's reported gain has no ½:** the library's `gain` (in tree dumps, and the number compared with γ) is
  SS_L + SS_R − SS. The README keeps the paper's Gain = ½[…] − γ and states the library convention.
* **scikit-learn 1.9 deprecations:** `criterion` in `GradientBoosting*` (no effect) and `SVC(probability=True)`.
  Both are noted where they matter.
* **Bagging's small-data warning** in the old notes gave the wrong reason (bootstrap overlap is ≈ 0.40 at every size);
  the real cause is that each tree's bias grows on tiny samples. Corrected in `02-bagging/` §8.4.

### 6.4 Not done / left as is

* `ABTesting.ipynb` was not moved (out of scope, see 2.9).
* `_archive/` still holds the originals (including the user's uncommitted edits in `Bagging.ipynb` and
  `XGBoost.ipynb`); delete it once reviewed.
* The theory READMEs run 650–800 lines, longer than PCA's 609, because each covers both a library and a
  from-scratch notebook.
* Nothing was committed.
