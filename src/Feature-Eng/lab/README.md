# Feature Engineering Lab — run it, look at it, understand it

This folder is the hands-on half of the feature-engineering notes. The theory is in [`../README.md`](../README.md);
every claim made there is shown here with code, numbers and plots, on one messy customer-churn table.

## What is where

| File | What it is |
|---|---|
| [`fe_lab.ipynb`](fe_lab.ipynb) | **The lab.** One Jupyter notebook, 21 labs (0–20) in 6 parts, ending with a summary table. Already executed: you can read every result on GitHub without installing anything. |
| [`requirements.txt`](requirements.txt) | The Python packages the notebook (and the pipeline scripts) need. |
| `README.md` | This guide. |
| [`../data/customer_churn.csv`](../data/customer_churn.csv) | The messy table every lab uses (3,000 customers, 35 columns, 50 / 50 churn). |
| [`../data/make_customer_churn.py`](../data/make_customer_churn.py) | The script that generated it (fixed seed; re-running gives the same file). Read it to see the "hidden truth". |
| [`../scripts/`](../scripts/README.md) | The same steps as a configurable pipeline; Lab 20 runs it. |

No internet is needed: the table is in the repository and the only other dataset (breast cancer, Lab 18) ships with
scikit-learn. All results appear directly **under the code cell that produced them**.

## How to run it (VS Code on an M4 Mac)

From the repository root, create a virtual environment **outside the repository** (here in `~/venvs`). On this
machine pyenv's Python 3.13.3 is broken, so use 3.12.8 explicitly (any other working Python ≥ 3.10 is fine):

```bash
~/.pyenv/versions/3.12.8/bin/python -m venv ~/venvs/feature-eng
```

```bash
source ~/venvs/feature-eng/bin/activate
```

```bash
pip install -r src/Feature-Eng/lab/requirements.txt
```

Then, in **VS Code** (with the Python and Jupyter extensions installed):

1. Open `src/Feature-Eng/lab/fe_lab.ipynb`.
2. Click **Select Kernel** (top right) → **Python Environments** → `~/venvs/feature-eng`. (If it is not listed,
   choose **Enter interpreter path** and paste `~/venvs/feature-eng/bin/python`.)
3. Run cells one by one with **Shift + Enter**, or everything with **Run All** (about 3 minutes in total; Lab 19's
   importance checks take about 1 minute and Lab 20's pipeline ablation about 20 seconds).

* **Always run the first code cell (imports) first.** Labs reuse tables built in earlier labs (`train`,
  `train_fixed`, `X_base`, ...), so run top to bottom.
* The notebook must run with `src/Feature-Eng/lab/` as its working folder (VS Code does this by default), because it
  reads `../data/customer_churn.csv` and imports `../scripts/run_pipeline.py`.
* Just want to read? Open `fe_lab.ipynb` on GitHub; the outputs are saved in it.

## How each lab is laid out

1. **Heading + Goal** (markdown) — which step from the README this lab is about.
2. Sometimes a **collapsible explanation** (`click to expand`) — a longer derivation, kept out of the way.
3. **Code cell** — short, commented, nothing hidden in helper files. Lines marked `# <- try changing this` are good
   knobs to experiment with.
4. **Output** — printed numbers and/or plots directly underneath, usually a **before vs after** score and, where it
   fits, a case where the step does **not** help or hurts.
5. **"How to read this" box** (a quoted markdown block) — what to notice in the output. Try to guess first.

Every score is a **5-fold cross-validated ROC-AUC on the 2,400 training rows** (0.5 = coin flip, 1.0 = perfect). The
600 test rows from Lab 1 are only scored in Lab 20.

## Which lab answers which question

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **Part A — Hygiene** | | | |
| 0 | What does the raw table look like? | 1 | three rows already showing text money, text dates, messy spellings; the collapsible "hidden truth" |
| 1 | Why split first, and why stratified? | 3 | histogram of test churn shares: 0.300–0.700 random vs always 0.475 stratified |
| 2 | What kind is each column? | 4 | the column report: numbers/dates stored as text, an ID, a constant, free text |
| 3 | Why drop IDs? | 5 | tree on the ID: 1.000 on its own rows, 0.518 on new rows |
| 4 | How do I spot a leak? | 6 | one column at 0.936 alone; model 0.702 → 0.960 (fake) |
| 5 | What do type fixes buy? | 7 | 13 spellings → 4; 0.702 → 0.776 |
| **Part B — Cleaning values** | | | |
| 6 | Fill gaps, or fill + flag? | 8 | income missing 26 % vs 6 %; 0.745 → 0.778 with flags; trees native 0.800 |
| 7 | What do outliers do, and when is capping wrong? | 9 | flattened vs steep fitted curve; fraud toy 0.999 → 0.981 |
| 8 | When does a log help? | 10 | four histograms; 0.745 → 0.759; trees identical to 6 decimals |
| 9 | Who needs scaling? | 11 | bar chart: k-NN 0.617 → 0.730, trees flat at 0.798 |
| **Part C — Categories and dates** | | | |
| 10 | How to encode 119 cities without leaking? | 12 | collapsible on cross-fitting; noise column 0.822 naive vs 0.498 honest |
| 11 | How rare is "rare"? | 13 | long-tail bar chart; threshold curve, 119 → 16 columns |
| 12 | How to use dates? | 14 | churn by month bar chart; months on a circle; 0.771 → 0.803 |
| **Part D — Creating features** | | | |
| 13 | Which domain features earn their place? | 15 | ratio vs income deciles; keyword flags 0.822 → 0.842; the "feature soup" trap |
| 14 | How do I give a linear model a U shape or an "it depends"? | 16 | U in age; premium line climbing fastest; 0.845 → 0.879; 435 columns 0.830 |
| 15 | When is binning a good idea? | 17 | staircase vs smooth U plot; trees 0.855 → 0.844 |
| **Part E — Fewer, better features** | | | |
| 16 | What do correlated columns break? | 18 | collapsible on VIF; heatmap; weight spread 0.345 vs 0.069; same score |
| 17 | How do I select features without fooling myself? | 19 | filter / RFE / lasso scores; pure noise 0.760 selected outside CV vs 0.450 inside |
| 18 | PCA or LDA? | 20 | 8 survey columns → 1 at no loss; PC1 vs LD1 histograms (0.971 vs 0.993) |
| 19 | Which features does the model actually use? | 19 | permutation vs SHAP bars (noise in red); U shape in age; one customer explained; leak, near-copy and impurity traps |
| **Part F — Putting it together** | | | |
| 20 | What is each step worth in the full pipeline? | 21–23 | the runner's stage log; ablation bar chart; with vs without feature engineering for 4 models |

## How to interpret the results — quick guide

* **CV ROC-AUC** — the average over 5 folds. Differences below about **0.01** are within the fold-to-fold spread, so
  treat them as "no change".
* **"before vs after"** — always the same rows, the same folds and the same model, so only the step changes.
* **A score that jumps suspiciously high** (Labs 4, 10, 17), or **one column towering over the rest** in an importance
  chart (Lab 19) — look for leakage first.
* **Trees vs linear models** — many steps (scaling, logs, interactions, binning) help linear models and leave trees
  unchanged; that is expected, not a bug.

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions and is fine.
If a lab errors, re-run the imports cell and the labs above it — most errors mean a variable from an earlier lab is
missing.
