# LDA Lab — run it, look at it, understand it

This folder is the hands-on half of the LDA notes. The theory is in [`../README.md`](../README.md); every claim made
there is shown here with code, numbers and plots.

## What is where

| File | What it is |
|---|---|
| [`lda_lab.ipynb`](lda_lab.ipynb) | **The lab.** One Jupyter notebook, 16 labs in 3 parts, plus a closing guide on when to use LDA. Already executed: you can read every result on GitHub without installing anything. |
| [`requirements.txt`](requirements.txt) | The Python packages the notebook needs. |
| `README.md` | This guide. |

There are no other scripts, data files or output folders: all data is generated inside the notebook or comes bundled
with scikit-learn (no internet needed), and all results (printed numbers and plots) appear directly **under the code
cell that produced them**.

## How to run it

These steps target **VS Code on a Mac** (tested on an M4). From the repository root, create a virtual environment
**outside** the repo folder so it never gets committed, e.g. in your home directory:

```bash
python3 -m venv ~/.venvs/lda-lab
```

> On this machine pyenv's Python 3.13.3 is broken (it is killed at startup). Use 3.12.8 explicitly instead:
> `~/.pyenv/versions/3.12.8/bin/python -m venv ~/.venvs/lda-lab` — or, with uv, `uv venv --python 3.12 ~/.venvs/lda-lab`.

```bash
source ~/.venvs/lda-lab/bin/activate
```

```bash
pip install -r src/LDA/lab/requirements.txt
```

Then, in **VS Code** (with the Python and Jupyter extensions installed):

1. Open `src/LDA/lab/lda_lab.ipynb`.
2. Click **Select Kernel** (top right) → **Python Environments** → the `lda-lab` environment you just created
   (if it isn't listed, choose **Select Another Kernel… → Python Environments → Enter interpreter path** and give
   `~/.venvs/lda-lab/bin/python`).
3. Run cells one by one with **Shift + Enter**, or everything with **Run All** (about 10 seconds in total).

* **Always run the first code cell (imports and helpers) first.** After that, labs can be run in order; a lab only
  reuses variables from earlier labs, so top-to-bottom always works.
* Prefer the browser instead of VS Code? `pip install jupyterlab`, then `jupyter lab src/LDA/lab/lda_lab.ipynb`.
* Just want to read? Open `lda_lab.ipynb` on GitHub; the outputs are saved in it.

## How each lab is laid out

Every lab has the same parts, so you always know where to look:

1. **Heading + Goal** (markdown) — which question from the README this lab answers.
2. **Proof / explanation** (some labs) — a collapsed cell, **click to expand**. Optional: you can follow the lab
   without it. Labs 1, 2, 3, 4, 5, 10 and 11 have one.
3. **Code cell(s)** — short, commented, nothing hidden in helper files. Lines marked `# <- try changing this` are
   good knobs to experiment with.
4. **Output** — printed numbers and/or plots directly underneath.
5. **“How to read this” box** (a quoted markdown block) — what you should notice in the output and what it proves.
   Try to guess before you read it.

The notebook ends with a **Summary** table listing every lab and what it showed, and a **“Beyond the labs: when to use
LDA”** guide.

## Which lab answers which question

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **Part A — How LDA works** | | | |
| 1 | Does the hand-worked example really give $w \propto (5, -1)$ and $J = 3.5$? | 5 | printed $S_W$, $S_B$, $w$; the two 1-D projection plots ($J$ 3.500 vs 2.885) |
| 2 | Is the top generalised eigenvector really the best direction? What is the worst? | 4.2–4.3 | collapsible proof; $J(\theta)$ curve with the eigenvector (129.7°), worst (59.0°, $J = 0$) and 90° marked |
| 3 | Do $\text{eig}(S_W^{-1}S_B)$, `eigh(Sb, Sw)` and scikit-learn agree? | 4.3, 4.6 | eigenvalues 32.192 / 0.285 three times; within-class correlation 0.484 → 0.000; axes $|\cos| = 0.176$ |
| 4 | Why at most C − 1 components? | 4.4 | rank of $S_B$ = 9 on digits; the cliff after eigenvalue 9; scikit-learn's error for wine |
| 5 | Is LDA-the-classifier the same as Fisher's projection? | 4.5 | `predict_proba` matched to 1e-15; $|\cos| = 1$; probabilities on one S-curve |
| **Part B — Why LDA is useful** | | | |
| 6 | What is the difference between PCA and LDA? | 1, 7 | PC1: 89.7 % variance, 0.490 accuracy; LD1: 10.3 % variance, 0.885 accuracy; wine 0.961 vs 0.989 |
| 7 | Can LDA show 10 classes in 2-D? | 6.4 | PCA vs LDA digit maps; 5-NN 0.620 vs 0.652 |
| 8 | How many LDA axes should I keep before another model? | 6.4 | accuracy vs k: LDA ahead at k = 3–4, PCA catches up by 9 |
| 9 | LDA vs QDA vs logistic regression? | 7, 8.1–8.2 | four-dataset table; learning curve (QDA 0.585 vs LDA 0.705 at 12 rows per class) |
| 10 | What if there are more features than rows? | 6.3, 8.4 | blue curve collapsing; shrinkage curves staying high (0.667 vs 0.887 / 0.900 at p = 100) |
| 11 | Do units / scaling matter? | 6.5 | plain LDA identical everywhere (100.0 %); fixed shrinkage 0.703 vs 0.989 |
| **Part C — Where LDA fails** | | | |
| 12 | What happens if I fit LDA before cross-validating? | 6.6 | 0.845 on pure noise vs 0.510 in a Pipeline; separated vs overlapping histograms |
| 13 | What if the classes have different shapes? | 8.1 | straight LDA line (0.693) vs QDA circle (0.887) |
| 14 | What if a class has two clusters, or the boundary curves? | 8.2, 8.3 | XOR: LDA 0.475 vs 15-NN 1.000; moons: 0.843 vs 0.923 |
| 15 | What do outliers do? | 8.5 | direction swinging 149.3° → 15.8° and back after the robust clean-up |
| 16 | What about very imbalanced classes? | 8.6 | minority recall 0.280 vs 0.840; two thresholds on one direction |

## How to interpret the results — quick guide

* **Fisher's ratio $J$** — between-class scatter / within-class scatter of a 1-D projection. Bigger = class centres
  further apart relative to how wide each class is. 0 = the class centres land on the same spot.
* **Eigenvalues** (of $S_B w = \lambda S_W w$) — the $J$ of each LDA direction. At most $C - 1$ are non-zero; tiny
  values like `1e-15` or `-0.` are zero up to round-off.
* **`explained_variance_ratio_`** — each LDA axis's share of the total **class separation** (the eigenvalues
  divided by their sum). Not a share of the variance, despite the name.
* **Directions / `scalings_`, `coef_`** — their **sign and length are arbitrary**: if your run shows `(-0.98, 0.20)`
  where the text says `(0.98, -0.20)`, it is the same axis. That is why the notebook compares directions with
  $|\cos|$ (1.000 = same axis) or angles folded into 0°–180°.
* **Scores / `transform(X)`** — the data's coordinates along the LDA directions; these are the features another model
  would use.
* **Accuracy numbers** — fraction of test samples classified correctly; 0.50 means coin-flip for two classes, 0.10 for
  ten classes. For imbalanced data (Lab 16) look at **recall** of the rare class and **balanced accuracy** instead.

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions and is fine.
If a lab errors, re-run the imports cell and the labs above it — most errors mean a variable from an earlier lab is
missing.
