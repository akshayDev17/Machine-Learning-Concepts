# PCA Lab — run it, look at it, understand it

This folder is the hands-on half of the PCA notes. The theory is in [`../README.md`](../README.md); every claim made
there is shown here with code, numbers and plots.

## What is where

| File | What it is |
|---|---|
| [`pca_lab.ipynb`](pca_lab.ipynb) | **The lab.** One Jupyter notebook, 17 labs in 4 parts, plus a closing guide on which models PCA helps. Already executed: you can read every result on GitHub without installing anything. |
| [`requirements.txt`](requirements.txt) | The Python packages the notebook needs. |
| `README.md` | This guide. |

There are no other scripts, data files or output folders: all data is generated inside the notebook or comes bundled
with scikit-learn (no internet needed, except Lab 17's Olivetti faces, about 1.4 MB, and Lab 16's California housing
data, about 400 KB, which scikit-learn downloads once and caches), and all results (printed numbers and plots) appear directly **under the code
cell that produced them**.

## How to run it

From the repository root:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r src/PCA/lab/requirements.txt
```

Then, in **VS Code** (with the Python and Jupyter extensions installed):

1. Open `src/PCA/lab/pca_lab.ipynb`.
2. Click **Select Kernel** (top right) → **Python Environments** → the `.venv` you just created.
3. Run cells one by one with **Shift + Enter**, or everything with **Run All** (about 10 seconds in total).

* **Always run the first code cell (imports) first.** After that, labs can be run in order; a lab only reuses
  variables from earlier labs, so top-to-bottom always works.
* Prefer the browser instead of VS Code? `pip install jupyterlab`, then `jupyter lab src/PCA/lab/pca_lab.ipynb`.
* Just want to read? Open `pca_lab.ipynb` on GitHub; the outputs are saved in it.

## How each lab is laid out

Every lab has the same four parts, so you always know where to look:

1. **Heading + Goal** (markdown) — which question from the README this lab answers.
2. **Code cell(s)** — short, commented, nothing hidden in helper files. Lines marked `# <- try changing this` are
   good knobs to experiment with.
3. **Output** — printed numbers and/or plots directly underneath.
4. **“How to read this” box** (a quoted markdown block) — what you should notice in the output and what it proves.
   Try to guess before you read it.

The notebook ends with a **Summary** table listing every lab and what it showed.

## Which lab answers which question

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **Part A — How PCA works** | | | |
| 1 | Does the hand-worked example really give λ = 4 and 1? | 5 | printed matrix / eigenvalues; the before/after scatter |
| 2 | What *is* an eigenvector? | 2.6 | circle → ellipse plot; “turn” curve crossing zero |
| 3 | Is the top eigenvector really the best direction? | 4.1–4.3, 4.6 | collapsible proof cell; variance and error curves vs angle; dotted eigenvector line |
| 4 | Are eig, SVD and scikit-learn the same? Are PCs uncorrelated? | 4.4, 4.7 | identical eigenvalues; correlation 0.82 → 0.00 |
| **Part B — Why PCA is useful** | | | |
| 5 | How many components do I need? | 6.3 | cumulative plot: 90 % at 21, 95 % at 29 of 64 |
| 6 | What does a component look like? Can I see 64-D data? | 6.4, 7 | “eigen-digit” images; 2-D scatter coloured by digit |
| 7 | How much do I lose by compressing? | 4.6 | reconstructed digits for each k; two overlapping error curves |
| 8 | Can PCA remove noise? | 7 | U-shaped error curve; clean / noisy / denoised image rows |
| 9 | Does a model on fewer PCs still work? | 6.1, 6.5, 7 | accuracy vs k plateauing at the no-PCA line |
| **Part C — Where PCA fails** | | | |
| 10 | What if features have different units? | 6.2, 8.1, 8.7 | 99.8 % vs 36 % explained by PC1; raw vs scaled scatter; loadings bar chart |
| 11 | What if the structure is curved? | 8.2 | overlapping PCA histogram vs separated kernel-PCA histogram; 0.46 vs 1.00 accuracy |
| 12 | What if the label hides in a low-variance direction? | 8.3 | PC1 accuracy 0.55 vs PC2 0.99 vs LDA 0.98 |
| 13 | What do outliers do? | 8.4 | PC1 arrow swinging from ~41° to ~124° |
| 14 | What if there is nothing to compress? | 8.5 | straight-line cumulative curve; white vs colourful correlation maps |
| 15 | Is “uncorrelated” the same as “un-mixed”? | 8.6 | PCA vs ICA output rows; recovery 0.89 vs 1.00 |
| **Part D — Does PCA help this model?** | | | |
| 16 | Does PCA help tree models? | notebook's model guide | one clean cut vs a staircase; original vs rotated scores (diabetes is the exception) |
| 17 | Does PCA help clustering? | notebook's model guide | distance contrast collapsing; ARI raw vs PCs (noise columns, digits, faces) |

## How to interpret the results — quick guide

* **Eigenvalues / `explained_variance_`** — the variance along each component. Bigger = more of the data's spread.
* **`explained_variance_ratio_`** — the same as a share of the total (adds up to 1). The cumulative sum tells you how
  much of the data you keep with *k* components.
* **Components / `components_`** — directions (one per row). Each entry is the weight of one original feature.
  Their **sign is arbitrary**: if your run shows `(-0.707, -0.707)` where the text says `(0.707, 0.707)`, that is the
  same axis.
* **Scores / `transform(X)`** — the data's coordinates in the new axes; these are the features a model would use.
* **Accuracy numbers** (labs 9, 11, 12) — fraction of test samples classified correctly; 0.50 means coin-flip for two
  classes, 0.10 for ten classes.

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions and is fine.
If a lab errors, re-run the imports cell and the labs above it — most errors mean a variable from an earlier lab is missing.
