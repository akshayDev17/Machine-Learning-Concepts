# Clustering Lab — run it, look at it, understand it

This folder is the hands-on half of the clustering notes. The theory is in [`../README.md`](../README.md); every claim
made there is shown here with code, numbers and plots: once with scikit-learn / SciPy, once rebuilt from scratch in
NumPy.

## What is where

| File | What it is |
|---|---|
| [`clustering_lab.ipynb`](clustering_lab.ipynb) | **The library lab.** 12 labs in 4 parts using scikit-learn's `KMeans`, `AgglomerativeClustering`, `DBSCAN`, `OPTICS`, `HDBSCAN` and SciPy's `linkage` / `dendrogram`, ending with a summary table and a "which algorithm when" guide. Already executed: you can read every result on GitHub without installing anything. |
| [`clustering_from_scratch.ipynb`](clustering_from_scratch.ipynb) | **The from-scratch notebook.** k-means and k-means++ (plus an exact 3-pair toy showing why random starts get stuck), the silhouette and the elbow, agglomerative clustering with the Lance–Williams update, DBSCAN and OPTICS, in NumPy. Each algorithm is explained step by step in markdown, commented line by line in code, and checked against the library on the same data, plus the implementation hurdles (empty clusters, memory, ties, border points, the priority queue). Already executed. |
| [`requirements.txt`](requirements.txt) | The Python packages the notebooks need. |
| `README.md` | This guide. |

There are no data files: all data is generated in the notebooks or ships with scikit-learn (iris, wine, digits), so no
internet is needed. All results (printed numbers and plots) appear directly **under the code cell that produced them**.

## How to run it

From the repository root:

```bash
python3 -m venv "src/Unsupervised Learning/lab/.venv"
```

```bash
source "src/Unsupervised Learning/lab/.venv/bin/activate"
```

```bash
pip install -r "src/Unsupervised Learning/lab/requirements.txt"
```

On this machine pyenv's Python 3.13.3 is broken; if `python3` points to it, create the venv with
`~/.pyenv/versions/3.12.8/bin/python -m venv ...` instead. (`.venv/` folders are ignored by git.)

Then, in **VS Code** (with the Python and Jupyter extensions installed):

1. Open `src/Unsupervised Learning/lab/clustering_lab.ipynb` (then `clustering_from_scratch.ipynb`).
2. Click **Select Kernel** (top right) → **Jupyter Kernel** → **Unsupervised lab (.venv, Python 3.12.8)** if it is
   listed (it was registered when this lab was built), or **Python Environments** → the `.venv` you just created.
3. Run cells one by one with **Shift + Enter**, or everything with **Run All**: the library lab takes about
   10 seconds, the from-scratch notebook about the same.

* **Always run the first code cell (imports and helpers) first.** After that, run top to bottom; later cells reuse
  functions and data from earlier ones.
* The notebooks must run with `lab/` as the working folder (VS Code does this by default).
* Prefer the browser? `pip install jupyterlab`, then `jupyter lab "src/Unsupervised Learning/lab/clustering_lab.ipynb"`.

## How each lab is laid out

1. **Heading + Goal** (markdown): which question from the README this lab answers.
2. Sometimes a **collapsible explanation** ("click to expand"): formulas and definitions, kept out of the way.
3. **Code cell**: short and commented. Lines marked `# <- try changing this` are good knobs to experiment with.
4. **Output**: printed numbers and plots directly underneath. In the scatter plots each colour is one cluster, grey
   crosses are noise points (density methods), and black stars are k-means centroids.
5. **"How to read this" box** (a quoted markdown block): what to notice in the output. Try to guess first.

The from-scratch notebook adds **Implementation hurdle** boxes: a problem you hit when writing the algorithm yourself,
demonstrated on a small case and then fixed.

## Which lab answers which question

**Library lab** (`clustering_lab.ipynb`)

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **Part A — k-means** | | | |
| 1 | How does k-means move? | 3.2–3.3 | centroid paths; inertia 4757 → 1719, never rising |
| 2 | Why does the start matter? | 3.4 | inertia histograms: 3 % (random) vs 50 % (k-means++) of runs reach the best |
| 3 | How do I choose k? | 3.5 | elbow and silhouette on clear blobs, two pairs, and pure noise (silhouette 0.41) |
| 4 | Where does k-means fail? | 3.6 | six shapes: stretched 0.56, moons 0.25, circles 0.00; outliers stealing a cluster |
| **Part B — Hierarchical** | | | |
| 5 | How do I read and cut a dendrogram? | 4.1 | linkage matrix rows; cut at height 3.0 = 3 clusters; scikit-learn agrees |
| 6 | What does the linkage change? | 4.2–4.3 | 3 × 4 grid: single gets the moons but breaks on a bridge; iris cophenetic correlations |
| **Part C — Density** | | | |
| 7 | How does DBSCAN work, and how do I pick eps? | 5.1–5.3 | k-distance plot (eps 0.164); core / border / noise; moons ARI 1.00 |
| 8 | When does DBSCAN fail? | 5.4 | eps × min_samples heatmap; no eps fits dense and sparse blobs together |
| 9 | What do OPTICS and HDBSCAN add? | 6 | reachability plot with DBSCAN cuts; xi and HDBSCAN find all three blobs |
| **Part D — In practice** | | | |
| 10 | Does scaling matter? | 2.2 | wine k-means 0.37 → 0.90 after standardising |
| 11 | How do I score clusters without labels? | 7 | silhouette prefers the wrong answer on the moons (0.49 vs 0.05) |
| 12 | Which method on real data? | 9 | ARI table for iris, wine, digits; no method wins everywhere |

**From-scratch notebook** (`clustering_from_scratch.ipynb`)

| Section | Algorithm | Matches | Hurdles shown |
|---|---|---|---|
| 1 | k-means (Lloyd), k-means++ | `KMeans`: identical labels, inertia, iterations | empty cluster → NaN; n × k × d memory |
| 1 · toy | why the start matters: 3 pairs, every random start traced, exact k-means++ probabilities | `KMeans` from all 20 starts; Monte Carlo of our k-means++ | random init stuck 20 % of the time at (2/3)[1 + (1 + r)²] × the optimum; k-means++ ≈ 0.49 / (1 + r)² |
| 2 | silhouette, elbow (chord rule) | `silhouette_score` at every k | — |
| 3 | agglomerative, 4 linkages (Lance–Williams) | SciPy `linkage`: identical merge heights and partitions | ties; rounding changes the tree |
| 4 | DBSCAN | `DBSCAN`: identical labels and core points | border point between two clusters |
| 5 | OPTICS, cluster extraction at eps | `OPTICS`, `cluster_optics_dbscan`: identical ordering, reachability, cuts | priority queue without decrease-key |

## How to interpret the results — quick guide

* **ARI** (adjusted Rand index): agreement with the groups the data was generated from. 1 = identical, about 0 =
  random. Real problems have no such groups; it is used here only to grade methods. Noise points count as one group.
* **Inertia**: sum of squared distances to the own centroid. Lower is tighter, but it always falls as k grows.
* **Silhouette**: −1 to 1, higher = tighter and better separated. Rewards round clusters (Lab 11).
* **Noise** (−1, grey crosses): points a density method refused to put in any cluster.
* **Times** depend on your machine; only compare them within one run.

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions and is fine.
If a cell errors, re-run the imports cell and the cells above it.
