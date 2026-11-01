# Clustering Lab — run it, look at it, understand it

This folder is the hands-on half of the clustering notes. The theory is in [`../README.md`](../README.md); every claim
made there is shown here with code, numbers and plots. There is **one folder per method**, and each holds two
notebooks: a **library lab** (scikit-learn / SciPy) and a **from-scratch** notebook (the same algorithm rebuilt in
NumPy and checked against the library). Each notebook stands alone and runs in a few seconds.

## What is where

| Folder | Library lab | From scratch | Labs |
|---|---|---|---|
| [`01-kmeans/`](01-kmeans/) | [`kmeans_lab.ipynb`](01-kmeans/kmeans_lab.ipynb) | [`kmeans_from_scratch.ipynb`](01-kmeans/kmeans_from_scratch.ipynb): Lloyd's algorithm, k-means++, the exact 3-pair toy, silhouette and elbow | 1–4 |
| [`02-hierarchical/`](02-hierarchical/) | [`hierarchical_lab.ipynb`](02-hierarchical/hierarchical_lab.ipynb) | [`hierarchical_from_scratch.ipynb`](02-hierarchical/hierarchical_from_scratch.ipynb): agglomerative clustering with the Lance–Williams update | 5–6 |
| [`03-dbscan/`](03-dbscan/) | [`dbscan_lab.ipynb`](03-dbscan/dbscan_lab.ipynb) | [`dbscan_from_scratch.ipynb`](03-dbscan/dbscan_from_scratch.ipynb): DBSCAN, breadth-first | 7–8 |
| [`04-optics-hdbscan/`](04-optics-hdbscan/) | [`optics_hdbscan_lab.ipynb`](04-optics-hdbscan/optics_hdbscan_lab.ipynb) | [`optics_from_scratch.ipynb`](04-optics-hdbscan/optics_from_scratch.ipynb): OPTICS and cutting it at an eps | 9 |
| [`05-in-practice/`](05-in-practice/) | [`in_practice_lab.ipynb`](05-in-practice/in_practice_lab.ipynb): scaling, scores without labels, every method on real data, the summary of all labs and a "which algorithm when" guide | — (no new algorithm) | 10–12 |

Also here: [`requirements.txt`](requirements.txt) (the Python packages every notebook needs) and this guide. All
notebooks are already executed: you can read every result on GitHub without installing anything. Lab numbers run
1–12 across the folders, so the README's "Lab N" references still point to one place.

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

1. Open the folder for the method you want, e.g. `src/Unsupervised Learning/lab/01-kmeans/kmeans_lab.ipynb`
   (then its `kmeans_from_scratch.ipynb`).
2. Click **Select Kernel** (top right) → **Jupyter Kernel** → **Unsupervised lab (.venv, Python 3.12.8)** if it is
   listed (it was registered when this lab was built), or **Python Environments** → the `.venv` you just created.
3. Run cells one by one with **Shift + Enter**, or everything with **Run All**. Measured kernel time per notebook:
   2–4 seconds each, except `kmeans_from_scratch.ipynb` at about 7 (its exact k-means++ enumeration and a
   100,000-draw check).

* **Always run the first code cell (imports and helpers) first.** After that, run top to bottom; later cells reuse
  functions and data from earlier ones.
* Each notebook runs in its own folder (VS Code does this by default) and needs nothing from the other notebooks.
* Prefer the browser? `pip install jupyterlab`, then `jupyter lab "src/Unsupervised Learning/lab"`.

## How each lab is laid out

1. **Heading + Goal** (markdown): which question from the README this lab answers.
2. Sometimes a **collapsible explanation** ("click to expand"): formulas and definitions, kept out of the way.
3. **Code cell**: short and commented. Lines marked `# <- try changing this` are good knobs to experiment with.
4. **Output**: printed numbers and plots directly underneath. In the scatter plots each colour is one cluster, grey
   crosses are noise points (density methods), and black stars are k-means centroids.
5. **"How to read this" box** (a quoted markdown block): what to notice in the output. Try to guess first.

The from-scratch notebooks add **Implementation hurdle** boxes: a problem you hit when writing the algorithm yourself,
demonstrated on a small case and then fixed.

## Which lab answers which question

**Library labs**

| Lab | Question | README section | What to look at |
|---|---|---|---|
| **[`01-kmeans/kmeans_lab.ipynb`](01-kmeans/kmeans_lab.ipynb)** | | | |
| 1 | How does k-means move? | 3.2–3.3 | centroid paths; inertia 4757 → 1719, never rising |
| 2 | Why does the start matter? | 3.4 | inertia histograms: 3 % (random) vs 50 % (k-means++) of runs reach the best |
| 3 | How do I choose k? | 3.5 | elbow and silhouette on clear blobs, two pairs, and pure noise (silhouette 0.41) |
| 4 | Where does k-means fail? | 3.6 | six shapes: stretched 0.56, moons 0.25, circles 0.00; outliers stealing a cluster |
| **[`02-hierarchical/hierarchical_lab.ipynb`](02-hierarchical/hierarchical_lab.ipynb)** | | | |
| 5 | How do I read and cut a dendrogram? | 4.1 | linkage matrix rows; cut at height 3.0 = 3 clusters; scikit-learn agrees |
| 6 | What does the linkage change? | 4.2–4.3 | 3 × 4 grid: single gets the moons but breaks on a bridge; iris cophenetic correlations |
| **[`03-dbscan/dbscan_lab.ipynb`](03-dbscan/dbscan_lab.ipynb)** | | | |
| 7 | How does DBSCAN work, and how do I pick eps? | 5.1–5.3 | k-distance plot (eps 0.164); core / border / noise; moons ARI 1.00 |
| 8 | When does DBSCAN fail? | 5.4 | eps × min_samples heatmap; no eps fits dense and sparse blobs together |
| **[`04-optics-hdbscan/optics_hdbscan_lab.ipynb`](04-optics-hdbscan/optics_hdbscan_lab.ipynb)** | | | |
| 9 | What do OPTICS and HDBSCAN add? | 6 | reachability plot with DBSCAN cuts; xi and HDBSCAN find all three blobs |
| **[`05-in-practice/in_practice_lab.ipynb`](05-in-practice/in_practice_lab.ipynb)** | | | |
| 10 | Does scaling matter? | 2.2 | wine k-means 0.37 → 0.90 after standardising |
| 11 | How do I score clusters without labels? | 7 | silhouette prefers the wrong answer on the moons (0.49 vs 0.05) |
| 12 | Which method on real data? | 9 | ARI table for iris, wine, digits; no method wins everywhere |

**From-scratch notebooks**

| Notebook | Algorithm | Matches | Hurdles shown |
|---|---|---|---|
| [`kmeans_from_scratch`](01-kmeans/kmeans_from_scratch.ipynb) | k-means (Lloyd), k-means++ | `KMeans`: identical labels, inertia, iterations | empty cluster → NaN; n × k × d memory |
| [↳ toy](01-kmeans/kmeans_from_scratch.ipynb#toy) | why the start matters: 3 pairs, every random start traced, exact k-means++ probabilities | `KMeans` from all 20 starts; Monte Carlo of our k-means++ | random init stuck 20 % of the time at (2/3)[1 + (1 + r)²] × the optimum; k-means++ ≈ 0.49 / (1 + r)² |
| [↳ choosing k](01-kmeans/kmeans_from_scratch.ipynb#sec-2) | silhouette, elbow (chord rule) | `silhouette_score` at every k | — |
| [`hierarchical_from_scratch`](02-hierarchical/hierarchical_from_scratch.ipynb) | agglomerative, 4 linkages (Lance–Williams) | SciPy `linkage`: identical merge heights and partitions | ties; rounding changes the tree |
| [`dbscan_from_scratch`](03-dbscan/dbscan_from_scratch.ipynb) | DBSCAN | `DBSCAN`: identical labels and core points | border point between two clusters |
| [`optics_from_scratch`](04-optics-hdbscan/optics_from_scratch.ipynb) | OPTICS, cluster extraction at eps | `OPTICS`, `cluster_optics_dbscan`: identical ordering, reachability, cuts | priority queue without decrease-key |

## How to interpret the results — quick guide

* **ARI** (adjusted Rand index): agreement with the groups the data was generated from. 1 = identical, about 0 =
  random. Real problems have no such groups; it is used here only to grade methods. Noise points count as one group.
* **Inertia**: sum of squared distances to the own centroid. Lower is tighter, but it always falls as k grows.
* **Silhouette**: −1 to 1, higher = tighter and better separated. Rewards round clusters (Lab 11).
* **Noise** (−1, grey crosses): points a density method refused to put in any cluster.
* **Times** depend on your machine; only compare them within one run.

If your numbers differ in the third decimal place from the saved outputs, that is due to library versions and is fine.
If a cell errors, re-run the imports cell and the cells above it.
