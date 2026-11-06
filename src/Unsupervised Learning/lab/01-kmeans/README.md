# k-means

> Part of the [clustering notes](../../README.md); see there for [choosing a method](../../README.md#3-choosing-a-method-the-flowchart).
> Labs in this folder: [`kmeans_lab.ipynb`](kmeans_lab.ipynb) (Labs 1–4, scikit-learn) and
> [`kmeans_from_scratch.ipynb`](kmeans_from_scratch.ipynb) (the same algorithm in NumPy). Distance, scaling and the
> scores used to judge clusters are in [clustering in practice](../05-in-practice/README.md).
> Running a fitted model in production (monitoring, retrain decisions, the retrain playbook) is in
> [`kmeans_in_production.ipynb`](kmeans_in_production.ipynb); its code lives in [`kmeans_production.py`](kmeans_production.py),
> a module you can import elsewhere.
> How to define the very first training window, for a new business and for one with two years of orders, is in
> [`first_go_live_scenarios.md`](first_go_live_scenarios.md).

**Family:** centroid. Each cluster is a centre; every point belongs to the nearest one. Finds round clusters; you give k.

1. [Inertia (within-cluster sum of squares, WCSS)](#1-inertia-within-cluster-sum-of-squares-wcss)
2. [The goal](#2-the-goal)
3. [The algorithm (Lloyd's algorithm)](#3-the-algorithm-lloyds-algorithm)
4. [Why it always stops](#4-why-it-always-stops)
5. [The start matters: k-means++](#5-the-start-matters-k-means)
6. [Choosing k](#6-choosing-k)
7. [Assumptions and failures](#7-assumptions-and-failures)
8. [Cost and variants](#8-cost-and-variants)
9. [Common misconceptions](#9-common-misconceptions)
10. [Terms](#10-terms)
11. [Further reading](#11-further-reading)

## 1. Inertia (within-cluster sum of squares, WCSS)

For a clustering with centres $\mu_1, \dots, \mu_k$, the **inertia** is the total squared distance from each point to
its own cluster's centre:

$$\text{inertia} = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

Small inertia = tight clusters. It always falls as k grows (with $k = n$ every point is its own centre and the inertia
is 0), so it can't be used alone to choose k ([§6](#6-choosing-k)).

> **Why "inertia"?** It is the physics *moment of inertia* $I_c = \sum_i m_i \lVert x_i - c \rVert^2$ with every point
> given mass $m_i = 1$ and measured about its own centroid: the inertia is each cluster's spread around its centre of
> mass. Two physics facts carry over: the centre of mass minimises $I_c$ (which is why k-means moves each centroid to
> the mean), and the parallel-axis theorem splits the total spread as
> $$\underbrace{\sum_i \lVert x_i - \mu \rVert^2}_{\text{total}} = \underbrace{\sum_i \lVert x_i - \mu_{c(i)} \rVert^2}_{\text{within (inertia)}} + \underbrace{\sum_k n_k \lVert \mu_k - \mu \rVert^2}_{\text{between}}$$
> so making clusters tight is the same as pushing their centroids apart.

## 2. The goal

Choose $k$ centres (**centroids**) and assign every point to one of them so that the inertia is as small as possible.
Finding the very best answer is computationally hard in general, so k-means finds a good local answer with a simple
loop.

## 3. The algorithm (Lloyd's algorithm)

1. Pick $k$ starting centroids ([§5](#5-the-start-matters-k-means)).
2. **Assign:** give every point to its nearest centroid.
3. **Update:** move every centroid to the **mean** of the points assigned to it.
4. Repeat 2–3 until the assignments stop changing (equivalently, no centroid moves).

Lab 1 shows the centroids walking to their final places; [k-means from scratch](kmeans_from_scratch.ipynb) implements
the two steps and matches scikit-learn's labels and inertia exactly from the same start.

## 4. Why it always stops

* The **assign** step can only lower the inertia: each point moves to a centroid at least as close as its old one.
* The **update** step can only lower it too: for a fixed group of points, the point $\mu$ that minimises
  $\sum_i \lVert x_i - \mu \rVert^2$ is their mean. (Setting the derivative $-2\sum_i (x_i - \mu)$ to zero gives
  $\mu = \frac{1}{n}\sum_i x_i$.)
* The inertia can't fall forever (there are finitely many ways to assign the points), so the loop stops, typically
  after a few to a few dozen iterations. Lab 1: 4757 → 1719 in 15 iterations, never rising.

It stops at a **local** minimum, which depends on the start.

## 5. The start matters: k-means++

With random starting centroids, two can land in the same true cluster and none in another; the loop then rarely
recovers.
- **k-means++** spreads the starts out: the first centroid is a random point from the sample; each next centroid is drawn from the remaining samples with probability
proportional to $D(x)^2$, the squared distance to the nearest centroid already chosen. 
- On 10 blobs (Lab 2), only 3% of random starts reach the best answer against 50 % for k-means++ (scikit-learn's greedy variant, which tries several
candidates per step; our plain version in [k-means from scratch](kmeans_from_scratch.ipynb) reaches 29 %). 
- k-means++ also comes with a guarantee: on average its result is within a factor of about $8(\ln k + 2)$ of the best possible inertia.

> In practice: use k-means++ (scikit-learn's default) and run it several times (`n_init`), keeping the lowest inertia.

## 6. Choosing k

* **Elbow method:** plot the inertia for k = 1, 2, 3, … and look for the bend, after which extra clusters buy little.
  The bend can be found automatically: rescale both axes to 0–1 and take the k farthest from the straight line between
  the curve's two ends (the "kneedle" idea, Lab 3). Often there is no clear bend.
* **Silhouette:** for each point, $s = \frac{b - a}{\max(a, b)}$, where $a$ is its mean distance to its own cluster and
  $b$ its mean distance to the nearest other cluster. Average over points; pick the k with the highest average (see
  [judging a clustering](../05-in-practice/README.md#3-judging-a-clustering) for where it misleads).
* **Cost:** both need one k-means fit per k; what each adds on top differs (measured in
  [k-means from scratch, Lab 3](kmeans_from_scratch.ipynb#lab-3-cost)).
  * **Inertia:** almost nothing. The assign step already computes each point's squared distance to its own centroid,
    so the inertia is just their sum: about n additions, no new memory (scikit-learn returns it as `inertia_`).
  * **Silhouette:** every point-to-point distance, which no k-means step computes. Building the n × n matrix takes
    about $n^2 d$ time and $n^2$ memory (once per dataset: it doesn't depend on k), then every k needs another
    pass over all $n^2$ entries.
  * At 100,000 points the matrix alone is 80 GB. scikit-learn's `silhouette_score` computes it in chunks, which bounds
    the memory but not the time; on big data pass `sample_size`.
* **What Lab 3 shows:** on four clear blobs both pick 4. On two pairs of close blobs both pick **2**: they find the
  coarsest clear structure. On uniform noise both still "pick" a k (4), but the silhouette is low (0.41 vs 0.77–0.88
  for real clusters): k-means always returns clusters, even when there are none.
* Others: the **gap statistic** (compares the inertia with that of random data), and domain knowledge ("we can run 5
  marketing campaigns").
* **Defending a k:** why the elbow matters, what goes wrong past it, and how to show it with or without true labels
  (silhouette, stability, gap) are worked through as interview questions in [interview discussion](interview_discussion.md).

## 7. Assumptions and failures

The nearest-centroid rule draws straight borders halfway between neighbouring centroids. So k-means works best when
clusters are **round, of similar size and spread**, and it fails otherwise (Lab 4, true k given):

| Data | [ARI](../05-in-practice/README.md#the-adjusted-rand-index-ari-step-by-step) | Why |
|---|---|---|
| round blobs | 1.00 | the case it assumes |
| stretched (sheared) blobs | 0.56 | long tilted clusters get cut across |
| unequal spreads | 0.79 | a wide cluster loses its edge to a tight neighbour |
| unequal sizes | 0.88 | the halfway border ignores how big each cluster is |
| moons | 0.25 | non-convex: no straight border separates them |
| circles | 0.00 | one circle inside another |

**Outliers:** every point must belong to a cluster and squared distances punish far points heavily, so a few outliers
can grab a centroid of their own (Lab 4: three far points got their own cluster and two real blobs were merged).
Options: remove outliers first, use a density method ([DBSCAN](../03-dbscan/README.md)) that labels them noise, or use
**k-medoids** (centres must be data points, less pulled by outliers).

## 8. Cost and variants

Each iteration costs about $n \cdot k \cdot d$ operations, so k-means handles millions of rows. **MiniBatchKMeans**
updates centroids from small random batches for even bigger data (Lab 4b in [the library lab](kmeans_lab.ipynb#lab-4b)
and [from scratch](kmeans_from_scratch.ipynb#lab-4b)):
* Its inertia was within 0.009 % of k-means' on 8 blobs of up to 3,000,000 points, and about 3× faster at that size.
  It was not faster at 1,000,000: easy data needs only 3 full k-means passes, and mini-batch's time depends on when its
  stopping rule fires, not on n.
* Its real strength is `partial_fit`: it learns from one chunk at a time, so the data never has to fit in memory.
* Each centroid's step must shrink (step 1 / points absorbed, a running mean); a fixed step keeps it jittering.

Implementation hurdles
([k-means from scratch](kmeans_from_scratch.ipynb)): a centroid that loses all its points (its mean is undefined), and
the memory needed to compute all point-to-centroid distances at once.

### k-means as a building block: vector search

k-means is also used inside other tools, where its job is to **partition** the data, not to discover groups. The best
known case is the IVF ("inverted file") index of [FAISS](https://github.com/facebookresearch/faiss), Meta's library
for nearest-neighbour search over embeddings:

* **Training:** k-means with `nlist` clusters runs on the vectors; its centroids become the centres of `nlist` cells,
  and every vector is stored in the cell of its nearest centroid.
* **Search:** a query is compared with the vectors in only its `nprobe` nearest cells, not with all n. More cells
  searched means better recall and slower queries.
* **k is a budget, not a discovery:** `nlist` is chosen for the speed / recall trade-off; the elbow and silhouette play
  no part, and nobody asks whether the cells are "true" groups.
* **Scale:** k-means trains on at most 256 points per centroid by default (larger training sets are subsampled); then
  all n vectors are assigned to cells, which costs only n · nlist · d and runs in parallel. This is the same "fit on a
  sample, assign everyone" idea as MiniBatchKMeans above.
* **Source:** [`faiss/IndexIVF.cpp`](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexIVF.cpp) trains
  the coarse quantizer with "k-means on the training rows" (`Clustering clus(d, nlist, cp)`, whose centroids are added
  to the quantizer); [`faiss/Clustering.h`](https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h)
  defines that class ("K-means clustering based on assignment - centroid update iterations") and
  `max_points_per_centroid = 256`.

## 9. Common misconceptions

| Misconception | Reality |
|---|---|
| "k-means finds the best clusters." | It finds a local minimum of the inertia, from its start; and only round clusters (Labs 2, 4). |
| "The elbow always shows the right k." | Often there is no clear elbow, and on pure noise the curve still bends somewhere (Lab 3). |
| "If an algorithm returns clusters, the data has clusters." | k-means cuts uniform noise into k pieces just the same (Lab 3). |

## 10. Terms

| Term | Meaning |
|---|---|
| **Centroid** | The centre (mean) of a cluster. |
| **Inertia / WCSS** | Sum of squared distances from each point to its own centroid. |
| **Lloyd's algorithm** | The assign / update loop of k-means. |
| **k-means++** | Starting centroids drawn far apart, with probability ∝ squared distance to the nearest chosen one. |
| **Elbow method** | Choose k where the inertia curve bends. |
| **Silhouette** | Per point: (distance to nearest other cluster − distance to own cluster) / the larger of the two. |

## 11. Further reading

* Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*.
