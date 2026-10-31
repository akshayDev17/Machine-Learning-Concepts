# Clustering (unsupervised learning) — explained from zero

> **Who this is for:** someone new to machine learning. You need school algebra (averages, squares, square roots) and a
> little Python. Every claim here is demonstrated in the labs: [`lab/clustering_lab.ipynb`](lab/clustering_lab.ipynb)
> (scikit-learn / SciPy) and [`lab/clustering_from_scratch.ipynb`](lab/clustering_from_scratch.ipynb) (the same
> algorithms in NumPy, checked against the libraries). "Lab N" below means the library notebook; "scratch §N" the
> from-scratch one.

## Table of contents

0. [Clustering in 30 seconds](#0-clustering-in-30-seconds)
1. [The setting: no labels, so what is a cluster?](#1-the-setting-no-labels-so-what-is-a-cluster)
2. [The toolbox: distances, scaling, inertia](#2-the-toolbox-distances-scaling-inertia)
3. [k-means](#3-k-means)
4. [Hierarchical (agglomerative) clustering](#4-hierarchical-agglomerative-clustering)
5. [DBSCAN](#5-dbscan)
6. [OPTICS (and HDBSCAN)](#6-optics-and-hdbscan)
7. [Judging a clustering](#7-judging-a-clustering)
8. [Many features: the curse of dimensionality](#8-many-features-the-curse-of-dimensionality)
9. [Which algorithm when](#9-which-algorithm-when)
10. [Common misconceptions](#10-common-misconceptions)
11. [The labs](#11-the-labs)
12. [Glossary](#12-glossary)
13. [Further reading](#13-further-reading)
14. [Appendix - Assumptions made](#appendix---assumptions-made)

---

## 0. Clustering in 30 seconds

* **Clustering** splits rows of data into groups (**clusters**) so that rows in the same group are similar and rows in
  different groups are not, **without** being told what the groups are. That is what makes it *unsupervised*: there
  is no target column to learn from.
* Uses: customer segments, grouping documents or images, finding unusual points (the ones that fit no group),
  compressing data (replace each point by its cluster's centre), exploring a new dataset.
* Four families are covered here:

| Family | Idea | Algorithm | Finds |
|---|---|---|---|
| **Centroid** | each cluster is a centre; points belong to the nearest one | **k-means** (§3) | round clusters; you give k |
| **Hierarchical** | merge the closest clusters step by step into a tree | **agglomerative** (§4) | a tree of clusters at every scale |
| **Density** | a cluster is a dense region; sparse points are noise | **DBSCAN** (§5) | any shape; flags noise; you give a radius |
| **Density, multi-scale** | order the points so every density level is visible | **OPTICS**, **HDBSCAN** (§6) | clusters of different densities |

* No algorithm is best everywhere (Lab 12). Each one carries an assumption about what a cluster *is*, and the
  results are only as good as the match between that assumption and your data.

## 1. The setting: no labels, so what is a cluster?

A dataset here is a table: $n$ rows (points, samples) and $d$ numeric columns (features). Each row is a point in
$d$-dimensional space. The output of clustering is one **label** per row: a cluster number, or −1 for "noise" in the
density methods.

There is no single definition of a cluster, and each algorithm picks one:

* **Compact:** points close to a common centre (k-means, ward linkage).
* **Connected:** points linked by a chain of close neighbours (single linkage, DBSCAN).
* **Dense:** a region with many points per unit of space, separated by sparser regions (DBSCAN, OPTICS, HDBSCAN).
* **Nested:** clusters inside clusters (hierarchical clustering shows every level).

Lab 3 shows why the definition matters: four blobs arranged as two pairs are "2 clusters" at one scale and
"4 clusters" at another, and both answers are right.

## 2. The toolbox: distances, scaling, inertia

### 2.1 Distance

Every method here measures how far apart two points are. The default is the **Euclidean distance** (straight line):

$$d(x, y) = \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2}$$

Others: **Manhattan** ($\sum_j |x_j - y_j|$, less sensitive to one large difference) and **cosine distance**
($1 - \cos\theta$, the angle between two vectors, common for text where only the direction of a word-count vector
matters, not its length). k-means is tied to squared Euclidean distance (its centroid is a mean); hierarchical,
DBSCAN and OPTICS accept any distance.

### 2.2 Scaling

A distance adds up differences over all features, so a feature measured in big numbers dominates. In the wine data
(Lab 10) the "proline" column spreads about 2,600 times more than the smallest one; unscaled, k-means essentially
clusters on proline alone (ARI 0.37), and after **standardising** (each feature: subtract its mean, divide by its
standard deviation) it scores 0.90. Standardise unless the features share a unit and a scale that mean something
(pixels, coordinates in metres).

### 2.3 Inertia (within-cluster sum of squares, WCSS)

For a clustering with centres $\mu_1, \dots, \mu_k$, the **inertia** is the total squared distance from each point to
its own cluster's centre:

$$\text{inertia} = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

Small inertia = tight clusters. It always falls as k grows (with $k = n$ every point is its own centre and the inertia
is 0), so it can't be used alone to choose k (§3.5).

> **Why "inertia"?** It is the physics *moment of inertia* $I_c = \sum_i m_i \lVert x_i - c \rVert^2$ with every point
> given mass $m_i = 1$ and measured about its own centroid: the inertia is each cluster's spread around its centre of
> mass. Two physics facts carry over: the centre of mass minimises $I_c$ (which is why k-means moves each centroid to
> the mean), and the parallel-axis theorem splits the total spread as
> $$\underbrace{\sum_i \lVert x_i - \mu \rVert^2}_{\text{total}} = \underbrace{\sum_i \lVert x_i - \mu_{c(i)} \rVert^2}_{\text{within (inertia)}} + \underbrace{\sum_k n_k \lVert \mu_k - \mu \rVert^2}_{\text{between}}$$
> so making clusters tight is the same as pushing their centroids apart.

## 3. k-means

### 3.1 The goal

Choose $k$ centres (**centroids**) and assign every point to one of them so that the inertia is as small as possible.
Finding the very best answer is computationally hard in general, so k-means finds a good local answer with a simple
loop.

### 3.2 The algorithm (Lloyd's algorithm)

1. Pick $k$ starting centroids (§3.4).
2. **Assign:** give every point to its nearest centroid.
3. **Update:** move every centroid to the **mean** of the points assigned to it.
4. Repeat 2–3 until the assignments stop changing (equivalently, no centroid moves).

Lab 1 shows the centroids walking to their final places; scratch §1 implements the two steps and matches
scikit-learn's labels and inertia exactly from the same start.

### 3.3 Why it always stops

* The **assign** step can only lower the inertia: each point moves to a centroid at least as close as its old one.
* The **update** step can only lower it too: for a fixed group of points, the point $\mu$ that minimises
  $\sum_i \lVert x_i - \mu \rVert^2$ is their mean. (Setting the derivative $-2\sum_i (x_i - \mu)$ to zero gives
  $\mu = \frac{1}{n}\sum_i x_i$.)
* The inertia can't fall forever (there are finitely many ways to assign the points), so the loop stops, typically
  after a few to a few dozen iterations. Lab 1: 4757 → 1719 in 15 iterations, never rising.

It stops at a **local** minimum, which depends on the start.

### 3.4 The start matters: k-means++

With random starting centroids, two can land in the same true cluster and none in another; the loop then rarely
recovers.
- **k-means++** spreads the starts out: the first centroid is a random point from the sample; each next centroid is drawn from the remaining samples with probability
proportional to $D(x)^2$, the squared distance to the nearest centroid already chosen. 
- On 10 blobs (Lab 2), only 3% of random starts reach the best answer against 50 % for k-means++ (scikit-learn's greedy variant, which tries several
candidates per step; our plain version in scratch §1 reaches 29 %). 
- k-means++ also comes with a guarantee: on average its result is within a factor of about $8(\ln k + 2)$ of the best possible inertia.

> In practice: use k-means++ (scikit-learn's default) and run it several times (`n_init`), keeping the lowest inertia.

### 3.5 Choosing k

* **Elbow method:** plot the inertia for k = 1, 2, 3, … and look for the bend, after which extra clusters buy little.
  The bend can be found automatically: rescale both axes to 0–1 and take the k farthest from the straight line between
  the curve's two ends (the "kneedle" idea, Lab 3). Often there is no clear bend.
* **Silhouette:** for each point, $s = \frac{b - a}{\max(a, b)}$, where $a$ is its mean distance to its own cluster and
  $b$ its mean distance to the nearest other cluster. Average over points; pick the k with the highest average (§7).
* **What Lab 3 shows:** on four clear blobs both pick 4. On two pairs of close blobs both pick **2**: they find the
  coarsest clear structure. On uniform noise both still "pick" a k (4), but the silhouette is low (0.41 vs 0.77–0.88
  for real clusters): k-means always returns clusters, even when there are none.
* Others: the **gap statistic** (compares the inertia with that of random data), and domain knowledge ("we can run 5
  marketing campaigns").

### 3.6 Assumptions and failures

The nearest-centroid rule draws straight borders halfway between neighbouring centroids. So k-means works best when
clusters are **round, of similar size and spread**, and it fails otherwise (Lab 4, true k given):

| Data | ARI | Why |
|---|---|---|
| round blobs | 1.00 | the case it assumes |
| stretched (sheared) blobs | 0.56 | long tilted clusters get cut across |
| unequal spreads | 0.79 | a wide cluster loses its edge to a tight neighbour |
| unequal sizes | 0.88 | the halfway border ignores how big each cluster is |
| moons | 0.25 | non-convex: no straight border separates them |
| circles | 0.00 | one circle inside another |

**Outliers:** every point must belong to a cluster and squared distances punish far points heavily, so a few outliers
can grab a centroid of their own (Lab 4: three far points got their own cluster and two real blobs were merged).
Options: remove outliers first, use a density method (§5) that labels them noise, or use **k-medoids** (centres must be
data points, less pulled by outliers).

### 3.7 Cost and variants

Each iteration costs about $n \cdot k \cdot d$ operations, so k-means handles millions of rows. **MiniBatchKMeans**
updates centroids from small random batches for even bigger data. Implementation hurdles (scratch §1): a centroid that
loses all its points (its mean is undefined), and the memory needed to compute all point-to-centroid distances at once.

## 4. Hierarchical (agglomerative) clustering

### 4.1 The algorithm

1. Start with every point as its own cluster ($n$ clusters).
2. Find the two **closest clusters** and merge them.
3. Update the distances from the merged cluster to all others.
4. Repeat until one cluster is left.

The record of merges is a binary tree, the **dendrogram**: each join is drawn at the distance at which the two
clusters merged. **Cutting** the tree with a horizontal line at some height (or asking for k clusters) gives a flat
clustering; any number of clusters can be read from the same tree (Lab 5). A large jump between consecutive merge
heights is a natural place to cut.

The other direction, **divisive** clustering (start with one cluster and split), exists but is rarely used.

### 4.2 Linkage: the distance between two clusters

| Linkage | Distance between clusters A and B | Behaviour (Lab 6) |
|---|---|---|
| **single** | closest pair of points | follows chains: finds moons (ARI 1.00), but one thin bridge of points merges two blobs (0.00) |
| **complete** | farthest pair | compact clusters of similar diameter; handles the bridge (0.99) |
| **average** | mean of all point pairs | between single and complete |
| **ward** | how much the total within-cluster sum of squares grows if A and B merge | round clusters of similar size, like k-means; usually the best default |
| centroid | distance between the two means | rarely used: merge heights can go *down*, which makes the tree hard to read |

After a merge, the distance from the new cluster to any other follows from distances already known (the
**Lance–Williams formula**, scratch §3), so point distances are computed only once.

### 4.3 Judging the tree: the cophenetic correlation

The **cophenetic distance** of two points is the height at which they first share a cluster. Its correlation with the
real distances measures how faithfully the tree summarises the data (1 = perfectly). On iris, average linkage keeps
the distances best (0.854) while ward gives the best 3-cluster split (ARI 0.62): a faithful tree and a good cut are not
the same thing.

### 4.4 Cost and use cases

The distance matrix needs $n^2$ memory; the naive algorithm takes about $n^3$ time (scratch §3), good libraries about
$n^2$. That limits it to roughly tens of thousands of points. Use it when the data is small enough, when you want to
see structure at **several scales** (clusters with sub-clusters: product categories, species, document topics), or
when you don't know k and want to choose by looking. Ties between equal distances are common with rounded
measurements; they change the order of merges but not the heights (scratch §3).

## 5. DBSCAN

**D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise.

### 5.1 Definitions

Two settings: a radius **eps** ($\varepsilon$) and a count **min_samples**.

* **Neighbourhood** of a point: all points within distance eps, the point itself included (scikit-learn's convention).
* **Core point:** its neighbourhood holds at least min_samples points.
* **Border point:** not a core point, but within eps of a core point.
* **Noise point:** neither.
* Two points are **density-connected** if a chain of core points, each within eps of the next, links them.

### 5.2 The algorithm

1. Label every point core or not.
2. Go through the points; each core point not yet in a cluster starts a new cluster.
3. Grow the cluster outwards: every core point in it adds all points in its neighbourhood. Border points join but
   don't add their own neighbours.
4. Whatever was never added is noise (label −1).

A border point within eps of core points from two clusters gets only one label, that of whichever cluster reaches it
first, so it depends on the order of the data (scratch §4). The original notes' rule "assign it to the nearest core
point" is a reasonable variant, but it's not what the original paper or scikit-learn does. Core points and noise never
depend on the order.

### 5.3 Choosing eps and min_samples

* **min_samples:** a common start is $2 \times d$ (twice the number of features), at least 3–5. Larger values smooth
  more and call more points noise.
* **eps:** the **k-distance plot**. For every point, take the distance to its min_samples-th nearest neighbour; sort
  these and plot them. Points inside clusters give small, flat values; noise gives a steep rise. Pick eps at the bend
  (Lab 7: eps = 0.164 on two moons with strays, giving a perfect split and 13 of 20 strays flagged as noise).
* Domain units help: "customers within 2 km of each other" is an eps.

### 5.4 Strengths and weaknesses

* **Strengths:** finds clusters of any shape (moons: ARI 1.00 where k-means gets 0.26), needs no k, labels noise.
* **Sensitive to eps** (Lab 8): on the moons only eps 0.15–0.2 works; smaller shatters them, larger merges them.
* **One density for all clusters** (Lab 8): with two dense blobs close together and one sparse blob far away, a small
  eps keeps the dense ones apart but calls the whole sparse blob noise; an eps large enough to find the sparse blob
  merges the dense ones. No single eps works.
* **High dimensions:** distances between points become similar (§8), so a radius loses meaning.
* Cost: with a spatial index, roughly $n \log n$ in low dimensions; up to $n^2$ in the worst case.

## 6. OPTICS (and HDBSCAN)

**O**rdering **P**oints **T**o **I**dentify the **C**lustering **S**tructure: in effect, DBSCAN at every eps at once.

### 6.1 Definitions

* **Core distance** of $p$: the distance to its min_samples-th nearest neighbour (counting $p$). It is the smallest
  eps at which $p$ would be a core point.
* **Reachability distance** of $o$ from $p$: $\max\big(\text{core distance}(p),\ d(p, o)\big)$. Points inside a dense
  region can't look closer than that region's core distance, which smooths out random gaps.

### 6.2 The algorithm

1. All reachabilities start at infinity.
2. Repeatedly take the unvisited point with the **smallest** reachability so far (or any unvisited point when none is
   reachable, which starts a new region) and append it to the **ordering**.
3. If it is a core point, lower each unvisited point's reachability to its reachability through this point, whenever
   that is smaller.

Plot reachability against position in the ordering: the **reachability plot**. A dense region is visited in one go
with low values (a **valley**); moving to the next region needs a high value (a **peak**). Valleys are clusters, and
valleys inside valleys are sub-clusters. Scratch §5 implements it and matches scikit-learn's ordering and
reachabilities exactly.

### 6.3 Extracting clusters

* **Horizontal cut at height eps:** gives the same clusters as DBSCAN with that eps (up to the odd border point), from
  one OPTICS run, without refitting (Lab 9, scratch §5).
* **xi method:** finds valleys by their steep sides (a drop of at least a fraction xi), so different valleys can be at
  different heights. With xi = 0.1 it finds all three blobs of the mixed-density data that defeated DBSCAN (Lab 9), but
  xi is a setting too: 0.05 calls a third of the sparse blob noise.

### 6.4 HDBSCAN

**HDBSCAN** (hierarchical DBSCAN) builds a tree of the density clusters at every level and keeps the most stable
ones. It handles different densities like OPTICS but returns flat clusters plus noise automatically, with one
intuitive setting, `min_cluster_size` (Lab 9: all three blobs, 2 % noise). It is often the best default when DBSCAN's
single eps doesn't fit, and it is fast (`sklearn.cluster.HDBSCAN`).

### 6.5 DBSCAN vs OPTICS: when to use which

* **DBSCAN** when clusters have similar density, a sensible eps is known or easy to read from the k-distance plot, and
  speed or memory matter. It's simple and fast.
* **OPTICS** when densities vary, eps is unknown, or you want to *see* the structure (the reachability plot shows the
  hierarchy) and can afford more compute. scikit-learn's OPTICS is noticeably slower than DBSCAN and HDBSCAN (Lab 9).
* **Neither** fixes high-dimensional distances: scale the features and reduce dimensions first (§8).

## 7. Judging a clustering

**With true labels** (only when testing a method on data where the answer is known):

* **ARI (adjusted Rand index):** over all pairs of points, how often the clustering and the truth agree on "same group"
  vs "different group", corrected so random labels score 0; 1 = identical. In the labs, noise points count as one
  extra group.
* **NMI (normalised mutual information):** how much a point's cluster tells you about its true group, 0–1.

**Without labels** (the usual situation):

* **Silhouette** (§3.5): −1 to 1, higher is better.
* **Davies–Bouldin:** for each cluster, the worst (spread of i + spread of j) / distance between their centres,
  averaged; lower is better.
* **Calinski–Harabasz:** between-cluster spread / within-cluster spread, adjusted for k; higher is better.

All three internal scores reward **round, well-separated** clusters. On the moons (Lab 11) DBSCAN is exactly right
(ARI 1.00) and k-means wrong (0.25), yet silhouette (0.05 vs 0.49), Davies–Bouldin and Calinski–Harabasz all prefer
k-means. Use them to compare settings of one method that makes round clusters, not to compare methods. The real test
is whether the clusters mean something to someone who knows the domain, and whether they are **stable** (similar
clusters on a resample of the data).

## 8. Many features: the curse of dimensionality

With many features, distances between random points become nearly equal: the farthest-to-nearest ratio falls from
hundreds in 2-D to about 1.2 in 1,000-D (PCA lab, Lab 0). "Nearest centroid", "within eps" and "closest cluster" all
lose meaning, and every irrelevant feature adds noise to every distance. Remedies: select the relevant features, or
**reduce dimensions** with PCA first (the PCA lab's Lab 17 shows k-means, agglomerative and Gaussian mixtures improving
after PCA on noisy data; Lab 12 here clusters the 64-pixel digits on 20 principal components). For visualising
clusters, t-SNE or UMAP on top of PCA is standard.

## 9. Which algorithm when

| Situation | First choice | Avoid |
|---|---|---|
| Round clusters, similar sizes, k known or easy to pick, lots of data | **k-means** (k-means++, several `n_init`) | single linkage |
| Want structure at several scales, or a tree; up to ~10⁴ rows | **agglomerative**: ward for compact clusters, single for chains with clean gaps | k-means if sub-clusters matter |
| Any shape, noise to flag, similar densities, meaningful distance scale | **DBSCAN** (eps from the k-distance plot) | k-means |
| Different densities, or eps unknown | **HDBSCAN**; **OPTICS** to explore the reachability plot | DBSCAN |
| Dozens to thousands of features | **reduce first** (PCA), then cluster | raw distances |
| Need cluster probabilities or elliptical clusters | Gaussian mixture models (not covered here) | k-means |

Always scale features (§2.2), try more than one method, and check that clusters make sense; one internal score is not
a verdict (§7).

## 10. Common misconceptions

| Misconception | Reality |
|---|---|
| "k-means finds the best clusters." | It finds a local minimum of the inertia, from its start; and only round clusters (Labs 2, 4). |
| "The elbow always shows the right k." | Often there is no clear elbow, and on pure noise the curve still bends somewhere (Lab 3). |
| "If an algorithm returns clusters, the data has clusters." | k-means cuts uniform noise into k pieces just the same (Lab 3). |
| "A higher silhouette means a better clustering." | Only among round clusters; it prefers the wrong k-means split of the moons over the right DBSCAN one (Lab 11). |
| "DBSCAN needs no tuning." | It needs eps, and the right range can be narrow (Lab 8). |
| "DBSCAN assigns a border point to the nearest core point's cluster." | It goes to whichever cluster reaches it first; the order of the data decides (scratch §4). |
| "OPTICS is a different algorithm with different results from DBSCAN." | Cutting its reachability plot at eps reproduces DBSCAN at eps (Lab 9, scratch §5). |
| "Scaling doesn't matter for clustering." | It can be the difference between ARI 0.37 and 0.90 (Lab 10). |
| "Hierarchical clustering is always O(n³)." | The naive version is; good implementations are about n² (scratch §3 vs SciPy). |

## 11. The labs

Everything above is demonstrated in [`lab/`](lab/README.md): setup, run instructions and a map of which lab answers
which question are in **[`lab/README.md`](lab/README.md)**.

| Notebook | Contents | Sections |
|---|---|---|
| [`clustering_lab.ipynb`](lab/clustering_lab.ipynb) | 12 labs in 4 parts with scikit-learn / SciPy: k-means (1–4), hierarchical (5–6), density (7–9), in practice (10–12) | 2–9 |
| [`clustering_from_scratch.ipynb`](lab/clustering_from_scratch.ipynb) | k-means + k-means++, silhouette, agglomerative (Lance–Williams), DBSCAN, OPTICS in NumPy, each checked against the library, with implementation hurdles | 3–6 |

The original notebooks this folder grew from are kept in [`_archive/`](_archive/).

## 12. Glossary

| Term | Meaning |
|---|---|
| **Cluster** | A group of points that are similar to each other and different from other groups. |
| **Unsupervised** | Learning without a target column: the algorithm only sees the features. |
| **Centroid** | The centre (mean) of a cluster. |
| **Inertia / WCSS** | Sum of squared distances from each point to its own centroid. |
| **Lloyd's algorithm** | The assign / update loop of k-means. |
| **k-means++** | Starting centroids drawn far apart, with probability ∝ squared distance to the nearest chosen one. |
| **Elbow method** | Choose k where the inertia curve bends. |
| **Silhouette** | Per point: (distance to nearest other cluster − distance to own cluster) / the larger of the two. |
| **Agglomerative clustering** | Merge the two closest clusters, repeatedly, from single points up to one cluster. |
| **Linkage** | The rule for the distance between two clusters (single, complete, average, ward). |
| **Dendrogram** | The tree of merges, each drawn at its merge distance. |
| **Lance–Williams formula** | Updates cluster distances after a merge from distances already known. |
| **Cophenetic distance** | The height at which two points first share a cluster in the dendrogram. |
| **eps (ε)** | DBSCAN's neighbourhood radius. |
| **min_samples** | Points needed in a neighbourhood (itself included) for a core point. |
| **Core / border / noise point** | Dense enough to start a cluster / near a core point / neither. |
| **Density-connected** | Linked by a chain of core points, each within eps of the next. |
| **k-distance plot** | Sorted distances to the k-th nearest neighbour; its bend suggests eps. |
| **Core distance** | Distance to the min_samples-th nearest neighbour: the smallest eps making a point core. |
| **Reachability distance** | max(core distance of p, distance from p to o). |
| **Reachability plot** | Reachability in OPTICS order; valleys are clusters. |
| **xi** | OPTICS setting: the minimum steepness of a valley's sides. |
| **HDBSCAN** | Density clustering over all eps levels, keeping the most stable clusters. |
| **ARI / NMI** | Agreement with known true groups (1 = identical, ~0 = random). |
| **Davies–Bouldin / Calinski–Harabasz** | Label-free scores: lower / higher is better. |
| **Standardising** | Subtract each feature's mean and divide by its standard deviation. |

## 13. Further reading

* scikit-learn user guide, [Clustering](https://scikit-learn.org/stable/modules/clustering.html) and the
  [comparison of clustering algorithms](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
  (the six shapes of Lab 4).
* Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*.
* Ester, Kriegel, Sander & Xu (1996), *A density-based algorithm for discovering clusters in large spatial databases
  with noise* (DBSCAN).
* Ankerst, Breunig, Kriegel & Sander (1999), *OPTICS: Ordering Points To Identify the Clustering Structure*.
* Campello, Moulavi & Sander (2013), *Density-Based Clustering Based on Hierarchical Density Estimates* (HDBSCAN).
* Müllner (2011), *Modern hierarchical, agglomerative clustering algorithms* (the fast algorithms behind SciPy).
* Naftali Harris, [Visualizing DBSCAN clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
  (interactive; from the original notes).
* Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, §14.3 (cluster analysis).

## Appendix - Assumptions made

1. **Scope.** The folder held four notebooks (k-means, hierarchical, DBSCAN, OPTICS), so those four algorithms are the
   core, plus HDBSCAN as the modern follow-up to OPTICS and the scores used to judge clusters. Gaussian mixture models,
   spectral clustering and mean shift are only mentioned.
2. **Structure.** As requested: this theory README, and a `lab/` folder with its own README, one notebook using the
   library implementations and one from-scratch notebook with markdown explanations and line-by-line code comments.
   The folder is called `lab/` like the PCA, LDA and Feature-Eng folders.
3. **Original notes.** Their content (algorithm steps, the elbow method and its programmatic detection, the problems
   lists, the hierarchical use cases, the DBSCAN reference) is merged into the sections above; the notebooks themselves
   were moved to `_archive/` unchanged. One rule was corrected: DBSCAN gives a border point to the first cluster that
   reaches it, not to the nearest core point (§5.2). The open questions in the notes are answered: how to choose eps and
   min_samples (§5.3), what to do with outliers in k-means (§3.6), what happens with sub-clusters, varying densities and
   sizes, and non-round shapes (Labs 3, 4, 8).
4. **Data.** Synthetic 2-D data (so results can be seen) plus scikit-learn's bundled iris, wine and digits; no
   internet needed. "True" groups are used only to grade results with ARI.
5. **Numbers.** Every number quoted here comes from the executed notebooks (scikit-learn 1.9.1, NumPy 2.5, SciPy 1.18,
   Python 3.12.8). Timings depend on the machine and are only described qualitatively.
