# Hierarchical (agglomerative) clustering

> Part of the [clustering notes](../../README.md); see there for [choosing a method](../../README.md#3-choosing-a-method-the-flowchart).
> Labs in this folder: [`hierarchical_lab.ipynb`](hierarchical_lab.ipynb) (Labs 5–6, SciPy and scikit-learn) and
> [`hierarchical_from_scratch.ipynb`](hierarchical_from_scratch.ipynb) (the same algorithm in NumPy). Distance, scaling
> and the scores used to judge clusters are in [clustering in practice](../05-in-practice/README.md).

**Family:** hierarchical. Merge the closest clusters step by step into a tree; any number of clusters can be read from
it.

1. [The algorithm](#1-the-algorithm)
2. [Linkage: the distance between two clusters](#2-linkage-the-distance-between-two-clusters)
3. [Judging the tree: the cophenetic correlation](#3-judging-the-tree-the-cophenetic-correlation)
4. [Cost and use cases](#4-cost-and-use-cases)
5. [Common misconceptions](#5-common-misconceptions)
6. [Terms](#6-terms)
7. [Further reading](#7-further-reading)

## 1. The algorithm

1. Start with every point as its own cluster ($n$ clusters).
2. Find the two **closest clusters** and merge them.
3. Update the distances from the merged cluster to all others.
4. Repeat until one cluster is left.

The record of merges is a binary tree, the **dendrogram**: each join is drawn at the distance at which the two
clusters merged. **Cutting** the tree with a horizontal line at some height (or asking for k clusters) gives a flat
clustering; any number of clusters can be read from the same tree (Lab 5). A large jump between consecutive merge
heights is a natural place to cut.

The other direction, **divisive** clustering (start with one cluster and split), exists but is rarely used.

## 2. Linkage: the distance between two clusters

| Linkage | Distance between clusters A and B | Behaviour (Lab 6) |
|---|---|---|
| **single** | closest pair of points | follows chains: finds moons (ARI 1.00), but one thin bridge of points merges two blobs (0.00) |
| **complete** | farthest pair | compact clusters of similar diameter; handles the bridge (0.99) |
| **average** | mean of all point pairs | between single and complete |
| **ward** | how much the total within-cluster sum of squares grows if A and B merge | round clusters of similar size, like k-means; usually the best default |
| centroid | distance between the two means | rarely used: merge heights can go *down*, which makes the tree hard to read |

After a merge, the distance from the new cluster to any other follows from distances already known (the
**Lance–Williams formula**, [hierarchical from scratch](hierarchical_from_scratch.ipynb)), so point distances are
computed only once.

## 3. Judging the tree: the cophenetic correlation

The **cophenetic distance** of two points is the height at which they first share a cluster. Its correlation with the
real distances measures how faithfully the tree summarises the data (1 = perfectly). On iris, average linkage keeps
the distances best (0.854) while ward gives the best 3-cluster split (ARI 0.62): a faithful tree and a good cut are not
the same thing.

## 4. Cost and use cases

The distance matrix needs $n^2$ memory; the naive algorithm takes about $n^3$ time
([hierarchical from scratch](hierarchical_from_scratch.ipynb)), good libraries about $n^2$. That limits it to roughly
tens of thousands of points. Use it when the data is small enough, when you want to see structure at **several
scales** (clusters with sub-clusters: product categories, species, document topics), or when you don't know k and want
to choose by looking. Ties between equal distances are common with rounded measurements; they change the order of
merges but not the heights ([hierarchical from scratch](hierarchical_from_scratch.ipynb)).

## 5. Common misconceptions

| Misconception | Reality |
|---|---|
| "Hierarchical clustering is always O(n³)." | The naive version is; good implementations are about n² ([hierarchical from scratch](hierarchical_from_scratch.ipynb) vs SciPy). |

## 6. Terms

| Term | Meaning |
|---|---|
| **Agglomerative clustering** | Merge the two closest clusters, repeatedly, from single points up to one cluster. |
| **Linkage** | The rule for the distance between two clusters (single, complete, average, ward). |
| **Dendrogram** | The tree of merges, each drawn at its merge distance. |
| **Lance–Williams formula** | Updates cluster distances after a merge from distances already known. |
| **Cophenetic distance** | The height at which two points first share a cluster in the dendrogram. |

## 7. Further reading

* Müllner (2011), *Modern hierarchical, agglomerative clustering algorithms* (the fast algorithms behind SciPy).
