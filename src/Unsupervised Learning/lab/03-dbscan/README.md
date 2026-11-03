# DBSCAN

> Part of the [clustering notes](../../README.md); see there for [choosing a method](../../README.md#3-choosing-a-method-the-flowchart).
> Labs in this folder: [`dbscan_lab.ipynb`](dbscan_lab.ipynb) (Labs 7–8, scikit-learn) and
> [`dbscan_from_scratch.ipynb`](dbscan_from_scratch.ipynb) (the same algorithm in NumPy). Distance, scaling and the
> scores used to judge clusters are in [clustering in practice](../05-in-practice/README.md).

**D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise. **Family:** density. A cluster is
a dense region; sparse points are noise. Finds any shape; you give a radius.

> **The words in the name.**
> * *Noise:* points in sparse areas that belong to no cluster (stray readings, outliers, background clutter).
>   * DBSCAN leaves them out (label −1, [§1](#1-definitions)).
>   * k-means, by contrast, must force every point into a cluster.
> * *Applications:* not a technical term.
>   * The [1996 paper](https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/) targeted large spatial databases (geographic data, satellite images).
>   * "Clustering of applications with noise" just means clustering for such uses, where the data contains noise.
>   * Nothing in the algorithm depends on it.

1. [Definitions](#1-definitions)
2. [The algorithm](#2-the-algorithm)
3. [Choosing eps and min_samples](#3-choosing-eps-and-min_samples)
4. [Strengths and weaknesses](#4-strengths-and-weaknesses)
5. [Common misconceptions](#5-common-misconceptions)
6. [Terms](#6-terms)
7. [Further reading](#7-further-reading)

## 1. Definitions

Two settings: a radius **eps** ($\varepsilon$) and a count **min_samples**.

* **Neighbourhood** of a point: all points within distance eps, the point itself included (scikit-learn's convention).
* **Core point:** its neighbourhood holds at least min_samples points.
* **Border point:** not a core point, but within eps of a core point.
* **Noise point:** neither.
* Two points are **density-connected** if a chain of core points, each within eps of the next, links them.

## 2. The algorithm

1. Label every point core or not.
2. Go through the points; each core point not yet in a cluster starts a new cluster.
3. Grow the cluster outwards: every core point in it adds all points in its neighbourhood. Border points join but
   don't add their own neighbours.
4. Whatever was never added is noise (label −1).

A border point within eps of core points from two clusters gets only one label, that of whichever cluster reaches it
first, so it depends on the order of the data ([DBSCAN from scratch](dbscan_from_scratch.ipynb)). The original notes'
rule "assign it to the nearest core point" is a reasonable variant, but it's not what the original paper or
scikit-learn does. Core points and noise never depend on the order.

## 3. Choosing eps and min_samples

* **min_samples:** a common start is $2 \times d$ (twice the number of features), at least 3–5. Larger values smooth
  more and call more points noise.
* **eps:** the **k-distance plot**. For every point, take the distance to its min_samples-th nearest neighbour; sort
  these and plot them. Points inside clusters give small, flat values; noise gives a steep rise. Pick eps at the bend
  (Lab 7: eps = 0.164 on two moons with strays, giving a perfect split and 13 of 20 strays flagged as noise).
* Domain units help: "customers within 2 km of each other" is an eps.

## 4. Strengths and weaknesses

* **Strengths:** finds clusters of any shape (moons: ARI 1.00 where k-means gets 0.26), needs no k, labels noise.
* **Sensitive to eps** (Lab 8): on the moons only eps 0.15–0.2 works; smaller shatters them, larger merges them.
* **One density for all clusters** (Lab 8): with two dense blobs close together and one sparse blob far away, a small
  eps keeps the dense ones apart but calls the whole sparse blob noise; an eps large enough to find the sparse blob
  merges the dense ones. No single eps works; [OPTICS and HDBSCAN](../04-optics-hdbscan/README.md) handle this.
* **High dimensions:** distances between points become similar
  ([many features](../05-in-practice/README.md#4-many-features-the-curse-of-dimensionality)), so a radius loses
  meaning.
* Cost: with a spatial index, roughly $n \log n$ in low dimensions; up to $n^2$ in the worst case.

## 5. Common misconceptions

| Misconception | Reality |
|---|---|
| "DBSCAN needs no tuning." | It needs eps, and the right range can be narrow (Lab 8). |
| "DBSCAN assigns a border point to the nearest core point's cluster." | It goes to whichever cluster reaches it first; the order of the data decides ([DBSCAN from scratch](dbscan_from_scratch.ipynb)). |

## 6. Terms

| Term | Meaning |
|---|---|
| **eps (ε)** | DBSCAN's neighbourhood radius. |
| **min_samples** | Points needed in a neighbourhood (itself included) for a core point. |
| **Core / border / noise point** | Dense enough to start a cluster / near a core point / neither. |
| **Density-connected** | Linked by a chain of core points, each within eps of the next. |
| **k-distance plot** | Sorted distances to the k-th nearest neighbour; its bend suggests eps. |

## 7. Further reading

* Ester, Kriegel, Sander & Xu (1996),
  [*A density-based algorithm for discovering clusters in large spatial databases with noise*](https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/)
  (DBSCAN; [PDF](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)).
* Naftali Harris, [Visualizing DBSCAN clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
  (interactive; from the original notes).
