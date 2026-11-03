# OPTICS (and HDBSCAN)

> Part of the [clustering notes](../../README.md); see there for [choosing a method](../../README.md#3-choosing-a-method-the-flowchart).
> Labs in this folder: [`optics_hdbscan_lab.ipynb`](optics_hdbscan_lab.ipynb) (Lab 9, scikit-learn) and
> [`optics_from_scratch.ipynb`](optics_from_scratch.ipynb) (OPTICS in NumPy). It builds on
> [DBSCAN](../03-dbscan/README.md); distance, scaling and the scores used to judge clusters are in
> [clustering in practice](../05-in-practice/README.md).

**O**rdering **P**oints **T**o **I**dentify the **C**lustering **S**tructure: in effect, DBSCAN at every eps at once.
**Family:** density, multi-scale. Order the points so every density level is visible; finds clusters of different
densities.

1. [Definitions](#1-definitions)
2. [The algorithm](#2-the-algorithm)
3. [Extracting clusters](#3-extracting-clusters)
4. [HDBSCAN](#4-hdbscan)
5. [DBSCAN vs OPTICS: when to use which](#5-dbscan-vs-optics-when-to-use-which)
6. [Common misconceptions](#6-common-misconceptions)
7. [Terms](#7-terms)
8. [Further reading](#8-further-reading)

## 1. Definitions

* **Core distance** of $p$: the distance to its min_samples-th nearest neighbour (counting $p$). It is the smallest
  eps at which $p$ would be a core point.
* **Reachability distance** of $o$ from $p$: $\max\big(\text{core distance}(p),\ d(p, o)\big)$. Points inside a dense
  region can't look closer than that region's core distance, which smooths out random gaps.

## 2. The algorithm

1. All reachabilities start at infinity.
2. Repeatedly take the unvisited point with the **smallest** reachability so far (or any unvisited point when none is
   reachable, which starts a new region) and append it to the **ordering**.
3. If it is a core point, lower each unvisited point's reachability to its reachability through this point, whenever
   that is smaller.

Plot reachability against position in the ordering: the **reachability plot**. A dense region is visited in one go
with low values (a **valley**); moving to the next region needs a high value (a **peak**). Valleys are clusters, and
valleys inside valleys are sub-clusters. [OPTICS from scratch](optics_from_scratch.ipynb) implements it and matches
scikit-learn's ordering and reachabilities exactly.

## 3. Extracting clusters

* **Horizontal cut at height eps:** gives the same clusters as DBSCAN with that eps (up to the odd border point), from
  one OPTICS run, without refitting (Lab 9, [OPTICS from scratch](optics_from_scratch.ipynb)).
* **xi method:** finds valleys by their steep sides (a drop of at least a fraction xi), so different valleys can be at
  different heights. With xi = 0.1 it finds all three blobs of the mixed-density data that defeated DBSCAN (Lab 9), but
  xi is a setting too: 0.05 calls a third of the sparse blob noise.

## 4. HDBSCAN

**HDBSCAN** (hierarchical DBSCAN) builds a tree of the density clusters at every level and keeps the most stable
ones. It handles different densities like OPTICS but returns flat clusters plus noise automatically, with one
intuitive setting, `min_cluster_size` (Lab 9: all three blobs, 2 % noise). It is often the best default when DBSCAN's
single eps doesn't fit, and it is fast (`sklearn.cluster.HDBSCAN`).

## 5. DBSCAN vs OPTICS: when to use which

* **DBSCAN** when clusters have similar density, a sensible eps is known or easy to read from the k-distance plot, and
  speed or memory matter. It's simple and fast.
* **OPTICS** when densities vary, eps is unknown, or you want to *see* the structure (the reachability plot shows the
  hierarchy) and can afford more compute. scikit-learn's OPTICS is noticeably slower than DBSCAN and HDBSCAN (Lab 9).
* **Neither** fixes high-dimensional distances: scale the features and reduce dimensions first
  ([many features](../05-in-practice/README.md#4-many-features-the-curse-of-dimensionality)).

## 6. Common misconceptions

| Misconception | Reality |
|---|---|
| "OPTICS is a different algorithm with different results from DBSCAN." | Cutting its reachability plot at eps reproduces DBSCAN at eps (Lab 9, [OPTICS from scratch](optics_from_scratch.ipynb)). |

## 7. Terms

| Term | Meaning |
|---|---|
| **Core distance** | Distance to the min_samples-th nearest neighbour: the smallest eps making a point core. |
| **Reachability distance** | max(core distance of p, distance from p to o). |
| **Reachability plot** | Reachability in OPTICS order; valleys are clusters. |
| **xi** | OPTICS setting: the minimum steepness of a valley's sides. |
| **HDBSCAN** | Density clustering over all eps levels, keeping the most stable clusters. |

## 8. Further reading

* Ankerst, Breunig, Kriegel & Sander (1999), *OPTICS: Ordering Points To Identify the Clustering Structure*.
* Campello, Moulavi & Sander (2013), *Density-Based Clustering Based on Hierarchical Density Estimates* (HDBSCAN).
