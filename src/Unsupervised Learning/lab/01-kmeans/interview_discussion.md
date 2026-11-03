# Interview discussion: choosing k, and defending it

> Part of the [k-means notes](README.md) ([§6 Choosing k](README.md#6-choosing-k)). Questions a senior data scientist
> might ask about the elbow method, and how to answer them with evidence.
> All numbers are from library [Lab 3](kmeans_lab.ipynb#lab-3)'s "4 clear blobs" (400 points, 2 features, 4 true
> groups); the code that produces them is at the [end](#reproduce-the-numbers).

1. [Why does the elbow matter? Why not keep lowering the inertia?](#1-why-does-the-elbow-matter-why-not-keep-lowering-the-inertia)
2. [What goes wrong if you go past the elbow?](#2-what-goes-wrong-if-you-go-past-the-elbow)
3. [How do you show that? First: are true labels available?](#3-how-do-you-show-that-first-are-true-labels-available)
4. [The silhouette needs every pairwise distance. What about large data?](#4-the-silhouette-needs-every-pairwise-distance-what-about-large-data)
5. [What does "ARI run vs run" mean? Is it comparing different k?](#5-what-does-ari-run-vs-run-mean-is-it-comparing-different-k)
6. [Isn't checking the elbow with the silhouette circular? Why not just use the silhouette?](#6-isnt-checking-the-elbow-with-the-silhouette-circular-why-not-just-use-the-silhouette)
7. [Answers to give](#7-answers-to-give)

**The evidence table** (used throughout):

| k | inertia | drop from k − 1 | drop on uniform data | ratio | silhouette | stability (ARI run vs run) | ARI vs truth |
|---|---|---|---|---|---|---|---|
| 1 | 20,725 | — | — | — | — | — | — |
| 2 | 10,691 | 10,034 | 8,229 | 1.2 | 0.50 | 1.00 | 0.50 |
| 3 | 5,780 | 4,911 | 4,234 | 1.2 | 0.59 | 0.80 | 0.71 |
| **4** | **1,136** | **4,644** | **2,529** | **1.8** | **0.77** | **1.00** | **1.00** |
| 5 | 1,019 | 117 | 865 | 0.1 | 0.66 | 0.87 | 0.91 |
| 6 | 905 | 114 | 717 | 0.2 | 0.56 | 0.84 | 0.82 |
| 7 | 803 | 102 | 554 | 0.2 | 0.45 | 0.86 | 0.72 |

## 1. Why does the elbow matter? Why not keep lowering the inertia?

* **The inertia always falls as k grows.** At k = n every point is its own centre and the inertia is 0.
  * That answer is useless: it says "each point is like itself" and summarises nothing.
  * So the inertia is the right thing to minimise *for a fixed k*, but not for *choosing* k. It behaves like training
    error, which always falls as a model gets more complex.
* **Why the drop changes sharply at the true k:**
  * *Before it*, a new centroid **separates two real groups** that were sharing one centre. That centre sat far from
    both, so a lot of squared distance disappears: drops of 10,034, 4,911 and 4,644 up to k = 4.
  * *After it*, a new centroid can only **cut a real, round group in half**. Its points were already close to their
    centre, so little disappears: drops of 117, 114, 102. About 40 times less.
  * That cliff is the elbow.
* **When going past the elbow is fine:** compression rather than discovery (colour quantisation, vector quantisation,
  codebooks). There is no "true" k; k is a budget (say 16 colours), and more clusters simply copy the data better.
  * A production example: FAISS's IVF vector-search index runs k-means to split embeddings into cells and picks the
    number of cells for search speed, not by any elbow
    ([README §8](README.md#k-means-as-a-building-block-vector-search)).

## 2. What goes wrong if you go past the elbow?

* **The extra clusters aren't real.** Where a round blob gets cut is arbitrary and moves with the random start or the
  sample, so the result does not reproduce.
* **You fit noise.** Each extra centroid adds d numbers to the model and buys only the small gain that any split of
  any blob would give.
* **You lose the summary.** "These n points are really k kinds" gets longer without adding a new kind.

Each of these is a claim; [§3](#3-how-do-you-show-that-first-are-true-labels-available) shows how to measure it.

## 3. How do you show that? First: are true labels available?

**Branch 1: true labels exist** (rare: simulated data, a benchmark, testing a method).

* Score each k's labels **against the truth** with the
  [ARI](../05-in-practice/README.md#the-adjusted-rand-index-ari-step-by-step): 0.50, 0.71, **1.00** at k = 4, then
  0.91, 0.82, 0.72 as real groups get cut.
* Caveat: with labels you already know the true k. This branch checks whether a k-choosing *rule* works, or compares
  methods; it is not how you discover k. The labs use it this way.

**Branch 2: no labels** (the usual case). Every check must work without them:

1. **Are the clusters tight and separated? The silhouette.** Needs only the data and one labeling. Peaks at k = 4
   (0.77) and falls after (0.66, 0.56, 0.45): points near an arbitrary cut are almost as close to the other half as
   to their own. On large data, see [§4](#4-the-silhouette-needs-every-pairwise-distance-what-about-large-data).
2. **Do they reproduce? Stability (ARI run vs run).** Refit on random subsamples and compare the runs' labelings
   ([§5](#5-what-does-ari-run-vs-run-mean-is-it-comparing-different-k)). 1.00 at k = 4; 0.84–0.87 after.
   * It doesn't fall further because only the blob being cut changes; the other three groups agree in every run.
   * Too few clusters is unstable too (0.80 at k = 3): which two blobs get merged is arbitrary.
3. **Is it more than noise? The gap idea.** Compare the inertia drop with the drop on **uniform data in the same box**
   (no clusters at all). At k = 4 the real drop is **1.8×** the uniform one; after, only **0.1–0.2×**: the real blobs
   are tight, so cutting them gains even less than cutting structureless noise. This is the idea behind the gap
   statistic (Tibshirani, Walther & Hastie, 2001).
4. **Is it useful? The domain.**
   * *Profiles:* do the two clusters from a split differ in any feature other than along the cut?
   * *An outside variable* the clustering never saw (churn, spend, diagnosis): do the two halves behave differently?
   * *Names:* can a domain expert name them differently? If not, they are one segment.

## 4. The silhouette needs every pairwise distance. What about large data?

* **The problem:** time and memory grow as n² (at 1,000,000 points, 8 TB of distances). scikit-learn computes them
  in chunks, which bounds the memory but not the time ([cost, README §6](README.md#6-choosing-k)).
* **Subsample:** `silhouette_score(X, labels, sample_size=10_000, random_state=...)`. 10,000 points is 10⁸ distances,
  seconds. Repeat on a few samples; if the score barely moves, trust it.
* **Simplified silhouette:** measure against **centroids** instead of every point: a = distance to its own centroid,
  b = distance to the nearest other centroid. That costs n · k, and k-means has already computed those distances.
* **Other label-free scores built on centroids:** Calinski–Harabasz and Davies–Bouldin, about n · k · d each
  ([judging a clustering](../05-in-practice/README.md#3-judging-a-clustering)).
* **Stability scales too:** the ARI needs only the contingency table, one pass over the labels. The cost is the extra
  k-means fits, not n².

## 5. What does "ARI run vs run" mean? Is it comparing different k?

* **No: it is the same k every time.** For k = 5:
  1. Draw random subsamples (here 80% of the points each).
  2. Fit k-means with k = 5 on each.
  3. Use every fit to label **the same full set of points**.
  4. Compute the ARI between every pair of these labelings and average: the stability at k = 5.
     * Where one number comes from: 10 runs give 10 × 9 / 2 = 45 pairs, so 45 ARIs; their mean is the stability. At
       k = 5 they range 0.82–1.00, mean **0.87**: every run must cut one blob in two, pairs that cut the same blob
       agree at about 0.96, pairs that cut different blobs at about 0.82. At k = 4 nothing is arbitrary, so all 45
       are 1.00.
* **The question it answers:** "if I redo this clustering on slightly different data, do I get the same groups?" Real
  groups come back (ARI near 1); arbitrary cuts move (ARI drops).
* **It needs no truth:** both labelings come from the method itself.
* **Comparing k with k + 1 is a different, also useful tool.** Their labelings always differ (k differs), so their
  ARI alone can't say whether k + 1 is real. But their contingency table shows **which** cluster split. Plotting that
  across k is the idea of clustering trees (Zappia & Oshlack, 2018).
  * Together: k vs k + 1 shows *what* split; stability at k + 1 shows whether that split *reproduces*.

## 6. Isn't checking the elbow with the silhouette circular? Why not just use the silhouette?

* **It isn't a proof of one by the other.** Neither is ground truth: every label-free score is a rule of thumb that
  answers a different question and **fails in different situations**:

  | method | question it answers | where it fails |
  |---|---|---|
  | elbow | where does adding a cluster stop buying tightness? | often no clear bend; still bends on pure noise (Lab 3) |
  | silhouette | how well separated are the clusters? | favours round, well-separated clusters: it prefers k-means' wrong split of the moons over DBSCAN's right one (0.49 vs 0.05, [Lab 11](../05-in-practice/README.md#3-judging-a-clustering)); undefined at k = 1, so it can never say "no clusters"; on uniform noise it still picks a k (0.41, Lab 3) |
  | stability | does the same clustering come back? | a wrong answer can be stable too (cutting the moons the same wrong way every time) |
  | gap statistic | better than data with no structure? | depends on choosing a sensible "no structure" reference; but it is the only one here that **can** answer k = 1 |

* **Agreement is the evidence.** Methods that fail differently are unlikely to be wrong in the same place. Here, the
  elbow, silhouette, stability and gap all say 4.
* **Disagreement is information too:**
  * Nested groups: on the two pairs of close blobs (Lab 3) both rules pick 2, the coarsest level. An elbow at 4 with a
    silhouette at 2 would point to groups within groups; look at both levels (e.g. a dendrogram).
  * Non-round shapes: a good silhouette with unstable or meaningless clusters suggests k-means is the wrong tool; try
    [DBSCAN](../03-dbscan/README.md) or [HDBSCAN](../04-optics-hdbscan/README.md).
* **Why do the elbow at all:** the inertia is free (it falls out of the fit; see
  [what each rule costs](kmeans_from_scratch.ipynb#lab-3-cost)). Use it to **shortlist** k cheaply, then check the
  shortlist with criteria that fail differently.
* **Using the silhouette alone is fine** on a few thousand points with roughly round clusters. The strong answer is
  knowing when it is not enough: other shapes, nested structure, or "is there any structure at all?".

## 7. Answers to give

**"Why not go past the elbow, if the inertia keeps falling?"**

> "The inertia always falls with k, down to 0 at k = n, so its value alone can't choose k. Before the true k, a new
> centroid separates real groups and the inertia drops a lot; after it, a centroid can only cut a real group in half,
> and the drop collapses (here from about 4,600 to about 115). Past that point the extra clusters are arbitrary cuts,
> not new groups. The exception is compression, where k is a budget."

**"How would you show those extra clusters are wrong?"**

> "First question: do I have labels? If yes, I score each k against them with the ARI, though then I'm testing the
> rule, not discovering k. Usually I don't, so I use label-free evidence: the silhouette (sampled, or the
> centroid-based version on large data), stability (refit on resamples, compare the labelings with the ARI; real
> clusters reproduce, arbitrary splits don't), and a gap test against structureless data. Here all three point to
> k = 4. Then the domain: do the clusters differ on something that matters?"

**"Aren't you just validating one heuristic with another?"**

> "I'm not validating the elbow with the silhouette; neither is ground truth. Each is a heuristic with known failure
> modes: the elbow is often ambiguous, and the silhouette favours round clusters and can't say 'no clusters'. So I use
> the free one to shortlist k, then check it with criteria that fail differently. If they agree, I'm confident; if
> they disagree, that tells me something, like nested groups or non-round shapes. In the end the domain decides
> whether the clusters mean anything."

## Reproduce the numbers

<details>
<summary><b>The code behind the evidence table</b> (scikit-learn; about a minute)</summary>

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

# library Lab 3's "4 clear blobs": 400 points, 2 features, true labels y
X, y = make_blobs(400, centers=[[-5, -5], [5, -5], [-5, 5], [5, 5]], cluster_std=1.2, random_state=1)
rng = np.random.default_rng(0)


def inertia(Z, k):
    """Inertia of the best of 10 k-means++ starts with k clusters."""
    return KMeans(k, n_init=10, random_state=0).fit(Z).inertia_


def stability(X, k, runs=10):
    """Mean ARI over all pairs of runs; each run fits k-means on a random 80% of the points and labels every point."""
    labelings = []
    for seed in range(runs):
        subsample = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        labelings.append(KMeans(k, n_init=10, random_state=seed).fit(X[subsample]).predict(X))
    return np.mean([adjusted_rand_score(labelings[i], labelings[j])
                    for i in range(runs) for j in range(i + 1, runs)])


def reference_drop(X, k, draws=5):
    """Mean inertia drop from k-1 to k on uniform data in the same box: what a new centroid buys when there are no clusters."""
    low, high = X.min(axis=0), X.max(axis=0)
    drops = []
    for _ in range(draws):
        U = rng.uniform(low, high, size=X.shape)
        drops.append(inertia(U, k - 1) - inertia(U, k))
    return np.mean(drops)


print(f"{'k':>2} {'inertia':>8} {'drop':>7} {'drop on uniform':>16} {'ratio':>6} {'silhouette':>11} {'stability':>10} {'ARI vs truth':>13}")
for k in range(1, 8):
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    row = f"{k:>2} {inertia(X, k):>8.0f}"
    if k > 1:
        drop, ref = inertia(X, k - 1) - inertia(X, k), reference_drop(X, k)
        row += (f" {drop:>7.0f} {ref:>16.0f} {drop / ref:>6.1f} {silhouette_score(X, labels):>11.2f}"
                f" {stability(X, k):>10.2f} {adjusted_rand_score(y, labels):>13.2f}")
    print(row)
```

Output:

```
 k  inertia    drop  drop on uniform  ratio  silhouette  stability  ARI vs truth
 1    20725
 2    10691   10034             8229    1.2        0.50       1.00          0.50
 3     5780    4911             4234    1.2        0.59       0.80          0.71
 4     1136    4644             2529    1.8        0.77       1.00          1.00
 5     1019     117              865    0.1        0.66       0.87          0.91
 6      905     114              717    0.2        0.56       0.84          0.82
 7      803     102              554    0.2        0.45       0.86          0.72
```

</details>

## References

* Tibshirani, Walther & Hastie (2001),
  [*Estimating the number of clusters in a data set via the gap statistic*](https://doi.org/10.1111/1467-9868.00293),
  JRSS B.
* Tibshirani & Walther (2005), [*Cluster validation by prediction strength*](https://doi.org/10.1198/106186005X59243),
  JCGS (a stability-style check: do a training half's centroids predict which test points end up together?).
* Zappia & Oshlack (2018),
  [*Clustering trees: a visualization for evaluating clusterings at multiple resolutions*](https://doi.org/10.1093/gigascience/giy083),
  GigaScience.
