"""
kmeans_production.py: monitoring and retraining a deployed k-means model.

Companion to kmeans_in_production.ipynb (same folder), which explains every rule used here and runs the scenarios.
Depends on numpy, scipy and scikit-learn only. Copy the file next to your own code and import from it.

Map to the notebook:
  part 1 (monitor)  ModelVersion, fit_model_version, holdout_share_for, compute_signals, build_bands, breaches,
                    AlarmRule, alarm_rule_table, data_checks
  part 2 (decide)   classify_trigger, previous_model_fits, decide_on_alarm, rebaseline
  part 3 (retrain)  backtest_window, k_scan, new_k_is_real, contingency, classify_contingency,
                    new_cluster_far_share, same_k_decision, fit_checks, hidden_shapes, hollow_clusters,
                    method_review, carry_names, names_from_cards, names_from_contingency, ab_test,
                    per_cluster_stability, merge_plan, pair_merge_costs, retrain_path
  part 4 (profile)  profile_clusters, format_profiles
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------------------------- small helpers
def format_table(rows, headers, fmt="{:.3f}"):
    """Rows and headers to a Markdown table string (floats formatted with fmt, percentages left to the caller)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return fmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def ks_d(sample, ref_sorted):
    """Exact two-sample Kolmogorov-Smirnov D between `sample` and a pre-sorted reference (ties handled)."""
    s = np.sort(np.asarray(sample, float))
    n, m = len(s), len(ref_sorted)
    f1_at = np.searchsorted(s, s, side="right") / n          # F_sample at each sample value
    f1_before = np.searchsorted(s, s, side="left") / n       # F_sample just before it
    f2_at = np.searchsorted(ref_sorted, s, side="right") / m
    f2_before = np.searchsorted(ref_sorted, s, side="left") / m
    return float(max(np.max(np.abs(f1_at - f2_at)), np.max(np.abs(f1_before - f2_before))))


def psi(p_now, p_ref, eps=1e-4):
    """Population stability index between two share vectors; zeros replaced by eps so the log is defined."""
    p_now = np.maximum(np.asarray(p_now, float), eps)
    p_ref = np.maximum(np.asarray(p_ref, float), eps)
    return float(np.sum((p_now - p_ref) * np.log(p_now / p_ref)))


def shares(labels, k):
    return np.bincount(labels, minlength=k) / len(labels)


def knee_by_chord(ks, values):
    """The k whose point lies farthest from the chord joining the curve's ends, both axes rescaled to 0-1."""
    x = (np.asarray(ks, float) - ks[0]) / (ks[-1] - ks[0])
    v = np.asarray(values, float)
    y = (v - v.min()) / (v.max() - v.min() + 1e-12)
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[int(np.argmax(d))])


def _d2_to(Z, C):
    """Squared distance of each row of Z to its nearest centroid in C, and that centroid's index."""
    D = cdist(Z, C, "sqeuclidean")
    lab = D.argmin(1)
    return D[np.arange(len(Z)), lab], lab


# ----------------------------------------------------------------------------------------------- part 1: the model version
@dataclass
class Band:
    lower: Optional[float]   # None for the one-sided signals (1 and 3)
    median: float
    upper: float

    def check(self, value):
        """'above', 'below' or None."""
        if value > self.upper:
            return "above"
        if self.lower is not None and value < self.lower:
            return "below"
        return None


@dataclass
class ModelVersion:
    """Everything stored at go-live: the model, its training window, the held-out rows, the baselines and the bands."""
    version: int
    feature_names: list
    scaler: StandardScaler
    kmeans: KMeans
    batch_size: int
    train_window: tuple
    train_sample: np.ndarray      # raw training rows (a sample), for signal 1
    train_labels: np.ndarray      # their segments, for signal 1 per segment
    train_shares: np.ndarray      # signal 3's baseline
    cutoff: float                 # signal 4's cut-off: 99th percentile of training distances (scaled space)
    holdout: np.ndarray           # raw held-out rows, never fitted; the bands come from them
    bands: dict = field(default_factory=dict)
    k_evidence: dict = field(default_factory=dict)
    names: dict = field(default_factory=dict)
    previous: Optional["ModelVersion"] = None
    _ref_sorted: dict = field(default_factory=dict, repr=False)

    @property
    def k(self):
        return self.kmeans.n_clusters

    def scaled(self, X):
        return self.scaler.transform(X)

    def predict(self, X):
        return self.kmeans.predict(self.scaled(X))

    def d2(self, X):
        """Squared distance of each raw row to its assigned centroid (scaled space), and the labels."""
        return _d2_to(self.scaled(X), self.kmeans.cluster_centers_)

    def centroids_raw(self):
        return self.scaler.inverse_transform(self.kmeans.cluster_centers_)

    def centroids_in(self, other: "ModelVersion"):
        """This version's centroids expressed in another version's scaled space."""
        return other.scaler.transform(self.centroids_raw())


def _ref_sorted(mv: ModelVersion):
    """Sorted training values per feature, overall and per segment (cached; used by every KS call)."""
    if not mv._ref_sorted:
        for j in range(len(mv.feature_names)):
            mv._ref_sorted[("all", j)] = np.sort(mv.train_sample[:, j])
            for g in range(mv.k):
                mv._ref_sorted[(g, j)] = np.sort(mv.train_sample[mv.train_labels == g, j])
    return mv._ref_sorted


def holdout_share_for(n_rows, batch_size, floor=0.2, cap=0.5, batches_needed=10):
    """Hold out what the bands need (~10 batches' worth), between 20% and 50% of the rows."""
    return float(min(cap, max(floor, batches_needed * batch_size / n_rows)))


def fit_model_version(X_raw, k, feature_names, batch_size, *, version=1, train_window=(None, None),
                      holdout_share=None, train_sample_size=5000, n_draws=4000, names=None, previous=None,
                      k_evidence=None, rng=None):
    """Hold out a random share, fit scaler + k-means on the rest, store the baselines and build the bands."""
    rng = np.random.default_rng(rng)
    if holdout_share is None:
        holdout_share = holdout_share_for(len(X_raw), batch_size)
    idx = rng.permutation(len(X_raw))
    n_hold = int(round(holdout_share * len(X_raw)))
    hold, fit = X_raw[idx[:n_hold]], X_raw[idx[n_hold:]]
    scaler = StandardScaler().fit(fit)
    Z = scaler.transform(fit)
    km = KMeans(k, n_init=10, random_state=int(rng.integers(1 << 31))).fit(Z)
    d2 = ((Z - km.cluster_centers_[km.labels_]) ** 2).sum(1)
    samp = rng.choice(len(fit), min(train_sample_size, len(fit)), replace=False)
    mv = ModelVersion(version, list(feature_names), scaler, km, batch_size, tuple(train_window),
                      fit[samp], km.labels_[samp], shares(km.labels_, k), float(np.percentile(d2, 99)), hold,
                      names=names or {j: f"S{j}" for j in range(k)}, previous=previous, k_evidence=k_evidence or {})
    mv.bands = build_bands(mv, n_draws=n_draws, rng=rng)
    return mv


# ----------------------------------------------------------------------------------------------- part 1: signals and bands
def _reweighted_mean(values, lab, train_shares):
    """Mean of per-point values, each segment weighted by its training share instead of its batch share.

    A mix shift (signal 3's job) changes how many points each segment holds; without this, a shift toward a tight
    or a wide segment would move signals 2 and 4 too. Segments absent from the batch drop out of the weights.
    """
    k = len(train_shares)
    present = np.array([(lab == g).any() for g in range(k)])
    w = np.where(present, train_shares, 0.0)
    w = w / w.sum()
    return float(sum(w[g] * values[lab == g].mean() for g in range(k) if present[g]))


def compute_signals(mv: ModelVersion, X_raw, min_segment_rows=30):
    """Signals 1-4 of one batch under one model version. Keys: 's1:<feature>', 's1seg', 's2', 's3', 's4'.

    Signals 2 and 4 are per-segment means reweighted to the training shares (see _reweighted_mean).
    """
    ref = _ref_sorted(mv)
    d2, lab = mv.d2(X_raw)
    out = {}
    for j, f in enumerate(mv.feature_names):
        out[f"s1:{f}"] = ks_d(X_raw[:, j], ref[("all", j)])
    seg = {}
    for g in range(mv.k):
        rows = X_raw[lab == g]
        if len(rows) >= min_segment_rows and len(ref[(g, 0)]) >= min_segment_rows:
            for j, f in enumerate(mv.feature_names):
                seg[(g, f)] = ks_d(rows[:, j], ref[(g, j)])
    out["s1seg"] = max(seg.values()) if seg else 0.0        # the max statistic over segments x features
    out["s1seg_detail"] = seg
    out["s2"] = _reweighted_mean(d2, lab, mv.train_shares)
    out["s3"] = psi(shares(lab, mv.k), mv.train_shares)
    out["s4"] = _reweighted_mean((d2 > mv.cutoff).astype(float), lab, mv.train_shares)
    out["shares"] = shares(lab, mv.k)
    return out


def build_bands(mv: ModelVersion, n_draws=4000, rng=None, min_segment_rows=30):
    """Draw batches of batch_size from the held-out rows, score each, take the edges per signal."""
    rng = np.random.default_rng(rng)
    H, n = mv.holdout, mv.batch_size
    replace = len(H) < 10 * n                      # small held-out set: bootstrap
    d2h, labh = mv.d2(H)
    ref = _ref_sorted(mv)
    vals = {key: [] for key in [f"s1:{f}" for f in mv.feature_names] + ["s1seg", "s2", "s3", "s4"]}
    for _ in range(n_draws):
        i = rng.choice(len(H), n, replace=replace)
        X, d2, lab = H[i], d2h[i], labh[i]
        for j, f in enumerate(mv.feature_names):
            vals[f"s1:{f}"].append(ks_d(X[:, j], ref[("all", j)]))
        segmax = 0.0
        for g in range(mv.k):
            m = lab == g
            if m.sum() >= min_segment_rows and len(ref[(g, 0)]) >= min_segment_rows:
                for j in range(X.shape[1]):
                    segmax = max(segmax, ks_d(X[m, j], ref[(g, j)]))
        vals["s1seg"].append(segmax)
        vals["s2"].append(_reweighted_mean(d2, lab, mv.train_shares))
        vals["s3"].append(psi(shares(lab, mv.k), mv.train_shares))
        vals["s4"].append(_reweighted_mean((d2 > mv.cutoff).astype(float), lab, mv.train_shares))
    bands = {}
    for key, v in vals.items():
        v = np.asarray(v)
        if key in ("s2", "s4"):   # two-sided: 0.25th and 99.75th percentiles
            bands[key] = Band(float(np.percentile(v, 0.25)), float(np.median(v)), float(np.percentile(v, 99.75)))
        else:                     # one-sided: 99.5th percentile
            bands[key] = Band(None, float(np.median(v)), float(np.percentile(v, 99.5)))
    return bands


def breaches(signals, bands):
    """{key: 'above' | 'below'} for the signals outside their band."""
    out = {}
    for key, band in bands.items():
        d = band.check(signals[key])
        if d:
            out[key] = d
    return out


@dataclass
class Alarm:
    key: str
    direction: str
    t_alarm: object
    t_alarm_window_earliest: object
    breached_batches: list


class AlarmRule:
    """m of the last n batches outside the band, per signal, sliding window; an open alarm does not re-fire."""

    def __init__(self, m=2, n=5):
        self.m, self.n = m, n
        self.history: dict = {}
        self.open: dict = {}

    def update(self, batch_id, breach_map, keys):
        fired = []
        for key in keys:
            h = self.history.setdefault(key, deque(maxlen=self.n))
            d = breach_map.get(key)
            h.append((batch_id, d))
            if d is None or key in self.open:
                continue
            hits = [(b, dd) for b, dd in h if dd is not None]
            if len(hits) >= self.m:
                a = Alarm(key, d, batch_id, hits[0][0], [b for b, _ in hits])
                self.open[key] = a
                fired.append(a)
        return fired

    def close(self, key=None):
        if key is None:
            self.open.clear()
        else:
            self.open.pop(key, None)


def _arl_m_of_n(m, n, q):
    """Expected batches until 'm of the last n' fires when each batch breaches independently with probability q."""
    if m == 1:
        return 1.0 / q
    states = list(itertools.product([0, 1], repeat=n - 1))
    idx = {s: i for i, s in enumerate(states)}
    A = np.eye(len(states))
    for s in states:
        for o, pr in ((1, q), (0, 1 - q)):
            w = s + (o,)
            if sum(w) >= m:
                continue
            A[idx[s], idx[w[1:]]] -= pr
    return float(np.linalg.solve(A, np.ones(len(states)))[idx[tuple([0] * (n - 1))]])


def alarm_rule_table(p_healthy=0.005, tests=1, rules=((1, 1), (2, 3), (2, 5), (3, 5)),
                     q_values=(0.1, 0.33, 0.5, 1.0), batches_per_year=52):
    """Per rule: false alarms a year across `tests` signals, and batches to detect a change that breaches with rate q."""
    rows = []
    for m, n in rules:
        fa = tests * batches_per_year / _arl_m_of_n(m, n, p_healthy)
        rows.append((f"{m} of {n}", fa, *[_arl_m_of_n(m, n, q) for q in q_values]))
    headers = ["rule", "false alarms / year"] + [f"batches to detect, q = {q}" for q in q_values]
    return rows, headers


def data_checks(X_raw, mv: ModelVersion, max_out_of_range=0.01, pile_share=0.03):
    """Cheap input checks on the column(s) signal 1 points at: missing, constant, piled on one value, out of range.

    'Piled': one exact value holds > pile_share of the batch while no value held a third of that in training
    (clipping, capping or a default value); a feature that was already discrete or clipped in training passes.
    """
    issues = []
    T = mv.train_sample
    for j, f in enumerate(mv.feature_names):
        x = X_raw[:, j]
        if np.isnan(x).any():
            issues.append(f"{f}: {np.isnan(x).mean():.1%} missing")
            x = x[~np.isnan(x)]
        if np.std(x) < 1e-12:
            issues.append(f"{f}: constant ({x[0]:.3g}) for every row")
            continue
        vals, counts = np.unique(x, return_counts=True)
        top = counts.max() / len(x)
        tvals, tcounts = np.unique(T[:, j], return_counts=True)
        if top > pile_share and tcounts.max() / len(T) < pile_share / 3:
            issues.append(f"{f}: {top:.0%} of rows piled on the single value {vals[counts.argmax()]:.3g}")
        lo, hi = T[:, j].min(), T[:, j].max()
        span = hi - lo
        out = ((x < lo - 0.5 * span) | (x > hi + 0.5 * span)).mean()
        if out > max_out_of_range:
            issues.append(f"{f}: {out:.1%} of rows far outside the training range")
    return issues


# ----------------------------------------------------------------------------------------------- part 2: decide
def classify_trigger(breach_map):
    """Which row of the trigger table applies, in the table's order of precedence."""
    up = {k for k, d in breach_map.items() if d == "above"}
    down = {k for k, d in breach_map.items() if d == "below"}
    if "s4" in up:
        return "new_group"
    if "s2" in up:
        return "fit_worse"
    if "s2" in down or "s4" in down:
        return "tighter"
    if "s1seg" in up:
        return "segment_shape"
    if "s3" in up:
        return "mix_shift"
    if any(k.startswith("s1:") for k in up):
        return "feature_drift_only"
    return "none"


def previous_model_fits(mv: ModelVersion, X_batch):
    """Rollback check: does the previous version sit inside all of its own bands on this batch?"""
    if mv.previous is None or mv.previous.feature_names != mv.feature_names:
        return False, {}
    b = breaches(compute_signals(mv.previous, X_batch), mv.previous.bands)
    return len(b) == 0, b


def decide_on_alarm(mv: ModelVersion, X_latest, breach_map, min_share=0.01):
    """The decision step, in the order the notebook gives: data checks, rollback, then the trigger table.

    Returns {'action': 'fix the data' | 'roll back' | 'keep + rebaseline' | 'watch' | 'log only' | 'retrain', ...}.
    Per-segment signal 1 is a watch signal like signal 3: it names a segment and a feature; signals 2 and 4 decide
    retraining.
    """
    out = {"trigger": classify_trigger(breach_map), "issues": data_checks(X_latest, mv)}
    if out["issues"]:
        out["action"] = "fix the data"
        out["why"] = "an input check failed on the column(s) signal 1 points at; a refit would learn the bug"
        return out
    fits, prev_b = previous_model_fits(mv, X_latest)
    out["previous_fits"] = fits
    if fits:
        out["action"] = "roll back"
        out["why"] = f"version {mv.previous.version} sits inside all of its own bands on this batch"
        return out
    t = out["trigger"]
    if t == "mix_shift":
        sh = compute_signals(mv, X_latest)["shares"]
        empty = [mv.names.get(g, g) for g in range(mv.k) if sh[g] < min_share]
        out["action"] = "keep + rebaseline"
        out["why"] = "customers moved between segments; the segments are still right, only their sizes changed"
        if empty:
            out["why"] += f"; segments now empty: {empty} (run the k-scan: a k - 1 candidate)"
        return out
    if t == "feature_drift_only":
        out["action"] = "log only"
        out["why"] = "a feature moved but the fit did not (signals 2 and 4 inside their bands); watch it"
        return out
    if t == "segment_shape" or (t == "tighter" and "s1seg" in breach_map):
        out["action"] = "watch"
        out["why"] = ("a feature's distribution changed inside one segment while the fit held (signals 2 and 4 inside "
                      "their bands): profile the segment, keep watching, re-draw its reference rows once the change is "
                      "accepted; signals 2 and 4 decide any refit")
        return out
    if t == "tighter":
        out["action"] = "log only"
        out["why"] = "points sit tighter than usual, inputs pass the checks, no segment changed shape"
        return out
    out["action"] = "retrain"
    out["why"] = {"new_group": "far points above the band: a group the model has no centroid for",
                  "fit_worse": "the fit got worse: centroids moved or a group appeared"}[t]
    return out


def rebaseline(mv: ModelVersion, X_window, rng=None, holdout_share=None, train_sample_size=5000, n_draws=4000):
    """Same scaler, centroids and cut-off; new held-out rows, segment shares and bands from an accepted new window."""
    rng = np.random.default_rng(rng)
    if holdout_share is None:
        holdout_share = holdout_share_for(len(X_window), mv.batch_size)
    idx = rng.permutation(len(X_window))
    n_hold = int(round(holdout_share * len(X_window)))
    hold, rest = X_window[idx[:n_hold]], X_window[idx[n_hold:]]
    lab = mv.predict(rest)
    samp = rng.choice(len(rest), min(train_sample_size, len(rest)), replace=False)
    new = ModelVersion(mv.version, mv.feature_names, mv.scaler, mv.kmeans, mv.batch_size, mv.train_window,
                       rest[samp], lab[samp], shares(lab, mv.k), mv.cutoff, hold, names=dict(mv.names),
                       k_evidence=mv.k_evidence, previous=mv)
    new.bands = build_bands(new, n_draws=n_draws, rng=rng)
    return new


def names_from_cards(mv: ModelVersion, cards):
    """Name each cluster after the nearest profile card ({name: feature values in original units}, scaled space)."""
    names = list(cards)
    D = cdist(mv.kmeans.cluster_centers_, mv.scaled(np.array([cards[n] for n in names], float)))
    return {c: names[int(D[c].argmin())] for c in range(mv.k)}


def names_from_contingency(M, old_names):
    """Name each new cluster after the old segment most of its members came from; shared parents get ' · a/b'."""
    M = np.asarray(M)
    parent = M.argmax(0)
    counts = Counter(parent.tolist())
    names, seen = {}, {}
    for c in range(M.shape[1]):
        j = int(parent[c])
        if counts[j] > 1:
            seen[j] = seen.get(j, 0) + 1
            names[c] = f"{old_names[j]} · {chr(96 + seen[j])}"
        else:
            names[c] = old_names[j]
    return names


# ----------------------------------------------------------------------------------------------- part 3: retrain
def backtest_window(batches, mv: ModelVersion, t_alarm_window_earliest, t_alarm,
                    extra=((13, "last 13 batches"), (26, "last 26 batches")), min_rows_per_cluster=39,
                    prefer_recent_tol=0.01):
    """Step 1: fit k-means (current k) on each candidate window, score on the test batch, keep the best.

    `batches` is the ordered list of (batch_id, X_raw) kept since t_start; the test batch is the one at t_alarm.
    One shared scaler (fit on the window since t_alarm_window_earliest) keeps the scores comparable. A window that
    reaches back before t_alarm_window_earliest must beat the post-change window by more than prefer_recent_tol
    (relative): an equal fit means the older rows add nothing, and they come from before the change.
    """
    ids = [b for b, _ in batches]
    pos = {b: i for i, b in enumerate(ids)}
    i_test, i_awe = pos[t_alarm], pos[t_alarm_window_earliest]
    test = batches[i_test][1]
    cands = {"since t_alarm_window_earliest": range(i_awe, i_test)}
    if mv.train_window[0] in pos and 0 < pos[mv.train_window[0]] < i_awe:
        # the current model was itself fitted after a change: everything since that change is a candidate too
        cands["since the current model's training window began"] = range(pos[mv.train_window[0]], i_test)
    for n_b, name in extra:
        cands[name] = range(max(0, i_test - n_b), i_test)
    cands["everything since t_start"] = range(0, i_test)
    stack = lambda r: np.vstack([batches[i][1] for i in r])
    shared = StandardScaler().fit(stack(cands["since t_alarm_window_earliest"]))
    Zt = shared.transform(test)
    rows, best, recent_score = [], None, None
    for name, r in cands.items():
        X = stack(r)
        if len(X) < min_rows_per_cluster * mv.k:
            rows.append((name, len(r), len(X), "too few rows"))
            continue
        km = KMeans(mv.k, n_init=5, random_state=0).fit(shared.transform(X))
        score = float(_d2_to(Zt, km.cluster_centers_)[0].mean())
        rows.append((name, len(r), len(X), score))
        if recent_score is None:
            recent_score = score                       # the first candidate is the post-change window
        bar = best[1] if best is not None else None
        if best is None or score < min(bar, recent_score * (1 - prefer_recent_tol)):
            best = (name, score, X, r)
    return {"table": rows, "headers": ["candidate window", "batches", "rows", "mean sq. distance on the test batch"],
            "chosen": best[0], "X": best[2], "batch_ids": [ids[i] for i in best[3]], "test": test}


def k_scan(Z, k_range=range(2, 9), n_subsamples=30, frac=0.8, n_refs=5, sample=4000, stability_floor=0.8,
           sil_tol=0.01, rng=None):
    """Step 2: elbow, silhouette, gap and stability per k; the k most of elbow/silhouette/gap agree on.

    The silhouette's pick is the smallest k within sil_tol of the best silhouette (the one-standard-error idea: a
    0.004 gap between two k is noise, so prefer the simpler model); ties between the three go to the silhouette.
    """
    rng = np.random.default_rng(rng)
    Z = Z[rng.choice(len(Z), min(sample, len(Z)), replace=False)]
    ks = list(k_range)
    inertia, sil, stab, gap, gap_s = [], [], [], [], []
    lo, hi = Z.min(0), Z.max(0)
    refs = [rng.uniform(lo, hi, Z.shape) for _ in range(n_refs)]
    for k in ks:
        km = KMeans(k, n_init=5, random_state=0).fit(Z)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z, km.labels_, sample_size=min(2000, len(Z)), random_state=0))
        aris = []
        for _ in range(n_subsamples):
            i = rng.choice(len(Z), int(frac * len(Z)), replace=False)
            sub = KMeans(k, n_init=3, random_state=int(rng.integers(1 << 31))).fit(Z[i])
            aris.append(adjusted_rand_score(km.labels_, sub.predict(Z)))
        stab.append(float(np.mean(aris)))
        log_ref = [np.log(KMeans(k, n_init=3, random_state=0).fit(R).inertia_) for R in refs]
        gap.append(float(np.mean(log_ref) - np.log(km.inertia_)))
        gap_s.append(float(np.std(log_ref) * math.sqrt(1 + 1 / n_refs)))
    picks = {"elbow": knee_by_chord(ks, inertia),
             "silhouette": min(k for k, v in zip(ks, sil) if v >= max(sil) - sil_tol)}
    g = [k for i, k in enumerate(ks[:-1]) if gap[i] >= gap[i + 1] - gap_s[i + 1]]
    picks["gap"] = g[0] if g else ks[-1]
    votes = Counter(picks.values())
    top = max(votes.values())
    tied = [k for k, c in votes.items() if c == top]
    chosen = picks["silhouette"] if picks["silhouette"] in tied else min(tied)
    table = list(zip(ks, inertia, sil, stab, gap))
    return {"table": table, "headers": ["k", "inertia", "silhouette", "stability (ARI)", "gap"], "picks": picks,
            "k": int(chosen), "stable": stab[ks.index(chosen)] >= stability_floor}


def new_k_is_real(Z, k_old, k_new, n_subsamples=30, frac=0.8, win_rate=0.95, time_slices=None, sample=4000, rng=None):
    """Step 4, 'real?': the new k must beat the old k's silhouette on >= win_rate of subsamples (and most time slices)."""
    rng = np.random.default_rng(rng)
    Z = Z[rng.choice(len(Z), min(sample, len(Z)), replace=False)]

    def sil(X, k):
        lab = KMeans(k, n_init=3, random_state=int(rng.integers(1 << 31))).fit_predict(X)
        return silhouette_score(X, lab, sample_size=min(2000, len(X)), random_state=0)

    gains = []
    for _ in range(n_subsamples):
        Xi = Z[rng.choice(len(Z), int(frac * len(Z)), replace=False)]
        gains.append(sil(Xi, k_new) - sil(Xi, k_old))
    gains = np.asarray(gains)
    wins = float((gains > 0).mean())
    tw = None
    if time_slices:
        tw = float(np.mean([sil(S, k_new) > sil(S, k_old) for S in time_slices]))
    real = wins >= win_rate and (tw is None or tw >= 0.5)
    return {"wins": wins, "gain_median": float(np.median(gains)), "gain_p5": float(np.percentile(gains, 5)),
            "gain_p95": float(np.percentile(gains, 95)), "time_window_wins": tw, "real": bool(real)}


def contingency(old_labels, new_labels):
    """Rows: old segments, columns: new clusters, cells: rows in both."""
    k_old, k_new = old_labels.max() + 1, new_labels.max() + 1
    M = np.zeros((k_old, k_new), int)
    np.add.at(M, (old_labels, new_labels), 1)
    return M


def new_cluster_far_share(mv_old: ModelVersion, X_raw, new_labels):
    """Per new cluster: the share of its members that were far points (beyond the cut-off) under the old model."""
    d2, _ = mv_old.d2(X_raw)
    far = d2 > mv_old.cutoff
    return np.array([far[new_labels == c].mean() if (new_labels == c).any() else 0.0 for c in range(new_labels.max() + 1)])


def classify_contingency(M, purity=0.9, far_share=None, far_floor=0.5):
    """'same partition', 'clean split', 'clean merge', 'new cluster(s) + ...' or 'reorganisation'."""
    M = np.asarray(M, float)
    new_cols = [c for c in range(M.shape[1]) if far_share is not None and far_share[c] >= far_floor]
    keep = [c for c in range(M.shape[1]) if c not in new_cols]
    R = M[:, keep] if keep else M
    col_pure = (R.max(0) / np.maximum(R.sum(0), 1) >= purity).all() if keep else True
    row_pure = (R.max(1) / np.maximum(R.sum(1), 1) >= purity).all() if keep else True
    if col_pure and row_pure:
        base = "same partition"
    elif col_pure:
        base = "clean split"
    elif row_pure:
        base = "clean merge"
    else:
        base = "reorganisation"
    if new_cols:
        return f"new cluster{'s' if len(new_cols) > 1 else ''} {new_cols} of mostly far points; the rest: {base}"
    return base


def same_k_decision(mv_old: ModelVersion, mv_new: ModelVersion, X_recent, X_test, n_refits=30, frac=0.8,
                    ari_floor=0.8, switch_eps=0.01, rng=None):
    """Step 5: shift vs noise, fit gain vs fit noise, switch share; all distances in the old model's scaled space."""
    rng = np.random.default_rng(rng)
    k = mv_old.k
    C_old = mv_old.kmeans.cluster_centers_
    C_new = mv_new.centroids_in(mv_old)
    D = cdist(C_old, C_new)
    r, c = linear_sum_assignment(D)
    perm = c                                         # old segment j <-> new centroid perm[j]
    shift = D[r, c]
    Zt = mv_old.scaled(X_test)
    d2_old, lab_old = _d2_to(Zt, C_old)
    d2_new, lab_new = _d2_to(Zt, C_new)
    gain = float(d2_old.mean() / d2_new.mean() - 1)
    inv = np.empty(k, int)
    inv[perm] = np.arange(k)
    switch = float((lab_old != inv[lab_new]).mean())
    dists, gains, preds = [], [], []
    for _ in range(n_refits):
        i = rng.choice(len(X_recent), int(frac * len(X_recent)), replace=False)
        sc = StandardScaler().fit(X_recent[i])
        km = KMeans(k, n_init=3, random_state=int(rng.integers(1 << 31))).fit(sc.transform(X_recent[i]))
        C_r = mv_old.scaler.transform(sc.inverse_transform(km.cluster_centers_))
        Dr = cdist(C_new, C_r)
        rr, cc = linear_sum_assignment(Dr)
        dists.append(Dr[rr, cc])
        d2_r, lab_r = _d2_to(Zt, C_r)
        gains.append(d2_old.mean() / d2_r.mean() - 1)
        preds.append(lab_r)
    noise_new_order = np.percentile(np.asarray(dists), 95, axis=0)
    noise = noise_new_order[perm]                   # back to the old segments' order
    fit_noise = float(np.percentile(np.abs(np.asarray(gains) - gain), 95))
    pairs = list(itertools.combinations(range(n_refits), 2))
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(30, len(pairs)), replace=False)]
    refit_ari = float(np.mean([adjusted_rand_score(preds[a], preds[b]) for a, b in pairs]))
    if refit_ari < ari_floor:
        decision = "investigate: refits disagree among themselves"
    elif np.all(shift <= noise):
        decision = "keep the old model: no centroid moved beyond its noise"
    elif gain <= fit_noise:
        decision = "keep the old model: fit gain within fit noise"
    elif switch < switch_eps:
        decision = "replace silently, under the same names"
    else:
        decision = "replace visibly: re-profile, tell the teams, watch the business metrics"
    table = [(mv_old.names.get(j, j), float(shift[j]), float(noise[j]), "yes" if shift[j] > noise[j] else "no")
             for j in range(k)]
    return {"table": table, "headers": ["old segment", "shift", "noise (95th pct of refits)", "moved?"],
            "fit_gain": gain, "fit_noise": fit_noise, "switch_share": switch, "refit_ari": refit_ari,
            "perm": perm, "decision": decision}


def hidden_shapes(Z, km_labels, sample=3000, min_cluster_size=None, stability_floor=0.7, spread_floor=0.8, rng=None):
    """Step 3, 'no hidden shapes': a stable HDBSCAN cluster spread over several k-means clusters."""
    rng = np.random.default_rng(rng)
    i = rng.choice(len(Z), min(sample, len(Z)), replace=False)
    Z, L = Z[i], km_labels[i]
    mcs = min_cluster_size or max(25, len(Z) // 50)
    hl = HDBSCAN(min_cluster_size=mcs, copy=True).fit(Z).labels_
    a = np.sort(rng.choice(len(Z), int(0.8 * len(Z)), replace=False))
    b = np.sort(rng.choice(len(Z), int(0.8 * len(Z)), replace=False))
    la = HDBSCAN(min_cluster_size=mcs, copy=True).fit(Z[a]).labels_
    lb = HDBSCAN(min_cluster_size=mcs, copy=True).fit(Z[b]).labels_
    common = np.intersect1d(a, b)
    pa, pb = {v: p for p, v in enumerate(a)}, {v: p for p, v in enumerate(b)}
    ari = adjusted_rand_score([la[pa[v]] for v in common], [lb[pb[v]] for v in common])
    clusters = []
    for cl in sorted(set(hl) - {-1}):
        m = hl == cl
        if m.mean() < 0.05:
            continue
        top = np.bincount(L[m]).max() / m.sum()
        clusters.append((int(cl), int(m.sum()), float(top)))
    hidden = ari >= stability_floor and any(top < spread_floor for _, _, top in clusters)
    return {"n_clusters": len(set(hl) - {-1}), "noise_share": float((hl == -1).mean()), "stability_ari": float(ari),
            "clusters": clusters, "headers": ["HDBSCAN cluster", "size", "share in its main k-means cluster"],
            "hidden": bool(hidden)}


def hollow_clusters(Z, labels, centroids, hollow_floor=1.3, k_nn=5, sample=500, rng=None):
    """Per cluster: how empty the space around the centroid is.

    Ratio = distance from the centroid to its k_nn nearest members / the members' median k_nn-nearest-neighbour
    distance. A round cluster has its centroid in its densest region (ratio below 1, about 0.5-0.9 on skewed
    blobs); a ring or crescent has it in empty space (1.5-2 for a half circle). k-means describes a cluster
    only by its centroid, so it can't fit a hollow one.
    """
    rng = np.random.default_rng(rng)
    rows = []
    for c in range(len(centroids)):
        m = np.where(labels == c)[0]
        if len(m) < 4 * k_nn:
            continue
        pts = Z[rng.choice(m, min(sample, len(m)), replace=False)]
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        typical = float(np.median(np.sort(D, axis=1)[:, k_nn - 1]))
        at_centroid = float(np.sort(np.linalg.norm(pts - centroids[c], axis=1))[k_nn - 1])
        rows.append((c, int(len(m)), at_centroid / typical))
    return {"clusters": rows, "headers": ["cluster", "size", f"{k_nn}-NN distance at the centroid / typical"],
            "hollow": [c for c, _, r in rows if r >= hollow_floor]}


def fit_checks(mv: ModelVersion, X_batch, rng=None):
    """Step 3: fit recovered, far points gone, no hidden shapes, no hollow clusters; any failure: review the method."""
    sig = compute_signals(mv, X_batch)
    checks = {
        "fit recovered (signal 2 inside its band)": mv.bands["s2"].check(sig["s2"]) is None,
        "far points gone (signal 4 inside its band)": mv.bands["s4"].check(sig["s4"]) is None,
    }
    Z, lab = mv.scaled(X_batch), mv.predict(X_batch)
    hs = hidden_shapes(Z, lab, rng=rng)
    checks["no hidden shapes (HDBSCAN)"] = not hs["hidden"]
    hs["hollow"] = hollow_clusters(Z, lab, mv.kmeans.cluster_centers_, rng=rng)
    checks["no hollow clusters (centroid among its members)"] = not hs["hollow"]["hollow"]
    return checks, sig, hs


def method_review(Z, k, sample=3000, n_pairs=5, min_cluster_size=None, rng=None):
    """Step 3's method review: clusters found, noise, stability on overlapping subsamples, agreement with k-means."""
    rng = np.random.default_rng(rng)
    Z = Z[rng.choice(len(Z), min(sample, len(Z)), replace=False)]
    mcs = min_cluster_size or max(25, len(Z) // 50)
    methods = {
        "k-means (k given)": lambda X: KMeans(k, n_init=5, random_state=0).fit_predict(X),
        "ward (k given)": lambda X: AgglomerativeClustering(k).fit_predict(X),
        "HDBSCAN": lambda X: HDBSCAN(min_cluster_size=mcs, copy=True).fit_predict(X),
    }
    full = {name: f(Z) for name, f in methods.items()}
    rows = []
    for name, f in methods.items():
        aris = []
        for _ in range(n_pairs):
            a = np.sort(rng.choice(len(Z), int(0.8 * len(Z)), replace=False))
            b = np.sort(rng.choice(len(Z), int(0.8 * len(Z)), replace=False))
            la, lb = f(Z[a]), f(Z[b])
            common = np.intersect1d(a, b)
            pa, pb = {v: p for p, v in enumerate(a)}, {v: p for p, v in enumerate(b)}
            aris.append(adjusted_rand_score([la[pa[v]] for v in common], [lb[pb[v]] for v in common]))
        lab = full[name]
        rows.append((name, int(len(set(lab) - {-1})), float((lab == -1).mean()), float(np.mean(aris)),
                     float(adjusted_rand_score(full["k-means (k given)"], lab))))
    return rows, ["method", "clusters", "noise share", "stability (ARI, overlapping subsamples)", "agreement with k-means (ARI)"]


def carry_names(mv_old: ModelVersion, mv_new: ModelVersion, far_share=None, far_floor=0.5):
    """Step 8: match new centroids to old ones (old model's space); split segments get 'name · a/b', merges 'A + B'.

    A new cluster made mostly of the old model's far points (far_share >= far_floor) is a new segment, not a split.
    """
    C_old = mv_old.kmeans.cluster_centers_
    C_new = mv_new.centroids_in(mv_old)
    D = cdist(C_old, C_new)
    k_old, k_new = mv_old.k, mv_new.k
    names, table = {}, []
    is_new = [far_share is not None and far_share[cc] >= far_floor for cc in range(k_new)]
    if k_new == k_old and not any(is_new):
        r, c = linear_sum_assignment(D)
        for j, cc in zip(r, c):
            names[int(cc)] = mv_old.names[j]
            table.append((int(cc), mv_old.names[j], float(D[j, cc])))
    elif k_new >= k_old:
        nearest = D.argmin(0)
        counts = Counter(int(nearest[cc]) for cc in range(k_new) if not is_new[cc])
        suffix = {}
        for cc in range(k_new):
            j = int(nearest[cc])
            if is_new[cc]:
                names[cc] = f"new segment {cc}"
            elif counts[j] > 1:
                suffix[j] = suffix.get(j, 0) + 1
                names[cc] = f"{mv_old.names[j]} · {chr(96 + suffix[j])}"
            else:
                names[cc] = mv_old.names[j]
            table.append((cc, names[cc], float(D[j, cc])))
    else:
        nearest = D.argmin(1)
        for cc in range(k_new):
            olds = [mv_old.names[j] for j in range(k_old) if nearest[j] == cc]
            names[cc] = " + ".join(olds) if olds else f"new {cc}"
            table.append((cc, names[cc], float(D[:, cc].min())))
    return names, table, ["new cluster", "name carried over", "distance to the matched old centroid"]


def ab_test(control, treatment):
    """Two-proportion z-test on 0/1 outcomes: rates, lift, 95% CI of the lift, p-value."""
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    n0, n1 = len(control), len(treatment)
    p0, p1 = control.mean(), treatment.mean()
    p = (control.sum() + treatment.sum()) / (n0 + n1)
    se = math.sqrt(p * (1 - p) * (1 / n0 + 1 / n1))
    z = (p1 - p0) / se if se > 0 else 0.0
    se_diff = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    return {"control": float(p0), "treatment": float(p1), "lift": float(p1 - p0),
            "ci": (float(p1 - p0 - 1.96 * se_diff), float(p1 - p0 + 1.96 * se_diff)),
            "p_value": float(2 * (1 - norm.cdf(abs(z))))}


def per_cluster_stability(Z, k, n_subsamples=30, frac=0.8, rng=None):
    """Fit k-means at a given k; per cluster, the mean Jaccard overlap with its matched cluster across refits."""
    rng = np.random.default_rng(rng)
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    jac = np.zeros((n_subsamples, k))
    for s in range(n_subsamples):
        i = rng.choice(len(Z), int(frac * len(Z)), replace=False)
        sub = KMeans(k, n_init=3, random_state=int(rng.integers(1 << 31))).fit(Z[i])
        r, c = linear_sum_assignment(cdist(km.cluster_centers_, sub.cluster_centers_))
        full_lab, sub_lab = km.labels_[i], sub.labels_
        for j, cc in zip(r, c):
            a, b = full_lab == j, sub_lab == cc
            jac[s, j] = (a & b).sum() / max((a | b).sum(), 1)
    return km, jac.mean(0)


def merge_plan(sizes, response, k_target, names=None):
    """Greedy merges by outcome similarity: at each step merge the pair whose merge loses the least expected outcome.

    `response[s, c]` is segment s's response rate to campaign c; each segment gets its best campaign.
    Returns the steps (who merged with whom, cost in expected conversions per 1,000 customers) and the final groups.
    """
    sizes = np.asarray(sizes, float)
    response = np.asarray(response, float)
    groups = [[s] for s in range(len(sizes))]
    names = list(names) if names else [f"S{s}" for s in range(len(sizes))]
    steps = []
    total = sizes.sum()

    def value(group):
        n = sizes[group].sum()
        r = (sizes[group, None] * response[group]).sum(0) / n
        return n * r.max()

    while len(groups) > k_target:
        best = None
        for a, b in itertools.combinations(range(len(groups)), 2):
            cost = value(groups[a]) + value(groups[b]) - value(groups[a] + groups[b])
            if best is None or cost < best[0]:
                best = (cost, a, b)
        cost, a, b = best
        steps.append((" + ".join(names[s] for s in groups[a]), " + ".join(names[s] for s in groups[b]),
                      1000 * cost / total))
        groups[a] = groups[a] + groups[b]
        del groups[b]
    final = [" + ".join(names[s] for s in g) for g in groups]
    return steps, ["merge", "with", "cost: conversions lost per 1,000 customers"], final


def pair_merge_costs(sizes, response, names=None):
    """The cost of merging each pair of segments (conversions lost per 1,000 of all customers), cheapest first."""
    sizes = np.asarray(sizes, float)
    response = np.asarray(response, float)
    names = list(names) if names else [f"S{s}" for s in range(len(sizes))]
    total = sizes.sum()

    def value(group):
        n = sizes[group].sum()
        return n * ((sizes[group, None] * response[group]).sum(0) / n).max()

    rows = [(names[a], names[b], 1000 * (value([a]) + value([b]) - value([a, b])) / total)
            for a, b in itertools.combinations(range(len(sizes)), 2)]
    return sorted(rows, key=lambda r: r[2]), ["segment", "merged with", "cost: conversions lost per 1,000 customers"]


# ----------------------------------------------------------------------------------------------- part 4: profiling
def profile_clusters(X_raw, labels, feature_names, Z=None, top=3, sample=4000, rng=None):
    """Per cluster: size, share, medians in original units, the 3 most distinguishing features (z vs overall), silhouette."""
    rng = np.random.default_rng(rng)
    k = labels.max() + 1
    mu, sd = X_raw.mean(0), X_raw.std(0) + 1e-12
    sil = None
    if Z is not None:
        i = rng.choice(len(Z), min(sample, len(Z)), replace=False)
        s = silhouette_samples(Z[i], labels[i])
        sil = {c: float(s[labels[i] == c].mean()) for c in range(k) if (labels[i] == c).any()}
    rows = []
    for c in range(k):
        m = labels == c
        if not m.any():
            continue
        z = (X_raw[m].mean(0) - mu) / sd
        order = np.argsort(-np.abs(z))[:top]
        rows.append({"cluster": c, "size": int(m.sum()), "share": float(m.mean()),
                     "medians": np.median(X_raw[m], 0),
                     "top": [(feature_names[j], float(np.median(X_raw[m, j])), float(z[j])) for j in order],
                     "silhouette": None if sil is None else sil.get(c)})
    return rows


def format_profiles(rows, names=None):
    out = []
    for r in rows:
        feats = ", ".join(f"{f} {v:.3g} ({z:+.1f} sd)" for f, v, z in r["top"])
        sil = "—" if r["silhouette"] is None else f"{r['silhouette']:.2f}"
        out.append((names.get(r["cluster"], r["cluster"]) if names else r["cluster"],
                    f"{r['size']:,} ({r['share']:.0%})", sil, feats))
    return format_table(out, ["cluster", "size", "silhouette", "distinguishing features (median, difference from average)"])


# ----------------------------------------------------------------------------------------------- the whole path
def retrain_path(mv: ModelVersion, batches, t_alarm_window_earliest, t_alarm, meaningful_fn: Callable = None,
                 k_range=range(2, 9), rng=None):
    """Steps 1-5 in order: backtest the window, k-scan, fit the candidate, then the same-k or different-k decision.

    The test batch (t_alarm) stays out of every fit; the alarm window (t_alarm_window_earliest .. t_alarm) is the
    post-change pool used for the time slices of 'real?', the contingency table and the far-point shares.
    meaningful_fn(mv_old, mv_new, X_pool, pool_batch_ids) -> {"meaningful": bool, ...} runs the business checks
    (profiles differ, outcomes differ, a holdout beats the old segmentation); without it the different-k branch
    stops at 'real?'.
    """
    rng = np.random.default_rng(rng)
    out = {"t_alarm": t_alarm, "t_alarm_window_earliest": t_alarm_window_earliest}
    bt = backtest_window(batches, mv, t_alarm_window_earliest, t_alarm)
    out["backtest"] = bt
    Xw, test = bt["X"], bt["test"]
    scaler = StandardScaler().fit(Xw)
    scan = k_scan(scaler.transform(Xw), k_range, rng=rng)
    out["scan"] = scan
    k_new = scan["k"]
    mv_new = fit_model_version(Xw, k_new, mv.feature_names, mv.batch_size, version=mv.version + 1, previous=mv,
                               train_window=(bt["batch_ids"][0], bt["batch_ids"][-1]), k_evidence=scan, rng=rng)
    out["candidate"] = mv_new
    ids = [b for b, _ in batches]
    window = batches[ids.index(t_alarm_window_earliest): ids.index(t_alarm) + 1]
    out["pool_ids"] = [b for b, _ in window]
    pool = np.vstack([X for _, X in window])
    far = new_cluster_far_share(mv, pool, mv_new.predict(pool))
    mv_new.names, out["name_table"], out["name_headers"] = carry_names(mv, mv_new, far_share=far)
    out["fit_checks"], _, out["hidden"] = fit_checks(mv_new, test, rng=rng)
    review = "" if all(out["fit_checks"].values()) else "; a fit check failed: review the method (step 3)"
    if k_new == mv.k:
        out["branch"] = "same k"
        out["same_k"] = same_k_decision(mv, mv_new, Xw, test, rng=rng)
        out["decision"] = out["same_k"]["decision"] + review
        return out
    out["branch"] = "different k"
    slices = [scaler.transform(X) for _, X in window] if len(window) >= 2 else None
    out["real"] = new_k_is_real(scaler.transform(Xw), mv.k, k_new, time_slices=slices, rng=rng)
    M = contingency(mv.predict(pool), mv_new.predict(pool))
    out["contingency"], out["far_share"], out["contingency_class"] = M, far, classify_contingency(M, far_share=far)
    tidy = out["real"]["real"] and "reorganisation" not in out["contingency_class"]
    if not tidy:
        # the new k is not real: keep the old k, but the data may still have moved, so step 5 decides on a same-k refit
        why = "not real" if not out["real"]["real"] else "a reorganisation, not a clean split or merge"
        same = fit_model_version(Xw, mv.k, mv.feature_names, mv.batch_size, version=mv.version + 1, previous=mv,
                                 train_window=(bt["batch_ids"][0], bt["batch_ids"][-1]), k_evidence=scan, rng=rng)
        same.names, _, _ = carry_names(mv, same)
        out["same_k_candidate"] = same
        out["same_k"] = same_k_decision(mv, same, Xw, test, rng=rng)
        out["decision"] = f"the new k is {why}: keep the old k; on a same-k refit, {out['same_k']['decision']}" + review
        return out
    if meaningful_fn is None:
        out["decision"] = "real; 'meaningful?' needs the business checks (no meaningful_fn given)" + review
        return out
    out["meaningful"] = meaningful_fn(mv, mv_new, pool, out["pool_ids"])
    out["decision"] = ("adopt the new k; map names" if out["meaningful"]["meaningful"]
                       else "keep the old k for actions; track the new structure as sub-segments") + review
    return out
