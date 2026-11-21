"""Stage 14 · cut chosen numeric columns into bins and add one 0/1 column per bin.

Bins let a linear model give each range its own effect, so it can follow any shape (a U, a step) without knowing
the formula. The cost: every value inside a bin is treated the same, so detail is lost. The bin edges are learned on
the training rows (quantile = equal-sized bins, uniform = equal-width bins). A missing value gets 0 in every bin.
"""
import numpy as np

from column_kinds import numeric_columns


def fit_transform(train, target, params):
    """Learn bin edges on the training rows. Return (train with bin columns, state)."""
    edges = {}
    for column in [c for c in params["columns"] if c in numeric_columns(train, target)]:
        values = train[column].dropna()
        if params["strategy"] == "quantile":
            cuts = np.unique(values.quantile(np.linspace(0, 1, params["n_bins"] + 1)).to_numpy())
        else:
            cuts = np.linspace(values.min(), values.max(), params["n_bins"] + 1)
        edges[column] = cuts[1:-1].tolist()               # inner edges; outer bins are open-ended
    state = {"edges": edges, "keep_original": params["keep_original"]}
    out = transform(train, state)
    state["summary"] = ", ".join(f"{c} -> {len(e) + 1} bins (edges {np.round(e, 1).tolist()})"
                                 for c, e in edges.items()) or "no columns to bin"
    return out, state


def transform(frame, state):
    """Add `<column>_bin<k>` 0/1 columns using the learned edges."""
    frame = frame.copy()
    for column, inner_edges in state["edges"].items():
        bin_index = np.digitize(frame[column], inner_edges)
        for k in range(len(inner_edges) + 1):
            frame[f"{column}_bin{k}"] = ((bin_index == k) & frame[column].notna()).astype(int)
        if not state["keep_original"]:
            frame = frame.drop(columns=column)
    return frame
