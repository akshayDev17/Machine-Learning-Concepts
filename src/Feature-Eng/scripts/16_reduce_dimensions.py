"""Stage 16 · compress a block of related columns into a few new ones (PCA, or LDA when labels are used).

Here the block is the 8 survey questions: they all measure one hidden thing (satisfaction), so a single summary
column keeps most of their information. The block is standardised first (mean 0, std 1) with training statistics.
* `pca` looks only at the columns: it keeps the directions in which the block varies most.
* `lda` also looks at the label: it keeps the direction that best separates churners from non-churners
  (for two classes there is exactly one such direction). Because it uses labels, it is fitted on training rows only.
"""
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fit_transform(train, target, params):
    """Fit the reducer on the training rows' block. Return (train with the block replaced, state)."""
    block = [c for c in train.columns if c.startswith(params["columns_prefix"])]
    if len(block) < 2:
        return train, {"block": [], "summary": f"skipped: fewer than 2 columns start with {params['columns_prefix']!r}"}
    means, stds = train[block].mean(), train[block].std().replace(0, 1)
    standardised = (train[block] - means) / stds
    if params["method"] == "lda":
        reducer = LinearDiscriminantAnalysis(n_components=1).fit(standardised, train[target])
    else:
        reducer = PCA(n_components=params["n_components"], random_state=params["random_state"]).fit(standardised)
    state = {"block": block, "means": means, "stds": stds, "reducer": reducer, "method": params["method"],
             "output_prefix": params["output_prefix"]}
    out = transform(train, state)
    kept = (f", {reducer.explained_variance_ratio_.sum():.0%} of the block's variance kept"
            if params["method"] == "pca" else "")
    state["summary"] = f"{params['method']}: {len(block)} columns -> {out.shape[1] - train.shape[1] + len(block)}{kept}"
    return out, state


def transform(frame, state):
    """Replace the block with its compressed columns."""
    if not state["block"]:
        return frame
    frame = frame.copy()
    scores = state["reducer"].transform((frame[state["block"]] - state["means"]) / state["stds"])
    for k in range(scores.shape[1]):
        frame[f"{state['output_prefix']}_{state['method']}{k + 1}"] = scores[:, k]
    return frame.drop(columns=state["block"])
