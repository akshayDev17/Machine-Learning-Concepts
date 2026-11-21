"""Stage 09 · cap (winsorise) extreme values of continuous columns.

Every value below the lower limit is raised to it, every value above the upper limit is lowered to it. The limits
are learned on the training rows: either quantiles (e.g. the 1st and 99th percentile) or the IQR rule
(Q1 - k·IQR, Q3 + k·IQR). Only columns with at least `min_unique` distinct values are touched, so flags, codes and
cyclical date features are left alone.
"""
from column_kinds import continuous_columns


def fit_transform(train, target, params):
    """Learn the limits on the training rows. Return (capped train, state)."""
    limits = {}
    for column in continuous_columns(train, target, params["min_unique"]):
        values = train[column]
        if params["method"] == "iqr":
            q1, q3 = values.quantile([0.25, 0.75])
            low, high = q1 - params["iqr_factor"] * (q3 - q1), q3 + params["iqr_factor"] * (q3 - q1)
        else:
            low, high = values.quantile([params["lower_quantile"], params["upper_quantile"]])
        limits[column] = (float(low), float(high))
    capped = {c: int(((train[c] < lo) | (train[c] > hi)).sum()) for c, (lo, hi) in limits.items()}
    state = {"limits": limits,
             "summary": f"{params['method']} limits on {len(limits)} columns; values capped in train: "
                        + ", ".join(f"{c} {n}" for c, n in capped.items() if n)}
    return transform(train, state), state


def transform(frame, state):
    """Clip each column to its learned limits."""
    frame = frame.copy()
    for column, (low, high) in state["limits"].items():
        frame[column] = frame[column].clip(low, high)
    return frame
