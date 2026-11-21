"""Stage 15 · drop columns that are near-copies of another column.

When two columns are almost perfectly correlated (|r| above `threshold`), the second says nothing new. Linear models
then can't decide how to split the credit between the two, so their coefficients become unstable. For each such pair
the column that comes later in the table is dropped. The decision is made on the training rows.
"""
import numpy as np

from column_kinds import numeric_columns


def fit_transform(train, target, params):
    """Find near-duplicate columns on the training rows. Return (train without them, state)."""
    columns = [c for c in numeric_columns(train, target) if train[c].nunique() > 1]
    corr = train[columns].corr().abs().to_numpy()
    dropped = {}
    for j in range(len(columns)):
        for i in range(j):
            if columns[i] not in dropped and corr[i, j] > params["threshold"] and not np.isnan(corr[i, j]):
                dropped[columns[j]] = (columns[i], corr[i, j])
                break
    state = {"columns_to_drop": list(dropped),
             "summary": f"|r| > {params['threshold']}: dropped "
                        + (", ".join(f"{c} (r={r:.2f} with {k})" for c, (k, r) in dropped.items()) or "nothing")}
    return transform(train, state), state


def transform(frame, state):
    """Drop the near-duplicate columns."""
    return frame.drop(columns=state["columns_to_drop"])
