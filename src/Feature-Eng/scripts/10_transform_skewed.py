"""Stage 10 · un-skew long-tailed numeric columns (log, Box-Cox or Yeo-Johnson).

A column is "skewed" when a few rows have values far bigger than the rest (income, counts, spend). Squashing the
long tail makes the column's relationship with the target closer to a straight line, which helps linear models.
Which columns are skewed, and the transform's parameters, are learned on the training rows.
log1p needs values >= 0 and Box-Cox needs values > 0; columns that break the rule fall back to Yeo-Johnson.
"""
import numpy as np
from sklearn.preprocessing import PowerTransformer

from column_kinds import continuous_columns


def choose_method(values, method):
    """The requested method if the column's values allow it, otherwise Yeo-Johnson (works for any value)."""
    if method == "log1p" and values.min() >= 0:
        return "log1p"
    if method == "box-cox" and values.min() > 0:
        return "box-cox"
    return "yeo-johnson"


def fit_transform(train, target, params):
    """Find skewed columns on the training rows and fit one transform per column. Return (train, state)."""
    transforms = {}
    for column in continuous_columns(train, target, params["min_unique"]):
        if abs(train[column].skew()) > params["skew_threshold"]:
            method = choose_method(train[column], params["method"])
            fitted = None if method == "log1p" else PowerTransformer(method=method, standardize=False).fit(
                train[[column]])
            transforms[column] = {"method": method, "fitted": fitted, "skew_before": float(train[column].skew())}
    state = {"transforms": transforms}
    out = transform(train, state)
    state["summary"] = "un-skewed: " + (", ".join(
        f"{c} {t['method']} (skew {t['skew_before']:.1f} -> {out[c].skew():.1f})" for c, t in transforms.items())
        or "nothing above the threshold")
    return out, state


def transform(frame, state):
    """Apply each column's fitted transform."""
    frame = frame.copy()
    for column, spec in state["transforms"].items():
        if spec["method"] == "log1p":
            frame[column] = np.log1p(frame[column].clip(lower=0))
        else:
            frame[column] = spec["fitted"].transform(frame[[column]])[:, 0]
    return frame
