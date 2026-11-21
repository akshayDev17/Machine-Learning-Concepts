"""Stage 13 · add interaction (product) and squared columns.

A linear model adds up one effect per column. If the effect of one column depends on another (support calls hurt
premium customers more), the product of the two gives the model that "it depends" term. A squared column lets a
linear model draw a U shape (young and old customers churn more than middle-aged ones). The column is centred on its
training mean before squaring, so the square is not just a near-copy of the column itself.
Pairs that refer to a missing or non-numeric column (e.g. `plan` when encoding is switched off) are skipped and
reported.
"""
from column_kinds import numeric_columns


def fit_transform(train, target, params):
    """Learn the centres for squares on the training rows, then add products and squares. Return (train, state)."""
    state = {**params, "square_centres": {c: float(train[c].mean()) for c in params["squares"]
                                    if c in numeric_columns(train, target)}}
    out = transform(train, state)
    made = [c for c in out.columns if c not in train.columns]
    wanted = len(params["products"]) + len(params["squares"])
    state["summary"] = f"made {made}" + (f"; skipped {wanted - len(made)} (input column missing or not numeric)"
                                         if len(made) < wanted else "")
    return out, state


def transform(frame, state):
    """Add `a_x_b` for each configured pair and `c_squared` for each configured column."""
    frame = frame.copy()
    numeric = numeric_columns(frame, None)
    for left, right in state["products"]:
        if left in numeric and right in numeric:
            frame[f"{left}_x_{right}"] = frame[left] * frame[right]
    for column in state["square_centres"]:
        if column in frame.columns:
            frame[f"{column}_squared"] = (frame[column] - state["square_centres"][column]) ** 2
    return frame
