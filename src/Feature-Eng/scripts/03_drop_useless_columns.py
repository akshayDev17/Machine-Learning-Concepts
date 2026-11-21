"""Stage 03 · drop columns that cannot carry information: IDs, constants and exact duplicates.

An ID is different on every row, so a model can only memorise it. A constant is the same on every row, so it can't
tell rows apart. A duplicate repeats a column that is already there.
"""
from column_kinds import text_columns


def fit_transform(train, target, params):
    """Decide which columns to drop by looking at the training rows. Return (train without them, state)."""
    reasons = {}
    for column in text_columns(train, target):
        if train[column].nunique() / len(train) >= params["id_unique_ratio"]:
            reasons[column] = "id-like (almost every value is different)"
    if params["drop_constant"]:
        for column in train.columns.drop(target):
            if train[column].nunique(dropna=False) <= 1:
                reasons[column] = "constant"
    if params["drop_duplicates"]:
        kept = []
        for column in train.columns.drop(target):
            twin = next((k for k in kept if train[column].equals(train[k])), None)
            if twin is None:
                kept.append(column)
            else:
                reasons.setdefault(column, f"duplicate of {twin}")
    state = {"columns_to_drop": list(reasons),
             "summary": f"dropped {len(reasons)}: " + ", ".join(f"{c} ({r})" for c, r in reasons.items())}
    return transform(train, state), state


def transform(frame, state):
    """Drop the columns chosen on the training rows."""
    return frame.drop(columns=state["columns_to_drop"])
