"""Stage 11 · group rare categories into one "other" category.

A category seen in only a handful of training rows gives the model almost nothing to learn from, and it may not
appear in new data at all. Categories whose share of the training rows is below `min_share` become "other".
Categories never seen in training (new cities at prediction time) also become "other".
"""
from column_kinds import text_columns


def fit_transform(train, target, params):
    """Learn which categories are common enough to keep. Return (train, state)."""
    keep = {}
    for column in text_columns(train, target):
        share = train[column].value_counts(normalize=True)
        keep[column] = list(share[share >= params["min_share"]].index)
    state = {"keep": keep, "other_label": params["other_label"]}
    out = transform(train, state)
    state["summary"] = ", ".join(f"{c} {train[c].nunique()}->{out[c].nunique()} levels" for c in keep
                                 if train[c].nunique() != out[c].nunique()) or "no rare categories"
    return out, state


def transform(frame, state):
    """Replace every category not in the keep-list with the "other" label (missing values stay missing)."""
    frame = frame.copy()
    for column, kept in state["keep"].items():
        if column in frame.columns:
            rare = frame[column].notna() & ~frame[column].isin(kept)
            frame[column] = frame[column].where(~rare, state["other_label"])
    return frame
