"""Stage 08 · fill missing values, and keep a "was missing" flag so the model still knows a gap was there.

Numbers are filled with the training median (or mean); text with the word "missing" (or the most frequent value).
The fill values are learned on the training rows only. A flag is added for every column whose share of missing
values in the training rows is at least `flag_min_rate`: when missingness itself says something (here, churners
hide their income more often), the flag carries that signal.
"""
from column_kinds import numeric_columns, text_columns


def fit_transform(train, target, params):
    """Learn fill values (and which columns get a flag) on the training rows. Return (train, state)."""
    numeric = numeric_columns(train, target)
    stat = params["numeric_strategy"]
    fill = {c: getattr(train[c], stat)() for c in numeric}
    for column in text_columns(train, target):
        if params["text_strategy"] == "most_frequent" and train[column].notna().any():
            fill[column] = train[column].mode().iloc[0]
        else:
            fill[column] = "missing"
    missing_rate = train.drop(columns=target).isna().mean()
    flagged = list(missing_rate[missing_rate >= params["flag_min_rate"]].index) if params["add_missing_flags"] else []
    state = {"fill_values": fill, "flag_columns": flagged,
             "summary": f"filled {int(train.drop(columns=target).isna().sum().sum())} gaps "
                        f"(numbers: {stat}, text: {params['text_strategy']}); flags added for {flagged}"}
    return transform(train, state), state


def transform(frame, state):
    """Add the flags, then fill every gap with the learned value."""
    frame = frame.copy()
    for column in state["flag_columns"]:
        frame[f"{column}_was_missing"] = frame[column].isna().astype(int)
    return frame.fillna({c: v for c, v in state["fill_values"].items() if c in frame.columns})
