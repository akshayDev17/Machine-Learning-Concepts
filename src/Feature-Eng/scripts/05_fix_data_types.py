"""Stage 05 · fix data types: numbers stored as text, yes/no stored as text, messy spellings, impossible values.

Nothing is learned from the data here, so the same rules apply to train and test. Dates are parsed by stage 06.
"""
import numpy as np
import pandas as pd

from column_kinds import text_columns

YES_NO = {"y": 1, "yes": 1, "true": 1, "1": 1, "n": 0, "no": 0, "false": 0, "0": 0}


def fit_transform(train, target, params):
    """Apply the fixing rules to the training rows. Return (fixed train, state)."""
    state = {**params}
    fixed = transform(train, state)
    before = {c: train[c].nunique() for c in params["tidy_columns"] if c in train.columns}
    after = {c: fixed[c].nunique() for c in before}
    tidied = ", ".join(f"{c} {before[c]}->{after[c]} values" for c in before if before[c] != after[c])
    state["summary"] = (f"money -> float: {[c for c in params['money_columns'] if c in train.columns]}; "
                        f"yes/no -> 0/1: {[c for c in params['yes_no_columns'] if c in train.columns]}; "
                        f"tidied spellings: {tidied or 'none'}; impossible values -> missing: "
                        f"{list(params['valid_ranges'])}")
    return fixed, state


def transform(frame, state):
    """Apply the same fixing rules to any table."""
    frame = frame.copy()
    for column in [c for c in state["money_columns"] if c in frame.columns]:
        frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(r"[$,\s]", "", regex=True),
                                      errors="coerce")
    for column in [c for c in state["yes_no_columns"] if c in frame.columns]:
        frame[column] = frame[column].astype(str).str.strip().str.lower().map(YES_NO)
    for column in [c for c in state["tidy_columns"] if c in text_columns(frame, None)]:
        tidy = frame[column].str.strip().str.lower().str.replace("_", " ")
        frame[column] = tidy.where(frame[column].notna(), np.nan)
    for column, (low, high) in state["valid_ranges"].items():
        if column in frame.columns:
            frame[column] = frame[column].where(frame[column].between(low, high) | frame[column].isna(), np.nan)
    return frame
