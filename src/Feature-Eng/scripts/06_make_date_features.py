"""Stage 06 · turn date columns into numbers a model can use.

A raw date string means nothing to a model. Useful numbers hidden in it:
* days since the date (how long ago, measured from a fixed reference day, the day the data was exported);
* the month, written as a point on a circle (sin, cos) so that December sits next to January;
* the day of the week, also on a circle.
Nothing is learned from the data, so train and test get identical treatment.
"""
import numpy as np
import pandas as pd


def fit_transform(train, target, params):
    """Create the date features on the training rows. Return (train, state)."""
    state = {**params}
    out = transform(train, state)
    state["summary"] = f"from {params['date_columns']} made {[c for c in out.columns if c not in train.columns]}"
    return out, state


def transform(frame, state):
    """Replace each date column with days-since and cyclical month / weekday features."""
    frame = frame.copy()
    reference = pd.Timestamp(state["reference_date"])
    for column in [c for c in state["date_columns"] if c in frame.columns]:
        dates = pd.to_datetime(frame[column], errors="coerce")
        frame[f"{column}_days_since"] = (reference - dates).dt.days
        if state["cyclical_month"]:
            frame[f"{column}_month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
            frame[f"{column}_month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
        if state["cyclical_weekday"]:
            frame[f"{column}_weekday_sin"] = np.sin(2 * np.pi * dates.dt.day_of_week / 7)
            frame[f"{column}_weekday_cos"] = np.cos(2 * np.pi * dates.dt.day_of_week / 7)
        frame = frame.drop(columns=column)
    return frame
