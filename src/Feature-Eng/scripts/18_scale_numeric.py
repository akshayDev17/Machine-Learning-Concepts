"""Stage 18 · put continuous columns on a common scale.

* `standard`: subtract the mean, divide by the standard deviation (most columns end up between about -3 and 3).
* `robust`: subtract the median, divide by the IQR (outliers barely affect the result).
* `minmax`: squeeze into [0, 1].
Matters for models that compare distances or penalise weights (k-NN, logistic regression with regularisation, SVM,
neural networks). Tree models don't care: they only ask "is x above some threshold?", and scaling keeps the order.
The statistics are learned on the training rows. 0/1 columns are left as they are.
"""
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from column_kinds import continuous_columns

SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


def fit_transform(train, target, params):
    """Learn scaling statistics on the training rows. Return (scaled train, state)."""
    columns = continuous_columns(train, target, min_unique=3)
    scaler = SCALERS[params["method"]]().fit(train[columns]) if columns else None
    state = {"columns": columns, "scaler": scaler,
             "summary": f"{params['method']} scaling of {len(columns)} columns"}
    return transform(train, state), state


def transform(frame, state):
    """Scale the same columns with the training statistics."""
    frame = frame.copy()
    if state["scaler"] is not None:
        frame[state["columns"]] = state["scaler"].transform(frame[state["columns"]])
    return frame
