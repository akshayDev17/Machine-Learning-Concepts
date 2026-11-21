"""Stage 12 · turn text categories into numbers.

* Ordinal: categories with a natural order (basic < standard < premium) become 0, 1, 2.
* One-hot: few categories, no order -> one 0/1 column per category (a 2-category column becomes a single 0/1 column).
* Many categories (more than `max_onehot_levels`):
  - `target`: each category becomes the churn rate of that category, shrunk towards the overall rate when the
    category is small. For the training rows the rate is computed *cross-fitted*: each row's value comes from the
    other folds, never from its own label. Otherwise the model would see its own answer (leakage).
  - `frequency`: each category becomes its share of the training rows. Uses no labels, so no leakage risk.
  - `onehot`: one column per category anyway.
Categories never seen in training get all-zero one-hot columns, the overall rate, or a frequency of 0.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from column_kinds import text_columns


def fit_transform(train, target, params):
    """Choose an encoding per text column and learn it on the training rows. Return (encoded train, state)."""
    ordinal = {c: order for c, order in params["ordinal"].items() if c in train.columns}
    onehot, frequency, target_encoded = {}, {}, []
    for column in [c for c in text_columns(train, target) if c not in ordinal]:
        levels = sorted(train[column].dropna().unique())
        if len(levels) <= params["max_onehot_levels"] or params["high_cardinality"] == "onehot":
            onehot[column] = levels[1:] if len(levels) == 2 else levels
        elif params["high_cardinality"] == "frequency":
            frequency[column] = train[column].value_counts(normalize=True).to_dict()
        else:
            target_encoded.append(column)
    encoder = None
    state = {"ordinal": ordinal, "onehot": onehot, "frequency": frequency, "target_encoded": target_encoded}
    out = transform(train, {**state, "encoder": None})
    if target_encoded:
        folds = StratifiedKFold(params["target_cv_folds"], shuffle=True, random_state=params["random_state"])
        encoder = TargetEncoder(target_type="binary", cv=folds)
        cross_fitted = encoder.fit_transform(train[target_encoded].astype(str), train[target])
        out[[f"{c}_target_enc" for c in target_encoded]] = cross_fitted
    state["encoder"] = encoder
    state["summary"] = (f"ordinal: {list(ordinal)}; one-hot: {list(onehot)}; "
                        f"{params['high_cardinality']}-encoded: {target_encoded or list(frequency)}")
    return out, state


def transform(frame, state):
    """Encode new rows with what was learned on the training rows."""
    frame = frame.copy()
    for column, order in state["ordinal"].items():
        frame[column] = frame[column].map({level: rank for rank, level in enumerate(order)})
    for column, levels in state["onehot"].items():
        dummies = pd.DataFrame({f"{column}={level}": (frame[column] == level).astype(int) for level in levels},
                               index=frame.index)
        frame = pd.concat([frame.drop(columns=column), dummies], axis=1)
    for column, share in state["frequency"].items():
        frame[f"{column}_freq"] = frame[column].map(share).fillna(0.0)
        frame = frame.drop(columns=column)
    if state["encoder"] is not None:
        frame[[f"{c}_target_enc" for c in state["target_encoded"]]] = state["encoder"].transform(
            frame[state["target_encoded"]].astype(str))
    return frame.drop(columns=[c for c in state["target_encoded"] if c in frame.columns])
