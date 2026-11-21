"""Stage 04 · remove columns that leak the answer.

A leaky column is only known *because of* the outcome, or after it (here: a retention call is made after a customer
asks to cancel). It makes validation scores look wonderful and is useless when predicting the future. Only domain
knowledge can confirm a leak, so the columns to drop are listed in the config. The stage also scans every column on
its own and flags any that predicts the target suspiciously well, so you can check it.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def out_of_fold_rates(values, y, random_state):
    """Each row's category churn rate, computed on the OTHER rows (5 folds), so a near-unique text column can't
    look predictive just by memorising its own labels. Unseen categories get the overall rate."""
    rates = pd.Series(float(y.mean()), index=values.index)
    for fit_rows, score_rows in KFold(5, shuffle=True, random_state=random_state).split(values):
        per_category = y.iloc[fit_rows].groupby(values.iloc[fit_rows]).mean()
        rates.iloc[score_rows] = values.iloc[score_rows].map(per_category).fillna(y.iloc[fit_rows].mean()).to_numpy()
    return rates


def single_column_auc(values, y, random_state):
    """ROC-AUC of one column used alone as a score (text columns: out-of-fold category churn rates)."""
    if not pd.api.types.is_numeric_dtype(values):
        values = out_of_fold_rates(values.fillna("missing"), y, random_state)
    values = values.fillna(values.median())
    auc = roc_auc_score(y, values)
    return max(auc, 1 - auc)


def fit_transform(train, target, params):
    """Drop the listed leaky columns; flag other columns that look too good. Return (train, state)."""
    to_drop = [c for c in params["columns"] if c in train.columns]
    scan = {c: single_column_auc(train[c], train[target], params["random_state"]) for c in train.columns.drop(target)}
    suspicious = [f"{c} (AUC {auc:.2f})" for c, auc in scan.items() if auc > params["warn_auc_above"]
                  and c not in to_drop]
    state = {"columns_to_drop": to_drop,
             "summary": "dropped " + ", ".join(f"{c} (AUC {scan[c]:.2f} on its own)" for c in to_drop)
                        + f"; suspicious columns still present: {suspicious or 'none'}"}
    return transform(train, state), state


def transform(frame, state):
    """Drop the leaky columns."""
    return frame.drop(columns=state["columns_to_drop"])
