"""Stage 17 · keep only the columns that help predict the target.

Three families (all fitted on the training rows only; choosing columns while looking at test rows is leakage):
* filter   `mutual_info`: score each column on its own against the target, keep the best `k`.
* wrapper  `rfe`: fit a model, drop the weakest column, refit, repeat until `k` remain.
* embedded `l1`: fit an L1 (lasso) logistic regression; columns whose weight is pushed to exactly 0 are dropped.
The selector sees a standardised copy with gaps filled by the median, so it works whatever earlier stages ran.
Text columns (if encoding was switched off) are passed through untouched.
"""
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from column_kinds import numeric_columns


def fit_transform(train, target, params):
    """Choose columns on the training rows. Return (train with only those columns, state)."""
    candidates = numeric_columns(train, target)
    X = train[candidates].fillna(train[candidates].median())
    X = (X - X.mean()) / X.std().replace(0, 1)
    y = train[target]
    if params["method"] == "mutual_info":
        scores = mutual_info_classif(X, y, random_state=params["random_state"])
        keep = [c for _, c in sorted(zip(scores, candidates), reverse=True)[:params["k"]]]
    elif params["method"] == "rfe":
        rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=params["k"]).fit(X, y)
        keep = [c for c, chosen in zip(candidates, rfe.support_) if chosen]
    else:
        lasso = LogisticRegression(l1_ratio=1.0, C=params["C"], solver="liblinear", max_iter=2000).fit(X, y)
        keep = [c for c, w in zip(candidates, lasso.coef_[0]) if w != 0]
    dropped = [c for c in candidates if c not in keep]
    state = {"columns_to_drop": dropped,
             "summary": f"{params['method']}: kept {len(keep)} of {len(candidates)}; dropped {dropped}"}
    return transform(train, state), state


def transform(frame, state):
    """Drop the columns the selector did not choose."""
    return frame.drop(columns=state["columns_to_drop"])
