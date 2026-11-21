"""Stage 19 · train the model and score it.

Required: the point of every earlier stage is a better score here.
Scores: accuracy (share of customers classified correctly; 0.5 = coin flip on this balanced target) and ROC-AUC
(the chance that a random churner gets a higher churn score than a random non-churner; 0.5 = coin flip, 1 = perfect).
Only numeric columns can go into a model. If an earlier stage was switched off and text columns or gaps remain,
they are handled here (text dropped, gaps filled with 0) and reported, so you can see the cost of skipping a stage.
"""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from column_kinds import numeric_columns, text_columns

HANDLES_GAPS = {"hist_gradient_boosting", "random_forest"}


def make_model(params):
    """Return an untrained scikit-learn classifier described by `params`."""
    name = params["model"]
    if name == "logistic_regression":
        return LogisticRegression(C=params["C"], max_iter=5000)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=params["random_state"],
                                      n_jobs=-1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(learning_rate=0.05, random_state=params["random_state"])
    raise ValueError(f"unknown model {name!r}; use logistic_regression, knn, random_forest or hist_gradient_boosting")


def to_matrix(frame, target, model_name):
    """Return (X, y, notes): the numeric feature table, the labels, and what had to be patched to make them usable."""
    notes = []
    left_as_text = text_columns(frame, target)
    if left_as_text:
        notes.append(f"dropped {len(left_as_text)} text columns the model can't read: {left_as_text}")
    X = frame[numeric_columns(frame, target)]
    gaps = int(X.isna().sum().sum())
    if gaps and model_name not in HANDLES_GAPS:
        notes.append(f"filled {gaps} remaining gaps with 0 ({model_name} can't handle missing values)")
        X = X.fillna(0)
    return X, frame[target], notes


def fit_and_score(train, valid, target, params):
    """Train on `train`, score on `valid`. Return (scores dict, notes, number of features)."""
    X_train, y_train, notes = to_matrix(train, target, params["model"])
    X_valid, y_valid, _ = to_matrix(valid, target, params["model"])
    model = make_model(params).fit(X_train, y_train)
    churn_probability = model.predict_proba(X_valid)[:, 1]
    scores = {"accuracy": accuracy_score(y_valid, churn_probability >= 0.5),
              "roc_auc": roc_auc_score(y_valid, churn_probability)}
    return scores, notes, X_train.shape[1]
