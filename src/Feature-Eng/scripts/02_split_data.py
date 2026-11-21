"""Stage 02 · split into train and test, and cut the train part into cross-validation folds.

Required: every later stage learns from the training rows only. Without a split there is nothing honest to test on.
Stratified means each part keeps the same share of churners as the full table (50 / 50 here).
"""
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_train_test(frame, target, test_size, random_state, stratify):
    """Return (train, test) DataFrames. The test part is locked away until the very end."""
    return train_test_split(frame, test_size=test_size, random_state=random_state,
                            stratify=frame[target] if stratify else None)


def make_cv_folds(train, target, n_folds, random_state):
    """Return a list of (fit_rows, valid_rows) position arrays: n_folds stratified folds of the training table."""
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(folds.split(train, train[target]))
