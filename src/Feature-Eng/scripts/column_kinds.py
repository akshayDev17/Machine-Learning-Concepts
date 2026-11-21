"""Small helpers that sort a table's columns into kinds. Used by several stages."""
import pandas as pd


def text_columns(frame, target):
    """Columns holding strings (categories or free text), excluding the target."""
    return [c for c in frame.columns
            if c != target and not pd.api.types.is_numeric_dtype(frame[c])
            and not pd.api.types.is_datetime64_any_dtype(frame[c])]


def numeric_columns(frame, target):
    """Columns holding numbers (including 0/1 flags), excluding the target."""
    return [c for c in frame.columns if c != target and pd.api.types.is_numeric_dtype(frame[c])]


def continuous_columns(frame, target, min_unique):
    """Numeric columns with at least `min_unique` distinct values: real measurements, not flags or codes."""
    return [c for c in numeric_columns(frame, target) if frame[c].nunique() >= min_unique]
