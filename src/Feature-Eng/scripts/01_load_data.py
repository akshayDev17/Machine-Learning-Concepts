"""Stage 01 · load the raw table.

Required: every later stage needs a table to work on.
"""
from pathlib import Path

import pandas as pd


def load_table(path):
    """Read the CSV at `path` and return it as a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with:  python src/Feature-Eng/data/make_customer_churn.py")
    return pd.read_csv(path)
