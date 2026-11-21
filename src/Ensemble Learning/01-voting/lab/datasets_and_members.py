"""Cleaned real datasets and their ready-made members for voting_lab.ipynb.

Everything here is data preparation, so that the notebook can stay about voting. Data comes from the CSV files that
fetch_datasets.py wrote into openml_data/; nothing in this module goes online. Model settings were fixed in advance
and nothing was tuned.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

DATA_HOME = Path(__file__).parent / "openml_data"


def _read(name):
    """Read openml_data/<name>.csv. Only an empty cell is a missing value ("NA" is a real category in House Prices)."""
    path = DATA_HOME / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `python fetch_datasets.py` in {path.parent.parent} once, then try again.")
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


# ---------------------------------------------------------------------------------------------- Titanic
def load_titanic():
    """1,309 passengers: X = class, sex, age, siblings/spouses, parents/children, fare, port; y = 1 if the passenger survived.

    `boat` (lifeboat number) and `body` (body number) are left out: they record the outcome itself. Name, ticket, cabin
    and home destination are left out too (free text, or mostly empty).
    """
    titanic = _read("titanic")
    return titanic[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]], titanic["survived"].astype(int).to_numpy()


def _titanic_preprocessing():
    """Fill gaps (median / most common value), scale the numbers, one-hot the two categories."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([("num", numeric, ["pclass", "age", "sibsp", "parch", "fare"]),
                              ("cat", categorical, ["sex", "embarked"])])


def titanic_members():
    """Five different kinds of classifier as (name, model) pairs, each with the preprocessing it needs."""
    return [
        ("softmax", make_pipeline(_titanic_preprocessing(), LogisticRegression(max_iter=1000))),
        ("tree", make_pipeline(_titanic_preprocessing(), DecisionTreeClassifier(max_depth=5, random_state=0))),
        ("knn", make_pipeline(_titanic_preprocessing(), KNeighborsClassifier(15))),
        ("nb", make_pipeline(_titanic_preprocessing(), FunctionTransformer(_to_dense), GaussianNB())),   # GaussianNB needs dense input
        ("svc", make_pipeline(_titanic_preprocessing(), SVC())),
    ]


# ---------------------------------------------------------------------------------------------- House Prices
def load_house_prices():
    """1,460 houses from Ames, Iowa: X = the 79 description columns (Id dropped); y = log of the sale price."""
    houses = _read("house_prices")
    return houses.drop(columns=["SalePrice", "Id"]), np.log(houses["SalePrice"].to_numpy())


def _houses_preprocessing(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical)])


def house_members(X):
    """Three different kinds of regressor as (name, model) pairs; X is only used to see which columns are numbers."""
    return [("linear", make_pipeline(_houses_preprocessing(X), LinearRegression())),   # plain least squares, no regularisation
            ("tree", make_pipeline(_houses_preprocessing(X), DecisionTreeRegressor(max_depth=6, random_state=0))),
            ("knn", make_pipeline(_houses_preprocessing(X), KNeighborsRegressor(10)))]
