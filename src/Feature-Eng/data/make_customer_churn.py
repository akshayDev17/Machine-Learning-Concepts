"""Generate the messy customer-churn table used by the lab notebook and the pipeline scripts.

Run from anywhere:
    python src/Feature-Eng/data/make_customer_churn.py

It writes `customer_churn.csv` next to this file. The seed is fixed, so every run gives the same table.

The "hidden truth": each customer gets a churn score built from a few real effects (listed in `churn_score`),
plus logistic noise. The top half of scores become churners (target = 1), so the classes are exactly 50 / 50.
Then the clean table is made messy on purpose (wrong types, typos, missing values, outliers, a leaky column, ...).
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_ROWS = 3000
SEED = 42
SNAPSHOT_DATE = pd.Timestamp("2025-01-01")  # the day the table was exported


def make_hidden_drivers(rng, n):
    """The latent quantities that really drive churn. None of them is a column in the final table."""
    return {
        "satisfaction": rng.normal(0, 1, n),   # measured (noisily) by the 8 survey questions
        "engagement": rng.normal(0, 1, n),     # measured by the login / session columns
    }


def make_clean_columns(rng, n, drivers):
    """Columns as they would look in a perfect world: right types, no gaps, no typos."""
    signup = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1796, n), unit="D")
    age = np.clip(rng.normal(45, 15, n), 18, 85).round()
    income = np.exp(rng.normal(8.3, 0.55, n)).round(-1)                        # skewed, median ~4,000
    plan = rng.choice(["basic", "standard", "premium"], n, p=[0.45, 0.35, 0.20])
    base_price = pd.Series(plan).map({"basic": 29.0, "standard": 59.0, "premium": 99.0}).to_numpy()
    charges = (base_price + rng.normal(0, 6, n)).round(2)
    tenure_months = ((SNAPSHOT_DATE - signup).days / 30.44).to_numpy()
    total_spend = (charges * tenure_months * rng.uniform(0.9, 1.1, n)).round(2)
    support_calls = rng.poisson(np.exp(0.6 - 0.6 * drivers["satisfaction"]))
    weekly_logins = rng.poisson(np.exp(1.3 + 0.45 * drivers["engagement"]))
    monthly_logins = np.round(weekly_logins * 4.3 + rng.normal(0, 1.0, n)).clip(0).astype(int)
    avg_session = np.exp(2.6 + 0.35 * drivers["engagement"] + rng.normal(0, 0.3, n)).round(1)
    survey = {f"survey_q{i}": np.clip(np.round(3 + 0.9 * drivers["satisfaction"] + rng.normal(0, 0.9, n)), 1, 5).astype(int)
              for i in range(1, 9)}
    last_login_days = rng.exponential(np.exp(2.5 - 0.5 * drivers["engagement"]), n).round()

    city_names = [f"City_{i:03d}" for i in range(1, 121)]
    city_weights = 1 / np.arange(1, 121) ** 1.1                               # a few big cities, a long tail
    city = rng.choice(city_names, n, p=city_weights / city_weights.sum())
    region = rng.choice(["North", "South", "East", "West", "Central", "Islands"], n,
                        p=[0.27, 0.25, 0.2, 0.18, 0.09, 0.01])
    payment = rng.choice(["credit card", "bank transfer", "paypal", "mailed check"], n, p=[0.35, 0.25, 0.25, 0.15])
    device = rng.choice(["ios", "android", "web"], n, p=[0.4, 0.4, 0.2])

    return pd.DataFrame({
        "signup_date": signup, "age": age, "monthly_income": income, "plan": plan,
        "monthly_charges": charges, "total_spend": total_spend, "num_support_calls": support_calls,
        "weekly_logins": weekly_logins, "monthly_logins": monthly_logins, "avg_session_minutes": avg_session,
        "last_login_date": SNAPSHOT_DATE - pd.to_timedelta(last_login_days, unit="D"),
        "city": city, "region": region, "payment_method": payment, "device": device, **survey,
    })


def churn_score(rng, clean, drivers):
    """Sum of the real effects. Each line is one thing a good feature-engineering step can uncover."""
    city_effect = dict(zip(sorted(clean["city"].unique()), rng.normal(0, 0.7, clean["city"].nunique())))
    tenure_days = (SNAPSHOT_DATE - clean["signup_date"]).dt.days
    month = clean["signup_date"].dt.month
    days_since_login = (SNAPSHOT_DATE - clean["last_login_date"]).dt.days
    charges_to_income = clean["monthly_charges"] / clean["monthly_income"]
    premium = (clean["plan"] == "premium").astype(float)
    return (
        - 0.9 * drivers["satisfaction"]                                   # unhappy customers leave
        - 0.5 * drivers["engagement"]                                     # people who stop using it leave
        - 0.6 * (np.log(tenure_days) - np.log(tenure_days).mean())        # new customers leave more (log shape)
        + 0.9 * np.cos(2 * np.pi * (month - 0.5) / 12)                    # Dec/Jan promo sign-ups leave (cyclic)
        + 0.5 * np.log1p(days_since_login)                                # not logged in lately -> leaving
        + 0.7 * ((clean["age"] - 45) / 15) ** 2                           # young and old leave more (U shape)
        + 50.0 * (charges_to_income - charges_to_income.median())         # can't afford it -> leave (a ratio)
        + 1.0 * premium * np.log1p(clean["num_support_calls"])            # premium + problems -> leave (interaction)
        + 0.8 * (clean["payment_method"] == "mailed check")               # manual payers leave more
        + clean["city"].map(city_effect)                                  # some cities churn more (high cardinality)
    )


def make_target(rng, score):
    """Top half of (score + noise) churn, so the target is exactly balanced."""
    noisy = score + rng.logistic(0, 0.8, len(score))
    return (noisy > np.median(noisy)).astype(int)


def make_messy(rng, clean, target):
    """Turn the clean table into a realistic export: every problem the lab fixes is added here."""
    n = len(clean)
    messy = clean.copy()

    # IDs, constants and copies: columns with no information
    messy.insert(0, "customer_id", [f"CUST-{i:06d}" for i in rng.permutation(n) + 100000])
    messy["country"] = "US"
    messy["is_senior"] = np.where(clean["age"] >= 65, "Y", "N")              # string instead of boolean

    # wrong types: money as text, dates as text, inconsistent spellings
    dollar = rng.random(n) < 0.7
    messy["monthly_charges"] = np.where(dollar, "$" + clean["monthly_charges"].map("{:.2f}".format),
                                        clean["monthly_charges"].map("{:.2f}".format))
    messy["monthly_fee"] = messy["monthly_charges"]                          # exact duplicate column
    messy["signup_date"] = clean["signup_date"].dt.strftime("%Y-%m-%d")
    messy["last_login_date"] = clean["last_login_date"].dt.strftime("%Y-%m-%d")
    spellings = {"credit card": ["credit card", "Credit Card", "credit_card", " credit card "],
                 "bank transfer": ["bank transfer", "Bank Transfer", "bank_transfer"],
                 "paypal": ["paypal", "PayPal", " Paypal"],
                 "mailed check": ["mailed check", "Mailed Check", "mailed_check"]}
    messy["payment_method"] = [rng.choice(spellings[p]) for p in clean["payment_method"]]

    # free text: support ticket notes, more often about problems when the customer is unhappy
    topics = {"billing": ["Billing question", "billing issue - card declined", "Wrong amount BILLED"],
              "slow": ["app is slow", "Slow loading", "very slow at night"],
              "crash": ["app crash on login", "Crashes after update"],
              "praise": ["great service, thanks", "Love the new feature"]}
    notes = []
    for calls, is_churner in zip(clean["num_support_calls"], target):
        if calls == 0:
            notes.append(np.nan)
            continue
        p = [0.35, 0.25, 0.25, 0.15] if is_churner else [0.2, 0.2, 0.15, 0.45]
        notes.append(rng.choice(topics[rng.choice(list(topics), p=p)]))
    messy["support_note"] = notes

    # vendor scores bought from a third party: pure noise
    for i in range(1, 6):
        messy[f"ext_score_{i}"] = rng.normal(50, 10, n).round(1)

    # the leak: a retention call is made AFTER a customer asks to cancel
    messy["retention_call_made"] = np.where(rng.random(n) < np.where(target == 1, 0.92, 0.04), "yes", "no")

    # missing values: age is missing at random, income is missing more often for churners
    messy.loc[rng.random(n) < 0.08, "age"] = np.nan
    income_missing = rng.random(n) < np.where(target == 1, 0.25, 0.07)
    messy.loc[income_missing, "monthly_income"] = np.nan
    messy.loc[rng.random(n) < 0.03, "device"] = np.nan

    # outliers: data-entry errors in age, a few automated accounts with huge call counts
    typo_rows = rng.choice(n, 25, replace=False)
    messy.loc[typo_rows, "age"] = rng.choice([0, 999, 199], 25)
    bot_rows = rng.choice(n, 15, replace=False)
    messy.loc[bot_rows, "num_support_calls"] = rng.integers(40, 80, 15)

    messy["churned"] = target
    return messy


def make_customer_churn(n=N_ROWS, seed=SEED):
    """Return the messy churn table (a DataFrame). Same seed -> same table."""
    rng = np.random.default_rng(seed)
    drivers = make_hidden_drivers(rng, n)
    clean = make_clean_columns(rng, n, drivers)
    target = make_target(rng, churn_score(rng, clean, drivers))
    return make_messy(rng, clean, target)


if __name__ == "__main__":
    out = Path(__file__).with_name("customer_churn.csv")
    table = make_customer_churn()
    table.to_csv(out, index=False)
    print(f"wrote {out}  ({table.shape[0]} rows x {table.shape[1]} columns, churn rate {table['churned'].mean():.2f})")
