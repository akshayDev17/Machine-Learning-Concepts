"""Stage 07 · build features from what you know about the business: ratios, differences, group statistics,
keyword flags from free text.

Examples here: monthly price relative to income (can they afford it?), active lifetime (days between sign-up and
last login), income relative to the median income of the customer's city, "does the support note mention billing?".
Group statistics (the city medians) are learned on the training rows only and reused on test rows.
A feature whose input column is missing or not numeric (because a stage that creates or fixes it was switched off)
is skipped and reported.
"""
import numpy as np

from column_kinds import numeric_columns


def fit_transform(train, target, params):
    """Learn group statistics on the training rows, then build all domain features. Return (train, state)."""
    group_values = {}
    for spec in params["group_stats"]:
        if spec["column"] in numeric_columns(train, target) and spec["by"] in train.columns:
            group_values[spec["name"]] = {
                "per_group": train.groupby(spec["by"])[spec["column"]].agg(spec["stat"]).to_dict(),
                "overall": train[spec["column"]].agg(spec["stat"]),
            }
    state = {**params, "group_values": group_values}
    out = transform(train, state)
    made = [c for c in out.columns if c not in train.columns]
    wanted = ([s["name"] for s in params["ratios"] + params["differences"] + params["group_stats"]]
              + [f"note_mentions_{w}" for w in params["keyword_flags"]["words"]])
    skipped = [w for w in wanted if w not in made]
    state["summary"] = f"made {made}" + (f"; skipped (input column missing or not numeric): {skipped}" if skipped else "")
    return out, state


def transform(frame, state):
    """Add ratio, difference, group-statistic and keyword-flag columns."""
    frame = frame.copy()
    numeric = set(numeric_columns(frame, None))
    for spec in state["ratios"]:
        if {spec["numerator"], spec["denominator"]} <= numeric:
            ratio = frame[spec["numerator"]] / frame[spec["denominator"]]
            frame[spec["name"]] = ratio.replace([np.inf, -np.inf], np.nan)
    for spec in state["differences"]:
        if {spec["left"], spec["right"]} <= numeric:
            frame[spec["name"]] = frame[spec["left"]] - frame[spec["right"]]
    for spec in state["group_stats"]:
        if spec["name"] in state["group_values"]:
            learned = state["group_values"][spec["name"]]
            group_stat = frame[spec["by"]].map(learned["per_group"]).fillna(learned["overall"])
            frame[spec["name"]] = frame[spec["column"]] / group_stat
    text_column = state["keyword_flags"]["column"]
    if text_column in frame.columns:
        text = frame[text_column].fillna("").str.lower()
        for word in state["keyword_flags"]["words"]:
            frame[f"note_mentions_{word}"] = text.str.contains(word, regex=False).astype(int)
        if state["keyword_flags"]["drop_text_column"]:
            frame = frame.drop(columns=text_column)
    return frame
