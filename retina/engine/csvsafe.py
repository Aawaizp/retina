"""Safe CSV cells: a text cell that starts with = + - @ (or a tab / carriage return) is run as a
FORMULA by Excel and similar programs ("CSV injection"). Uploaded product names reach RETINA's
output files, so every text cell we write starts with a ' when it would otherwise be a formula.
Numbers are never touched (a negative number stays a number)."""

import pandas as pd

RISKY = ("=", "+", "-", "@", "\t", "\r")


def neutralise_text(value):
    if isinstance(value, str) and value.startswith(RISKY):
        return "'" + value
    return value


def neutralise(df):
    """A copy of df with risky text cells neutralised (only text columns are looked at)."""
    text = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    if not text:
        return df
    out = df.copy()
    for c in text:
        out[c] = out[c].map(neutralise_text)
    return out


def to_csv(df, path, **kwargs):
    """df.to_csv with risky cells neutralised."""
    neutralise(df).to_csv(path, **kwargs)
