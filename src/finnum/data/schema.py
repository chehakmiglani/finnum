"""Canonical daily stock schema shared by every loader."""
import pandas as pd

STOCK_COLUMNS = ["date", "symbol", "isin", "series", "open", "high", "low", "close",
                 "prev_close", "volume", "turnover", "trades"]


def validate_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast and sanity-check a daily stock frame. Raises on bad data."""
    missing = set(STOCK_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df[STOCK_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in ["open", "high", "low", "close", "prev_close", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    for c in ["volume", "trades"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices found")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1) - 1e-9).any():
        raise ValueError("high below open/close/low")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1) + 1e-9).any():
        raise ValueError("low above open/close/high")
    if df.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate (date, symbol) rows")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)
