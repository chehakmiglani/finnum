"""Fallback: daily NSE prices from Yahoo Finance (symbol + ".NS").

Use when NSE blocks your network (common on cloud machines like Colab).
Prices are raw traded prices (auto_adjust=False). Yahoo has no turnover or
trade count, so turnover is approximated as volume x (high+low+close)/3 and
trades is set to 0. Say so in the paper if you use this source.
"""
from __future__ import annotations

import pandas as pd

from .schema import validate_stocks


def normalize_yahoo(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # yfinance >= 0.2.40 returns (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date", "index": "date"})
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[df["volume"] > 0]
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].round(2)
    # Yahoo occasionally reports a high/low that is inconsistent with open/close; repair it
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["prev_close"] = df["close"].shift(1)
    df = df.dropna(subset=["prev_close"])
    df["turnover"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).round(2)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.assign(symbol=symbol, isin="", series="EQ", trades=0)
    df["volume"] = df["volume"].astype("int64")
    return validate_stocks(df)


def fetch_yahoo(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for s in symbols:
        raw = yf.download(f"{s}.NS", start=start, end=end, auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            print(f"[skip] {s}: no data from Yahoo")
            continue
        frames.append(normalize_yahoo(raw, s))
        print(f"{s}: {len(frames[-1])} rows")
    if not frames:
        raise RuntimeError("Yahoo returned no data")
    return validate_stocks(pd.concat(frames, ignore_index=True))
