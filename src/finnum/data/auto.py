"""One call to get daily stock data from the best available source, cached to CSV."""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from .sample import make_sample
from .schema import validate_stocks

DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT",
                   "BHARTIARTL", "MARUTI", "MRF", "PAGEIND", "YESBANK", "SUZLON", "IRFC"]


def load_stocks(source: str = "auto", symbols: list[str] | None = None, start: str = "2023-10-02",
                cache_csv: str | None = None, nse_cache_dir: str = "nse_cache") -> tuple[pd.DataFrame, str]:
    """Return (stocks, source_used).

    source: "auto" (NSE, falling back to Yahoo), "nse", "yahoo", "sample", or a path to a CSV.
    """
    symbols = symbols or DEFAULT_SYMBOLS
    if cache_csv and os.path.exists(cache_csv):
        return validate_stocks(pd.read_csv(cache_csv)), f"cache:{cache_csv}"
    if source == "sample":
        print("WARNING: synthetic data - pipeline check only, do not report these numbers")
        return make_sample(start=start), "sample"
    if source not in ("auto", "nse", "yahoo"):
        return validate_stocks(pd.read_csv(source)), source

    stocks, used = None, None
    if source in ("auto", "nse"):
        from .nse import NSEClient
        client = NSEClient(cache_dir=nse_cache_dir, retries=2)
        try:
            probe = next(d for d in pd.bdate_range(end=date.today(), periods=10)[::-1].date
                         if client.fetch_day(d) is not None)
            print(f"NSE reachable (probe {probe}) - downloading, this takes a while")
            stocks, used = client.fetch_range(date.fromisoformat(start), date.today(), symbols), "nse"
        except Exception as e:  # noqa: BLE001 - any network failure means "use the fallback"
            if source == "nse":
                raise
            print(f"NSE not reachable ({type(e).__name__}) - using Yahoo Finance")
    if stocks is None:
        from .yahoo import fetch_yahoo
        stocks, used = fetch_yahoo(symbols, start, str(date.today())), "yahoo"
    if cache_csv:
        stocks.to_csv(cache_csv, index=False)
    return stocks, used
