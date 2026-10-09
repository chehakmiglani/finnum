"""Synthetic NSE-like data for offline development and tests.

This is NOT real market data. It mimics the shape of the real thing
(price levels, tick size, volume spikes, holidays) so the benchmark code
can be built and tested anywhere. Real experiments use NSE bhavcopies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import validate_stocks

# symbol: (approx price, daily volatility, typical daily volume)
UNIVERSE = {
    "RELIANCE": (2900, 0.015, 6_000_000), "TCS": (4100, 0.013, 2_000_000),
    "INFY": (1800, 0.016, 7_000_000), "HDFCBANK": (1650, 0.013, 15_000_000),
    "ICICIBANK": (1250, 0.014, 14_000_000), "SBIN": (820, 0.017, 16_000_000),
    "ITC": (470, 0.012, 12_000_000), "LT": (3600, 0.015, 2_000_000),
    "BHARTIARTL": (1600, 0.014, 6_000_000), "HINDUNILVR": (2450, 0.011, 1_500_000),
    "TATAMOTORS": (950, 0.022, 18_000_000), "MARUTI": (12400, 0.013, 400_000),
    "MRF": (130000, 0.012, 8_000), "PAGEIND": (42000, 0.014, 15_000),
    "SUZLON": (58, 0.035, 90_000_000), "YESBANK": (22, 0.03, 200_000_000),
    "IRFC": (160, 0.03, 40_000_000), "ADANIENT": (2950, 0.028, 3_000_000),
    "ZOMATO": (240, 0.03, 60_000_000), "BAJFINANCE": (7100, 0.018, 1_200_000),
}

# a few fixed non-trading days so calendars are not just Mon-Fri
HOLIDAYS = ["2024-01-26", "2024-03-25", "2024-08-15", "2024-10-02", "2024-11-01",
            "2025-01-26", "2025-03-14", "2025-08-15", "2025-10-02", "2025-10-21",
            "2026-01-26", "2026-03-03", "2026-08-15", "2026-10-02"]


def _tick(x: np.ndarray) -> np.ndarray:
    return np.round(x * 20) / 20  # NSE 0.05 tick


def make_sample(start: str = "2023-10-02", end: str = "2026-10-06", seed: int = 7,
                spike_prob: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    days = days[~days.isin(pd.to_datetime(HOLIDAYS))]
    n = len(days)
    frames = []
    for i, (sym, (p0, vol, v0)) in enumerate(UNIVERSE.items()):
        rets = rng.normal(0.0003, vol, n)
        close = _tick(p0 * np.exp(np.cumsum(rets)))
        close = np.maximum(close, 0.05)
        prev = np.concatenate([[_tick(np.array([p0]))[0]], close[:-1]])
        open_ = _tick(prev * (1 + rng.normal(0, vol / 3, n)))
        hi = _tick(np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n))))
        lo = _tick(np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n))))
        lo = np.maximum(lo, 0.05)
        volume = rng.lognormal(np.log(v0), 0.35, n)
        spikes = rng.random(n) < spike_prob
        volume[spikes] *= rng.uniform(4, 8, spikes.sum())
        volume = volume.astype("int64")
        vwap = (hi + lo + close) / 3
        turnover = np.round(volume * vwap, 2)
        trades = np.maximum((volume / rng.uniform(80, 400)).astype("int64"), 1)
        frames.append(pd.DataFrame({
            "date": days, "symbol": sym, "isin": f"INE{i:03d}SYNTH0", "series": "EQ",
            "open": open_, "high": hi, "low": lo, "close": close, "prev_close": prev,
            "volume": volume, "turnover": turnover, "trades": trades,
        }))
    return validate_stocks(pd.concat(frames, ignore_index=True))
