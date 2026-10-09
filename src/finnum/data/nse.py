"""NSE end-of-day equity bhavcopy downloader.

Run this on your own machine: NSE often blocks cloud/datacentre IPs.

Two archive formats exist:
  * UDiFF (from 2024-07-08):  BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
  * Legacy (before that):     cmDDMONYYYYbhav.csv.zip
NSE changes URLs occasionally -- if downloads 404 on known trading days,
check the "All Reports" page on nseindia.com and update the templates below.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .schema import validate_stocks

UDIFF_START = date(2024, 7, 8)
UDIFF_URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
LEGACY_URL = "https://nsearchives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mon}/cm{dd}{mon}{yyyy}bhav.csv.zip"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

UDIFF_MAP = {
    "TradDt": "date", "TckrSymb": "symbol", "ISIN": "isin", "SctySrs": "series",
    "OpnPric": "open", "HghPric": "high", "LwPric": "low", "ClsPric": "close",
    "PrvsClsgPric": "prev_close", "TtlTradgVol": "volume", "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
}
LEGACY_MAP = {
    "TIMESTAMP": "date", "SYMBOL": "symbol", "ISIN": "isin", "SERIES": "series",
    "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
    "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume", "TOTTRDVAL": "turnover",
    "TOTALTRADES": "trades",
}


def bhavcopy_url(d: date) -> str:
    if d >= UDIFF_START:
        return UDIFF_URL.format(ymd=d.strftime("%Y%m%d"))
    mon = d.strftime("%b").upper()
    return LEGACY_URL.format(yyyy=d.year, mon=mon, dd=d.strftime("%d"))


def normalize_bhavcopy(raw: pd.DataFrame, series: tuple[str, ...] = ("EQ",)) -> pd.DataFrame:
    """Map either bhavcopy format onto the canonical schema."""
    raw = raw.rename(columns=lambda c: c.strip())
    mapping = UDIFF_MAP if "TckrSymb" in raw.columns else LEGACY_MAP
    df = raw[list(mapping)].rename(columns=mapping)
    for c in ("symbol", "series", "isin"):
        df[c] = df[c].astype(str).str.strip()
    df = df[df["series"].isin(series)]
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=mapping is LEGACY_MAP)
    return validate_stocks(df)


class NSEClient:
    def __init__(self, cache_dir: str | Path = "data/raw/nse", pause: float = 1.0, retries: int = 3):
        self.cache_dir = Path(cache_dir)
        self.pause = pause
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._warmed = False

    def _warm(self) -> None:
        # NSE sets cookies on the home page; archive requests work more reliably afterwards.
        if not self._warmed:
            try:
                self.session.get("https://www.nseindia.com/", timeout=15)
            except requests.RequestException:
                pass
            self._warmed = True

    def fetch_raw(self, d: date) -> bytes | None:
        """Zip bytes for one trading day, or None for holidays/missing files. Cached on disk."""
        path = self.cache_dir / f"{d:%Y}" / f"{d:%Y%m%d}.csv.zip"
        if path.exists():
            return path.read_bytes()
        self._warm()
        url = bhavcopy_url(d)
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
                time.sleep(self.pause)
                return r.content
            except requests.RequestException:
                time.sleep(self.pause * 2 ** attempt)
        raise RuntimeError(f"failed to download {url} after {self.retries} attempts")

    def fetch_day(self, d: date) -> pd.DataFrame | None:
        blob = self.fetch_raw(d)
        if blob is None:
            return None
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            raw = pd.read_csv(zf.open(name))
        return normalize_bhavcopy(raw)

    def fetch_range(self, start: date, end: date, symbols: list[str] | None = None) -> pd.DataFrame:
        frames = []
        for d in pd.bdate_range(start, end).date:
            day = self.fetch_day(d)
            if day is None:
                continue
            if symbols:
                day = day[day["symbol"].isin(symbols)]
            frames.append(day)
            print(f"{d}: {len(day)} rows")
        if not frames:
            raise RuntimeError("no data downloaded -- check network access to NSE")
        return validate_stocks(pd.concat(frames, ignore_index=True))
