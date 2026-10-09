"""Mutual fund NAVs: AMFI daily file + mfapi.in history. Run on your own machine."""
from __future__ import annotations

import pandas as pd
import requests

NAVALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_URL = "https://api.mfapi.in/mf/{code}"


def parse_navall(text: str) -> pd.DataFrame:
    """Parse AMFI NAVAll.txt.

    The file mixes data rows (6 ';'-separated fields) with section headers:
    a category line like "Open Ended Schemes(Equity Scheme - Large Cap Fund)"
    followed by an AMC name line, then the AMC's schemes.
    """
    rows, category, amc = [], None, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("Scheme Code"):
            continue
        parts = line.split(";")
        if len(parts) == 6:
            code, isin_g, isin_r, name, nav, dt = (p.strip() for p in parts)
            try:
                nav_val = float(nav)
            except ValueError:  # "N.A." etc.
                continue
            rows.append({"scheme_code": int(code), "isin_growth": isin_g or None,
                         "isin_reinvest": isin_r or None, "scheme_name": name, "nav": nav_val,
                         "date": pd.to_datetime(dt, format="%d-%b-%Y"), "amc": amc, "category": category})
        elif "Schemes(" in line or line.endswith("Schemes"):
            category, amc = line, None
        else:
            amc = line
    return pd.DataFrame(rows)


def fetch_navall(session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    r = s.get(NAVALL_URL, timeout=60)
    r.raise_for_status()
    return parse_navall(r.text)


def parse_mfapi(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = df["nav"].astype(float)
    df["scheme_code"] = int(payload["meta"]["scheme_code"])
    df["scheme_name"] = payload["meta"]["scheme_name"]
    return df.sort_values("date").reset_index(drop=True)


def fetch_history(code: int, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    r = s.get(MFAPI_URL.format(code=code), timeout=60)
    r.raise_for_status()
    return parse_mfapi(r.json())
