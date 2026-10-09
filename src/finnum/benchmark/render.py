"""Render a small market table as text in a chosen number style and table format."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from finnum.numbers import format_number


@dataclass(frozen=True)
class ColSpec:
    header: str
    decimals: int | None = 2   # None = not numeric (text / date)
    scalable: bool = False     # may become lakh/crore/million in scaled styles


COLS: dict[str, ColSpec] = {
    "date": ColSpec("date", None),
    "symbol": ColSpec("symbol", None),
    "open": ColSpec("open"),
    "high": ColSpec("high"),
    "low": ColSpec("low"),
    "close": ColSpec("close"),
    "prev_close": ColSpec("prev_close"),
    "volume": ColSpec("volume", 0, scalable=True),
    "turnover": ColSpec("turnover_inr", 2, scalable=True),
}

TABLE_FORMATS = ("csv", "markdown", "json")


def _cell(value, spec: ColSpec, style: str) -> str:
    if spec.decimals is None:
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value)
    return format_number(value, style, decimals=spec.decimals, scalable=spec.scalable)


def render_table(df: pd.DataFrame, cols: list[str], style: str, fmt: str) -> str:
    headers = [COLS[c].header for c in cols]
    rows = [[_cell(r[c], COLS[c], style) for c in cols] for _, r in df.iterrows()]
    if fmt == "csv":
        # commas inside grouped numbers would break CSV, so quote cells that contain them
        def q(s: str) -> str:
            return f'"{s}"' if "," in s else s
        return "\n".join([",".join(headers)] + [",".join(q(c) for c in row) for row in rows])
    if fmt == "markdown":
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)
    if fmt == "json":
        # numbers are kept as the styled strings so the style is what the model sees
        return json.dumps([dict(zip(headers, row)) for row in rows], ensure_ascii=False, separators=(",", ":"))
    raise ValueError(f"unknown table format {fmt!r}")
