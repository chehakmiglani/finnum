"""FinNum question generator.

Every *base question* is generated once from the data with an exact gold
answer, then rendered in every (number style x table format) combination.
Because the underlying question is identical across variants, accuracy
differences between styles are a clean, paired comparison.

Tasks
-----
lookup      read one value                         -> number
pct_change  % change between two dates             -> number (2 dp)
mean        average close over a window            -> number (2 dp)
compare     which of two stocks moved more         -> choice
magnitude   which stock had the highest turnover   -> choice (tests lakh/crore/million units)
anomaly     is today's volume > 3x 20-day average  -> yes/no
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from finnum.numbers import STYLES

from .render import TABLE_FORMATS, render_table

INSTRUCTIONS = {
    "number": "Answer with only the number.",
    "choice": "Answer with only the stock symbol.",
    "bool": "Answer with only yes or no.",
}


@dataclass
class BaseQuestion:
    id: str
    task: str
    question: str
    answer_type: str            # number | choice | bool
    gold: float | str | bool
    tolerance: float = 0.0      # absolute tolerance for numbers
    options: list[str] = field(default_factory=list)
    table: pd.DataFrame | None = None
    cols: list[str] = field(default_factory=list)


@dataclass
class Variant:
    id: str
    base_id: str
    task: str
    style: str
    table_format: str
    prompt: str
    answer_type: str
    gold: float | str | bool
    tolerance: float
    options: list[str]
    n_rows: int
    differs_from_plain: bool = True  # False when this style renders identically to "plain" (e.g. small numbers)


class Generator:
    def __init__(self, df: pd.DataFrame, seed: int = 0, min_history: int = 25):
        self.rng = random.Random(seed)
        self.by_symbol = {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}
        self.symbols = sorted(self.by_symbol)
        self.by_date = {d: g.reset_index(drop=True) for d, g in df.groupby("date")}
        self.min_history = min_history
        self._n = 0

    # ---------------------------------------------------------------- helpers
    def _id(self, task: str) -> str:
        self._n += 1
        return f"{task}-{self._n:06d}"

    def _window(self, k: int) -> tuple[str, pd.DataFrame]:
        sym = self.rng.choice(self.symbols)
        g = self.by_symbol[sym]
        end = self.rng.randrange(max(k, self.min_history), len(g))
        return sym, g.iloc[end - k:end].reset_index(drop=True)

    @staticmethod
    def _d(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d")

    # ---------------------------------------------------------------- tasks
    def lookup(self) -> BaseQuestion:
        sym, w = self._window(10)
        row = w.iloc[self.rng.randrange(len(w))]
        return BaseQuestion(self._id("lookup"), "lookup",
                            f"What was the closing price of {sym} on {self._d(row['date'])}?",
                            "number", round(float(row["close"]), 2), 0.005,
                            table=w, cols=["date", "close", "volume"])

    def pct_change(self) -> BaseQuestion:
        sym, w = self._window(10)
        i, j = sorted(self.rng.sample(range(len(w)), 2))
        c1, c2 = float(w.loc[i, "close"]), float(w.loc[j, "close"])
        gold = round((c2 / c1 - 1) * 100, 2)
        return BaseQuestion(self._id("pct_change"), "pct_change",
                            f"By what percentage did the closing price of {sym} change from "
                            f"{self._d(w.loc[i, 'date'])} to {self._d(w.loc[j, 'date'])}? "
                            "Round to 2 decimal places.",
                            "number", gold, 0.011, table=w, cols=["date", "open", "high", "low", "close"])

    def mean(self) -> BaseQuestion:
        k = self.rng.choice([5, 7, 10])
        sym, w = self._window(k)
        gold = round(float(w["close"].mean()), 2)
        return BaseQuestion(self._id("mean"), "mean",
                            f"What was the average closing price of {sym} over these {k} trading days? "
                            "Round to 2 decimal places.",
                            "number", gold, 0.011, table=w, cols=["date", "close"])

    def compare(self) -> BaseQuestion:
        for _ in range(100):
            date = self.rng.choice(list(self.by_date))
            day = self.by_date[date]
            if len(day) < 2:
                continue
            a, b = day.sample(2, random_state=self.rng.randrange(2**31)).itertuples()
            pa = a.close / a.prev_close - 1
            pb = b.close / b.prev_close - 1
            if abs(pa - pb) < 0.001:  # avoid near-ties
                continue
            t = day[day["symbol"].isin([a.symbol, b.symbol])]
            return BaseQuestion(self._id("compare"), "compare",
                                f"On {self._d(date)}, which stock had the larger percentage change from its "
                                f"previous close: {a.symbol} or {b.symbol}? (Larger means more positive.)",
                                "choice", a.symbol if pa > pb else b.symbol,
                                options=[a.symbol, b.symbol], table=t,
                                cols=["symbol", "prev_close", "close", "volume"])
        raise RuntimeError("could not build compare question")

    def magnitude(self) -> BaseQuestion:
        for _ in range(100):
            date = self.rng.choice(list(self.by_date))
            day = self.by_date[date]
            if len(day) < 5:
                continue
            t = day.sample(5, random_state=self.rng.randrange(2**31)).reset_index(drop=True)
            top2 = t["turnover"].nlargest(2).to_numpy()
            if top2[0] < top2[1] * 1.05:  # need a clear winner even after rounding
                continue
            sym = t.loc[t["turnover"].idxmax(), "symbol"]
            return BaseQuestion(self._id("magnitude"), "magnitude",
                                f"On {self._d(date)}, which of these stocks had the highest turnover in rupees?",
                                "choice", sym, options=list(t["symbol"]), table=t,
                                cols=["symbol", "close", "turnover"])
        raise RuntimeError("could not build magnitude question")

    def anomaly(self, want: bool | None = None) -> BaseQuestion:
        want = self.rng.random() < 0.5 if want is None else want
        for _ in range(5000):
            sym, w = self._window(21)
            ratio = w["volume"].iloc[-1] / w["volume"].iloc[:-1].mean()
            is_spike = ratio > 3
            if is_spike != want or 2.7 < ratio < 3.3:  # skip ambiguous ratios near the threshold
                continue
            last = self._d(w["date"].iloc[-1])
            return BaseQuestion(self._id("anomaly"), "anomaly",
                                f"Was the volume of {sym} on {last} more than 3 times the average volume "
                                "of the previous 20 trading days shown?",
                                "bool", bool(is_spike), table=w, cols=["date", "close", "volume"])
        raise RuntimeError("could not build anomaly question with the requested label")

    TASKS = ("lookup", "pct_change", "mean", "compare", "magnitude", "anomaly")

    # ---------------------------------------------------------------- rendering
    @staticmethod
    def render(q: BaseQuestion, style: str, fmt: str) -> Variant:
        table = render_table(q.table, q.cols, style, fmt)
        prompt = (f"Here is market data for NSE-listed stocks (prices in INR).\n\n{table}\n\n"
                  f"Question: {q.question}\n{INSTRUCTIONS[q.answer_type]}")
        return Variant(id=f"{q.id}|{style}|{fmt}", base_id=q.id, task=q.task, style=style, table_format=fmt,
                       prompt=prompt, answer_type=q.answer_type, gold=q.gold, tolerance=q.tolerance,
                       options=q.options, n_rows=len(q.table))

    def build(self, n_per_task: int, styles=None, formats=None, tasks=None) -> list[Variant]:
        styles = list(styles or STYLES)
        formats = list(formats or TABLE_FORMATS)
        out = []
        for task in tasks or self.TASKS:
            for _ in range(n_per_task):
                q = getattr(self, task)()
                vs = [self.render(q, s, f) for s in styles for f in formats]
                plain = {v.table_format: v.prompt for v in vs if v.style == "plain"}
                for v in vs:
                    v.differs_from_plain = v.style == "plain" or v.prompt != plain.get(v.table_format)
                out.extend(vs)
        return out


def write_jsonl(variants: list[Variant], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for v in variants:
            fh.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]
