"""Render and parse financial numbers in the styles LLMs actually meet.

Why this module exists
----------------------
The same value can reach a model as very different strings:

    12345678.9  ->  "12345678.90"       (plain)
                ->  "12,345,678.90"     (international grouping)
                ->  "1,23,45,678.90"    (Indian lakh/crore grouping)
                ->  "1.23 crore"        (Indian scale words)
                ->  "12.35 million"     (international scale words)

Each string tokenizes differently, so each is a separate experimental
condition. Rendering uses Decimal so the string is exact, never a float artefact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

# ---------------------------------------------------------------- grouping


def group_international(int_digits: str) -> str:
    """'12345678' -> '12,345,678' (groups of 3 from the right)."""
    out = []
    while len(int_digits) > 3:
        out.append(int_digits[-3:])
        int_digits = int_digits[:-3]
    out.append(int_digits)
    return ",".join(reversed(out))


def group_indian(int_digits: str) -> str:
    """'12345678' -> '1,23,45,678' (last 3 digits, then groups of 2)."""
    if len(int_digits) <= 3:
        return int_digits
    head, tail = int_digits[:-3], int_digits[-3:]
    out = []
    while len(head) > 2:
        out.append(head[-2:])
        head = head[:-2]
    if head:
        out.append(head)
    return ",".join(reversed(out)) + "," + tail


# ---------------------------------------------------------------- styles

INDIAN_SCALES = [(Decimal(10) ** 7, "crore"), (Decimal(10) ** 5, "lakh")]
INTL_SCALES = [(Decimal(10) ** 9, "billion"), (Decimal(10) ** 6, "million")]


@dataclass(frozen=True)
class NumberStyle:
    """How a number is written.

    grouping: digit separators for the integer part ("none" | "intl" | "indian")
    scale:    scale words for large values ("none" | "indian" | "intl")
    """

    name: str
    grouping: str = "none"
    scale: str = "none"
    scale_decimals: int = 2


STYLES: dict[str, NumberStyle] = {
    "plain": NumberStyle("plain"),
    "intl": NumberStyle("intl", grouping="intl"),
    "indian": NumberStyle("indian", grouping="indian"),
    "scaled_indian": NumberStyle("scaled_indian", grouping="indian", scale="indian"),
    "scaled_intl": NumberStyle("scaled_intl", grouping="intl", scale="intl"),
}


def _quantize(value: Decimal, decimals: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def _group(int_digits: str, grouping: str) -> str:
    if grouping == "intl":
        return group_international(int_digits)
    if grouping == "indian":
        return group_indian(int_digits)
    if grouping == "none":
        return int_digits
    raise ValueError(f"unknown grouping {grouping!r}")


def format_number(value: float | int | str | Decimal, style: NumberStyle | str = "plain",
                  decimals: int = 2, scalable: bool = False) -> str:
    """Render ``value`` in ``style``.

    decimals: digits after the point for unscaled output (0 for counts).
    scalable: only columns like volume/turnover may be converted to
              lakh/crore/million words; prices never are.
    """
    if isinstance(style, str):
        style = STYLES[style]
    d = Decimal(str(value))
    sign = "-" if d < 0 else ""
    d = abs(d)

    suffix = ""
    if scalable and style.scale != "none":
        table = INDIAN_SCALES if style.scale == "indian" else INTL_SCALES
        for size, word in table:
            if d >= size:
                d = d / size
                decimals = style.scale_decimals
                suffix = " " + word
                break

    d = _quantize(d, decimals)
    text = f"{d:f}"
    int_part, _, frac = text.partition(".")
    out = _group(int_part, style.grouping)
    if frac:
        out += "." + frac
    return sign + out + suffix


# ---------------------------------------------------------------- parsing

_SUFFIX_MULT = {
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "thousand": 1e3, "k": 1e3,
}

NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<open>\()?"
    r"(?P<sign>[-+−])?\s*"
    r"(?P<cur>₹|Rs\.?|INR)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*(?P<suffix>%|crores?|cr\b|lakhs?|lacs?|L\b|million|mn\b|M\b|billion|bn\b|B\b|thousand|[Kk]\b))?"
    r"(?P<close>\))?",
)


@dataclass(frozen=True)
class ParsedNumber:
    value: float
    text: str
    start: int
    end: int
    is_percent: bool = False
    currency: str | None = None
    unit: str | None = None


def find_numbers(text: str) -> list[ParsedNumber]:
    """Find every number in ``text`` and return its numeric value.

    Handles ₹/Rs/INR, Indian or international commas, %, accounting
    negatives "(1,234)", and scale words (crore, lakh, million, ...).
    """
    found = []
    for m in NUMBER_RE.finditer(text):
        value = float(m.group("num").replace(",", ""))
        suffix = m.group("suffix")
        unit = None
        is_pct = False
        if suffix == "%":
            is_pct = True
        elif suffix:
            unit = suffix.lower()
            value *= _SUFFIX_MULT[unit]
        negative = m.group("sign") in ("-", "−") or (m.group("open") and m.group("close"))
        if negative:
            value = -value
        found.append(ParsedNumber(value=value, text=m.group(0).strip(), start=m.start(), end=m.end(),
                                  is_percent=is_pct, currency=m.group("cur"), unit=unit))
    return found


def parse_number(text: str) -> float | None:
    """Value of the single (first) number in ``text``, or None."""
    nums = find_numbers(text)
    return nums[0].value if nums else None
