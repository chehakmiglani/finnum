"""Numeric feature vectors for each number encoding method (numpy only, no torch).

Each method turns a parsed number into a fixed-length float vector; a small
trainable network (encoders.py) maps that vector into the LLM's embedding space.

  xval        1 dim   signed log-magnitude; the <num> embedding is SCALED by it (xVal, Golkar et al. 2023).
                      We use sign*log10(1+|x|)/12 instead of raw x because market values span
                      ~10^-2 .. 10^12 (documented deviation from the paper).
  fone        29 dims cos/sin of x on periods 10^-1 .. 10^12 (one pair per digit position, FoNE,
                      Zhou et al. ICLR 2026) + sign. Digit-exact: each pair encodes one digit.
  bits        64 dims IEEE-754 float64 bit pattern as +-1 (BitTokens, Kreitner et al. 2025)
"""
from __future__ import annotations

import numpy as np

from finnum.numbers.formats import ParsedNumber

FOURIER_PERIODS = np.array([10.0 ** k for k in range(-1, 13)])  # 0.1 .. 10^12 -> hundredths .. 10^12 digit
def xval_features(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return (np.sign(v) * np.log10(1 + np.abs(v)) / 12.0)[:, None]


def fone_features(values: np.ndarray) -> np.ndarray:
    v = np.abs(np.asarray(values, dtype=np.float64))[:, None]
    # reduce modulo each period FIRST, so the angle keeps full precision for huge numbers
    phase = 2 * np.pi * np.mod(v, FOURIER_PERIODS[None, :]) / FOURIER_PERIODS[None, :]
    sign = np.sign(np.asarray(values, dtype=np.float64))[:, None]
    return np.concatenate([np.cos(phase), np.sin(phase), sign], axis=1)


def bit_features(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=">f8")
    bits = np.unpackbits(v.view(np.uint8).reshape(-1, 8), axis=1)
    return bits.astype(np.float64) * 2 - 1


FEATURE_DIMS = {"xval": 1, "fone": 2 * len(FOURIER_PERIODS) + 1, "bits": 64}


def features(nums: list[ParsedNumber], kind: str) -> np.ndarray:
    values = np.array([n.value for n in nums], dtype=np.float64)
    if kind == "xval":
        out = xval_features(values)
    elif kind == "fone":
        out = fone_features(values)
    elif kind == "bits":
        out = bit_features(values)
    else:
        raise ValueError(f"unknown feature kind {kind!r}")
    out = out.reshape(len(nums), FEATURE_DIMS[kind])
    return out.astype(np.float32)


def decode_fone(feat: np.ndarray) -> float:
    """Invert fone features back to a number (used in tests to prove they are digit-exact).

    Digits are read from the finest period upward; each step subtracts the value of the
    digits already decoded, which makes the decoding robust to float wrap-around (x mod P ~ P).
    """
    n = len(FOURIER_PERIODS)
    cos, sin, sign = feat[:n], feat[n:2 * n], feat[-1]
    frac = np.mod(np.arctan2(sin, cos) / (2 * np.pi), 1.0)  # (x mod P) / P
    lower = 0.0
    for i, p in enumerate(FOURIER_PERIODS):
        d = int(round((frac[i] * p - lower) / (p / 10))) % 10
        lower += d * p / 10
    return float((-1.0 if sign < 0 else 1.0) * lower)
