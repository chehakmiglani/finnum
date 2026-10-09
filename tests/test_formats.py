import random

import pytest

from finnum.numbers import STYLES, find_numbers, format_number, parse_number
from finnum.numbers.formats import group_indian, group_international


@pytest.mark.parametrize("digits,intl,indian", [
    ("7", "7", "7"),
    ("123", "123", "123"),
    ("1234", "1,234", "1,234"),
    ("123456", "123,456", "1,23,456"),
    ("1234567", "1,234,567", "12,34,567"),
    ("12345678", "12,345,678", "1,23,45,678"),
    ("1234567890", "1,234,567,890", "1,23,45,67,890"),
])
def test_grouping(digits, intl, indian):
    assert group_international(digits) == intl
    assert group_indian(digits) == indian


def test_format_styles():
    v = 12345678.9
    assert format_number(v, "plain") == "12345678.90"
    assert format_number(v, "intl") == "12,345,678.90"
    assert format_number(v, "indian") == "1,23,45,678.90"
    assert format_number(v, "scaled_indian", scalable=True) == "1.23 crore"
    assert format_number(v, "scaled_intl", scalable=True) == "12.35 million"
    assert format_number(250000, "scaled_indian", decimals=0, scalable=True) == "2.50 lakh"
    # prices are never scaled
    assert format_number(130500.5, "scaled_indian") == "1,30,500.50"
    assert format_number(-3.456, "plain") == "-3.46"
    assert format_number(5, "plain", decimals=0) == "5"


def test_rounding_is_half_up_not_float_artifact():
    assert format_number(2.675, "plain") == "2.68"  # float would give 2.67


@pytest.mark.parametrize("text,value", [
    ("₹12,34,567", 1234567),
    ("1,234,567.50", 1234567.5),
    ("₹12.35 crore", 12.35e7),
    ("4.5 cr", 4.5e7),
    ("40 lakh", 40e5),
    ("12.35 million", 12.35e6),
    ("12.35M", 12.35e6),
    ("-3.45%", -3.45),
    ("(1,234)", -1234),
    ("Rs. 500", 500),
    (".5", 0.5),
])
def test_parse(text, value):
    assert parse_number(text) == pytest.approx(value)


def test_percent_flag_and_multiple():
    nums = find_numbers("TCS rose 2.5% to ₹3,456.70 on volume of 1.2 lakh")
    assert [n.value for n in nums] == pytest.approx([2.5, 3456.70, 1.2e5])
    assert nums[0].is_percent and nums[1].currency == "₹" and nums[2].unit == "lakh"


def test_no_match_inside_words():
    assert find_numbers("Q3 results for FY25") == []


def test_roundtrip_random():
    rng = random.Random(0)
    for _ in range(2000):
        v = round(rng.uniform(-1e9, 1e9), 2)
        for name in ("plain", "intl", "indian"):
            assert parse_number(format_number(v, name)) == pytest.approx(v)
        for name in ("scaled_indian", "scaled_intl"):
            # scaled output keeps 2 decimals of the scaled unit -> bounded relative error
            got = parse_number(format_number(v, name, scalable=True))
            assert got == pytest.approx(v, rel=0.01, abs=0.01)
    assert set(STYLES) == {"plain", "intl", "indian", "scaled_indian", "scaled_intl"}
