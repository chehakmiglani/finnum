import io

import pandas as pd
import pytest

from finnum.data import make_sample, validate_stocks
from finnum.data.amfi import parse_mfapi, parse_navall
from finnum.data.nse import bhavcopy_url, normalize_bhavcopy

UDIFF_CSV = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2026-10-06,2026-10-06,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE INDUSTRIES LTD,2890.00,2925.50,2880.10,2915.35,2916.00,2889.40,,2915.35,,,5123456,14912345678.55,198765,F1,1,,,,,
2026-10-06,2026-10-06,CM,NSE,STK,9999,INE000X00000,SOMEBOND,N1,,,,,SOME BOND,100,101,99,100.5,100.5,100,,100.5,,,10,1005,2,F1,1,,,,,
"""

LEGACY_CSV = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,
TCS,EQ,3500,3550,3480,3540.25,3541,3495.6,1234567,4370000000.5,05-FEB-2024,85432,INE467B01029,
"""

NAVALL = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Large Cap Fund)

Axis Mutual Fund

120465;INF846K01EW2;-;Axis Bluechip Fund - Direct Plan - Growth;62.4500;06-Oct-2026
120466;INF846K01EX0;INF846K01EY8;Axis Bluechip Fund - Direct Plan - IDCW;N.A.;06-Oct-2026
"""


def test_normalize_udiff_filters_series():
    df = normalize_bhavcopy(pd.read_csv(io.StringIO(UDIFF_CSV)))
    assert list(df["symbol"]) == ["RELIANCE"]
    row = df.iloc[0]
    assert row["close"] == 2915.35 and row["volume"] == 5123456 and row["prev_close"] == 2889.40
    assert row["date"] == pd.Timestamp("2026-10-06")


def test_normalize_legacy():
    df = normalize_bhavcopy(pd.read_csv(io.StringIO(LEGACY_CSV)))
    assert df.iloc[0]["symbol"] == "TCS" and df.iloc[0]["date"] == pd.Timestamp("2024-02-05")


def test_bhavcopy_url_switches_format():
    from datetime import date
    assert "BhavCopy_NSE_CM" in bhavcopy_url(date(2026, 10, 6))
    assert bhavcopy_url(date(2024, 2, 5)).endswith("cm05FEB2024bhav.csv.zip")


def test_parse_navall():
    df = parse_navall(NAVALL)
    assert len(df) == 1  # N.A. row skipped
    r = df.iloc[0]
    assert r["scheme_code"] == 120465 and r["nav"] == 62.45
    assert r["amc"] == "Axis Mutual Fund" and "Large Cap" in r["category"]


def test_parse_mfapi():
    df = parse_mfapi({"meta": {"scheme_code": 120465, "scheme_name": "X"},
                      "data": [{"date": "06-10-2026", "nav": "62.45"}, {"date": "03-10-2026", "nav": "61.9"}]})
    assert list(df["nav"]) == [61.9, 62.45]


def test_sample_is_valid_and_deterministic():
    a = make_sample(start="2025-01-01", end="2025-03-31")
    b = make_sample(start="2025-01-01", end="2025-03-31")
    pd.testing.assert_frame_equal(a, b)
    assert a["symbol"].nunique() == 20
    assert pd.Timestamp("2025-03-14") not in set(a["date"])  # holiday removed


def test_validate_rejects_duplicates():
    df = make_sample(start="2025-01-01", end="2025-01-10")
    with pytest.raises(ValueError):
        validate_stocks(pd.concat([df, df.head(1)]))
