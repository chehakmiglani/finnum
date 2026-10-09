import pandas as pd


def test_normalize_yahoo_multiindex_and_repairs():
    import numpy as np
    from finnum.data.yahoo import normalize_yahoo
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-07"]), name="Date")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["TCS.NS"]])
    vals = np.array([[100, 101, 99, 100.5, 100.5, 1000],
                     [100.5, 100.0, 99.5, 101.2, 101.2, 2000],   # high < close: must be repaired
                     [101, 102, 100, 101.9, 101.9, 0]])          # zero volume: dropped
    df = normalize_yahoo(pd.DataFrame(vals, index=idx, columns=cols), "TCS")
    assert len(df) == 1                       # first row has no prev_close, last has no volume
    r = df.iloc[0]
    assert r["high"] == 101.2 and r["prev_close"] == 100.5 and r["symbol"] == "TCS"
