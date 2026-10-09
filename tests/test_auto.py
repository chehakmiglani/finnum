
def test_load_stocks_sample_cache_and_fallback(tmp_path, monkeypatch):
    from finnum.data import auto
    df, used = auto.load_stocks("sample", start="2025-06-01")
    assert used == "sample" and len(df) > 0
    # auto: NSE fails -> Yahoo fallback is used and cached
    import finnum.data.nse as nse
    import finnum.data.yahoo as yahoo
    monkeypatch.setattr(nse.NSEClient, "fetch_day", lambda self, d: (_ for _ in ()).throw(RuntimeError("blocked")))
    monkeypatch.setattr(yahoo, "fetch_yahoo", lambda syms, s, e: df)
    cache = tmp_path / "s.csv"
    got, used = auto.load_stocks("auto", cache_csv=str(cache))
    assert used == "yahoo" and cache.exists() and len(got) == len(df)
    again, used = auto.load_stocks("auto", cache_csv=str(cache))
    assert used.startswith("cache:")
