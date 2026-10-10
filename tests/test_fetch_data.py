import numpy as np
import pandas as pd

from src.fetch_data import fetch_data


def make_fake_download(counter):
    """Stand-in for yf.download that returns a yfinance-style MultiIndex frame."""
    def fake_download(*args, **kwargs):
        counter["n"] += 1
        idx = pd.date_range("2024-01-01", periods=3, name="Date")
        cols = pd.MultiIndex.from_product(
            [["Adj Close", "Close"], ["TEST"]], names=["Price", "Ticker"]
        )
        return pd.DataFrame(np.arange(6.0).reshape(3, 2), index=idx, columns=cols)
    return fake_download


def test_second_call_uses_cache(tmp_path, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr("src.fetch_data.yf.download", make_fake_download(calls))

    first = fetch_data("TEST", "2024-01-01", "2024-01-04", data_dir=str(tmp_path))
    second = fetch_data("TEST", "2024-01-01", "2024-01-04", data_dir=str(tmp_path))

    assert calls["n"] == 1  # network hit only once
    assert np.allclose(first.values, second.values)
    assert list(second.columns) == ["Adj Close", "Close"]
    assert isinstance(second.index, pd.DatetimeIndex)


def test_use_cache_false_forces_redownload(tmp_path, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr("src.fetch_data.yf.download", make_fake_download(calls))

    fetch_data("TEST", "2024-01-01", "2024-01-04", data_dir=str(tmp_path))
    fetch_data("TEST", "2024-01-01", "2024-01-04", use_cache=False, data_dir=str(tmp_path))

    assert calls["n"] == 2


def test_empty_download_returns_none_and_writes_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.fetch_data.yf.download", lambda *a, **k: pd.DataFrame())

    assert fetch_data("BAD", "2024-01-01", "2024-01-04", data_dir=str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []