from pathlib import Path

import pandas as pd

from stock_forecast.data import load_time_series


def test_remote_data_is_reused_from_cache(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_retrieve(url: str, destination: Path) -> None:
        calls.append(url)
        pd.DataFrame({"price": [1.0]}, index=pd.date_range("2020-01-01", periods=1)).to_csv(destination)

    monkeypatch.setattr("stock_forecast.data.urlretrieve", fake_retrieve)
    url = "https://example.com/train.csv"

    load_time_series(url, target_column="price", require_target=True, cache_dir=tmp_path)
    load_time_series(url, target_column="price", require_target=True, cache_dir=tmp_path)

    assert calls == [url]
