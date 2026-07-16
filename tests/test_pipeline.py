from pathlib import Path

import pandas as pd

from stock_forecast.pipeline import ForecastConfig, run_forecast


def test_run_forecast_writes_predictions_with_test_index(tmp_path: Path, monkeypatch) -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    train = pd.DataFrame({"price": range(12)}, index=dates)
    test_dates = pd.date_range("2020-01-13", periods=3, freq="D")
    test = pd.DataFrame(index=test_dates)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    output_path = tmp_path / "submission.csv"
    train.to_csv(train_path)
    test.to_csv(test_path)

    monkeypatch.setattr(
        "stock_forecast.pipeline.forecast_chronos",
        lambda target, horizon, config: pd.Series([100.0] * horizon),
    )
    result = run_forecast(ForecastConfig(train_path=train_path, test_path=test_path, output_path=output_path, model="chronos"))

    assert output_path.exists()
    assert result.index.equals(test_dates)
    assert list(result.columns) == ["price"]
    assert len(result) == len(test)
