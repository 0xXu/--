from pathlib import Path

import pandas as pd
import pytest

from stock_forecast.data import load_time_series, validate_forecast_dates


def test_load_time_series_rejects_missing_calendar_dates(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    pd.DataFrame({"price": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-03"])).to_csv(path)

    with pytest.raises(ValueError, match="equally spaced"):
        load_time_series(path, target_column="price", require_target=True)


def test_validate_forecast_dates_requires_test_to_follow_training() -> None:
    train = pd.DataFrame({"price": [1.0]}, index=pd.date_range("2020-01-01", periods=1, freq="D"))
    test = pd.DataFrame(index=pd.date_range("2020-01-03", periods=1, freq="D"))

    with pytest.raises(ValueError, match="immediately follow"):
        validate_forecast_dates(train, test)
