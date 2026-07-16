"""Application orchestration for stock-price forecasting."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import PathLike, load_time_series, training_target
from .model import ArimaOrder, forecast_arima
from .plotting import save_forecast_plot

DEFAULT_TRAIN_URL = "https://www.dropbox.com/scl/fi/1gzfa2otxd3vqruw4n98e/train.csv?rlkey=2e1kobduti6fqh4bu7r0srpo6&st=fhqqu5xz&dl=1"
DEFAULT_TEST_URL = "https://www.dropbox.com/scl/fi/k8gvgdwfmvang7ac9z3p4/test.csv?rlkey=zxpgjhlp6o7glnkek89nrx0vd&st=8eebzl17&dl=1"


@dataclass(frozen=True)
class ForecastConfig:
    """Inputs and model settings for a single forecast run."""

    train_path: PathLike = DEFAULT_TRAIN_URL
    test_path: PathLike = DEFAULT_TEST_URL
    output_path: PathLike = "submission.csv"
    target_column: str = "price"
    arima_order: ArimaOrder = (7, 1, 2)
    plot_path: PathLike | None = None


def run_forecast(config: ForecastConfig) -> pd.DataFrame:
    """Create and persist a submission DataFrame indexed like the test dataset."""
    train_data = load_time_series(config.train_path, target_column=config.target_column, require_target=True)
    test_data = load_time_series(config.test_path, target_column=config.target_column, require_target=False)
    target = training_target(train_data, config.target_column)

    predictions = forecast_arima(target, len(test_data), config.arima_order)
    submission = pd.DataFrame({config.target_column: predictions.to_numpy()}, index=test_data.index)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path)

    if config.plot_path is not None:
        save_forecast_plot(target, submission[config.target_column], config.plot_path)
    return submission
