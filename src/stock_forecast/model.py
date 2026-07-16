"""Forecast model abstraction."""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

ArimaOrder = tuple[int, int, int]


def forecast_arima(training_target: pd.Series, horizon: int, order: ArimaOrder) -> pd.Series:
    """Fit an ARIMA model and forecast ``horizon`` future observations."""
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    if len(order) != 3 or any(not isinstance(value, int) or value < 0 for value in order):
        raise ValueError("ARIMA order must be a tuple of three non-negative integers.")

    fitted_model = ARIMA(training_target, order=order).fit()
    predictions = fitted_model.forecast(steps=horizon)
    return pd.Series(predictions, copy=False)
