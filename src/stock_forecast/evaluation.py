"""Leakage-free long-horizon backtesting for model selection."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ForecastFunction = Callable[[pd.Series, int], pd.Series]


@dataclass(frozen=True)
class BacktestResult:
    model: str
    mae: float
    rmse: float
    fold_mae: tuple[float, ...]


def persistence_forecast(history: pd.Series, horizon: int) -> pd.Series:
    return pd.Series(np.repeat(history.iloc[-1], horizon))


def ets_forecast(history: pd.Series, horizon: int) -> pd.Series:
    fitted = ExponentialSmoothing(history, trend="add", damped_trend=True, initialization_method="estimated").fit()
    return pd.Series(fitted.forecast(horizon).to_numpy())


def arima_forecast(history: pd.Series, horizon: int) -> pd.Series:
    return pd.Series(ARIMA(history, order=(7, 1, 2)).fit().forecast(horizon).to_numpy())


def evaluate(model: str, forecast: ForecastFunction, target: pd.Series, *, horizon: int, folds: int) -> BacktestResult:
    """Evaluate a forecaster on expanding training windows with a fixed real horizon."""
    if horizon <= 0 or folds <= 0 or len(target) < horizon * (folds + 1):
        raise ValueError("Not enough observations for the requested horizon and folds.")
    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    for fold in range(folds, 0, -1):
        cutoff = len(target) - fold * horizon
        actual = target.iloc[cutoff : cutoff + horizon].to_numpy()
        predicted = forecast(target.iloc[:cutoff], horizon).to_numpy()
        if len(predicted) != horizon:
            raise ValueError(f"{model} returned {len(predicted)} predictions; expected {horizon}.")
        errors = actual - predicted
        fold_mae.append(float(np.mean(np.abs(errors))))
        fold_rmse.append(float(np.sqrt(np.mean(errors**2))))
    return BacktestResult(model, float(np.mean(fold_mae)), float(np.mean(fold_rmse)), tuple(fold_mae))


def results_frame(results: list[BacktestResult]) -> pd.DataFrame:
    """Create a leaderboard ordered by the primary MAE metric."""
    return pd.DataFrame(
        [{"model": item.model, "mae": item.mae, "rmse": item.rmse, "fold_mae": list(item.fold_mae)} for item in results]
    ).sort_values("mae", ignore_index=True)
