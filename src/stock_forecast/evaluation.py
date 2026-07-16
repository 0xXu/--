"""Leakage-free long-horizon backtesting for model selection."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ForecastFunction = Callable[[pd.Series, int], pd.Series]
Transform = Literal["level", "log", "log_diff"]


@dataclass(frozen=True)
class BacktestResult:
    model: str
    mae: float
    rmse: float
    fold_mae: tuple[float, ...]
    fold_rmse: tuple[float, ...]
    fold_mase: tuple[float, ...]
    step_mae: tuple[float, ...]


def persistence_forecast(history: pd.Series, horizon: int) -> pd.Series:
    return pd.Series(np.repeat(history.iloc[-1], horizon))


def ets_forecast(history: pd.Series, horizon: int) -> pd.Series:
    fitted = ExponentialSmoothing(history, trend="add", damped_trend=True, initialization_method="estimated").fit()
    return pd.Series(fitted.forecast(horizon).to_numpy())


def arima_forecast(history: pd.Series, horizon: int) -> pd.Series:
    return pd.Series(ARIMA(history, order=(7, 1, 2)).fit().forecast(horizon).to_numpy())


def auto_arima_forecast(history: pd.Series, horizon: int) -> pd.Series:
    """Fit a deliberately small AIC-selected ARIMA grid inside one training fold."""
    orders = (
        (0, 0, 0), (1, 0, 0), (0, 0, 1),
        (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (2, 1, 2), (3, 1, 1),
        (0, 2, 0), (1, 2, 0), (0, 2, 1), (1, 2, 1),
    )
    fitted_models = []
    for order in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                fitted = ARIMA(history, order=order).fit()
            if fitted.mle_retvals.get("converged", True):
                fitted_models.append(fitted)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not fitted_models:
        raise RuntimeError("No candidate ARIMA model converged for this training window.")
    best = min(fitted_models, key=lambda model: model.aic)
    return pd.Series(best.forecast(horizon).to_numpy())


def transformed_forecast(base_forecast: ForecastFunction, transform: Transform, history: pd.Series, horizon: int) -> pd.Series:
    """Fit a candidate on a leakage-free target transform and invert its forecast."""
    if transform == "level":
        return base_forecast(history, horizon)
    if (history <= 0).any():
        raise ValueError("Log-based forecasting requires strictly positive prices.")
    log_history = pd.Series(np.log(history.to_numpy()))
    if transform == "log":
        return pd.Series(np.exp(base_forecast(log_history, horizon).to_numpy()))
    if transform == "log_diff":
        predicted_changes = base_forecast(log_history.diff().dropna(), horizon).to_numpy()
        return pd.Series(np.exp(np.log(history.iloc[-1]) + np.cumsum(predicted_changes)))
    raise ValueError(f"Unsupported target transform: {transform}")


def statistical_candidates() -> dict[str, ForecastFunction]:
    """Candidates cheap enough to use for automatic, leakage-free selection."""
    return {
        "persistence": persistence_forecast,
        "damped_ets": ets_forecast,
        "auto_arima": auto_arima_forecast,
        "damped_ets_log": lambda history, horizon: transformed_forecast(ets_forecast, "log", history, horizon),
        "auto_arima_log": lambda history, horizon: transformed_forecast(auto_arima_forecast, "log", history, horizon),
        "damped_ets_log_diff": lambda history, horizon: transformed_forecast(ets_forecast, "log_diff", history, horizon),
        "auto_arima_log_diff": lambda history, horizon: transformed_forecast(auto_arima_forecast, "log_diff", history, horizon),
    }


def evaluate(model: str, forecast: ForecastFunction, target: pd.Series, *, horizon: int, folds: int) -> BacktestResult:
    """Evaluate a forecaster on expanding training windows with a fixed real horizon."""
    if horizon <= 0 or folds <= 0 or len(target) < horizon * (folds + 1):
        raise ValueError("Not enough observations for the requested horizon and folds.")
    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_mase: list[float] = []
    step_errors: list[np.ndarray] = []
    for fold in range(folds, 0, -1):
        cutoff = len(target) - fold * horizon
        actual = target.iloc[cutoff : cutoff + horizon].to_numpy()
        predicted = forecast(target.iloc[:cutoff], horizon).to_numpy()
        if len(predicted) != horizon:
            raise ValueError(f"{model} returned {len(predicted)} predictions; expected {horizon}.")
        if not np.isfinite(predicted).all():
            raise ValueError(f"{model} returned non-finite predictions.")
        errors = actual - predicted
        fold_mae.append(float(np.mean(np.abs(errors))))
        fold_rmse.append(float(np.sqrt(np.mean(errors**2))))
        scale = float(np.mean(np.abs(np.diff(target.iloc[:cutoff].to_numpy()))))
        fold_mase.append(float(np.mean(np.abs(errors)) / scale) if scale else float("nan"))
        step_errors.append(np.abs(errors))
    step_mae = tuple(float(value) for value in np.mean(np.stack(step_errors), axis=0))
    return BacktestResult(model, float(np.mean(fold_mae)), float(np.mean(fold_rmse)), tuple(fold_mae), tuple(fold_rmse), tuple(fold_mase), step_mae)


def results_frame(results: list[BacktestResult]) -> pd.DataFrame:
    """Create a leaderboard ordered by the primary MAE metric."""
    return pd.DataFrame(
        [
            {
                "model": item.model,
                "mae": item.mae,
                "rmse": item.rmse,
                "mase": float(np.nanmean(item.fold_mase)),
                "fold_mae": list(item.fold_mae),
                "fold_rmse": list(item.fold_rmse),
                "fold_mase": list(item.fold_mase),
            }
            for item in results
        ]
    ).sort_values("mae", ignore_index=True)


def step_error_frame(results: list[BacktestResult]) -> pd.DataFrame:
    """Return one diagnostic row per forecast horizon and candidate."""
    return pd.DataFrame(
        [
            {"model": result.model, "step": step, "mae": mae}
            for result in results
            for step, mae in enumerate(result.step_mae, start=1)
        ]
    )


def write_backtest_reports(results: list[BacktestResult], destination: str | Path) -> tuple[Path, Path]:
    """Persist the compact leaderboard and separate horizon-level diagnostics."""
    leaderboard_path = Path(destination)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    step_path = leaderboard_path.with_name(f"{leaderboard_path.stem}-step-mae.csv")
    results_frame(results).to_csv(leaderboard_path, index=False)
    step_error_frame(results).to_csv(step_path, index=False)
    return leaderboard_path, step_path


def select_champion(
    results: list[BacktestResult], *, baseline_model: str = "persistence", min_improvement: float = 0.03, min_fold_wins: int = 3
) -> BacktestResult:
    """Select only a candidate that consistently clears a predeclared baseline margin."""
    by_name = {result.model: result for result in results}
    baseline = by_name[baseline_model]
    eligible = []
    for result in results:
        if result.model == baseline_model:
            continue
        improvement = 1 - result.mae / baseline.mae
        fold_wins = sum(candidate < reference for candidate, reference in zip(result.fold_mae, baseline.fold_mae, strict=True))
        latest_improvement = 1 - result.fold_mae[-1] / baseline.fold_mae[-1]
        if improvement >= min_improvement and fold_wins >= min_fold_wins and latest_improvement >= min_improvement:
            eligible.append(result)
    return min(eligible, key=lambda result: result.mae) if eligible else baseline
