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


@dataclass(frozen=True)
class EnsemblePlan:
    """Weights learned on calibration folds and checked on a held-out final fold."""

    weights: tuple[tuple[str, float], ...]
    accepted: bool
    calibration_mae: float
    calibration_baseline_mae: float
    calibration_fold_wins: int
    confirmation_mae: float
    confirmation_baseline_mae: float


def persistence_forecast(history: pd.Series, horizon: int) -> pd.Series:
    return pd.Series(np.repeat(history.iloc[-1], horizon))


def seasonal_forecast(history: pd.Series, horizon: int, *, lag: int = 2) -> pd.Series:
    """Repeat a short, explicitly tested cycle without recursively feeding predictions."""
    if lag <= 0 or len(history) < lag:
        raise ValueError("Seasonal lag must be positive and available in history.")
    values = history.to_numpy()
    return pd.Series([values[len(values) - lag + step % lag] for step in range(horizon)])


def ets_forecast(history: pd.Series, horizon: int) -> pd.Series:
    fitted = ExponentialSmoothing(history, trend="add", damped_trend=True, initialization_method="estimated").fit()
    return pd.Series(fitted.forecast(horizon).to_numpy())


def arima_forecast(history: pd.Series, horizon: int) -> pd.Series:
    return pd.Series(ARIMA(history, order=(7, 1, 2)).fit().forecast(horizon).to_numpy())


def analog_forecast(
    history: pd.Series, horizon: int, *, context_length: int = 60, neighbors: int = 5, stride: int = 7
) -> pd.Series:
    """Forecast from the subsequent paths of historically similar return windows."""
    if context_length <= 0 or neighbors <= 0 or stride <= 0:
        raise ValueError("Analog parameters must be positive.")
    if (history <= 0).any():
        raise ValueError("Analog forecasting requires strictly positive prices.")
    values = np.log(history.to_numpy(dtype=float))
    if len(values) < context_length + horizon + 2:
        return persistence_forecast(history, horizon)
    recent = np.diff(values[-(context_length + 1) :])
    recent_scale = max(float(recent.std()), 1e-8)
    normalized_recent = (recent - recent.mean()) / recent_scale
    matches: list[tuple[float, np.ndarray]] = []
    for end in range(context_length + 1, len(values) - horizon + 1, stride):
        pattern = np.diff(values[end - context_length - 1 : end])
        pattern_scale = max(float(pattern.std()), 1e-8)
        distance = float(np.mean((normalized_recent - (pattern - pattern.mean()) / pattern_scale) ** 2))
        future_relative_path = values[end : end + horizon] - values[end - 1]
        matches.append((distance, future_relative_path))
    if not matches:
        return persistence_forecast(history, horizon)
    closest = sorted(matches, key=lambda match: match[0])[:neighbors]
    distances = np.array([match[0] for match in closest])
    weights = 1 / np.maximum(distances, 1e-8)
    weights /= weights.sum()
    paths = np.stack([match[1] for match in closest])
    return pd.Series(np.exp(values[-1] + np.average(paths, axis=0, weights=weights)))


def _project_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the non-negative simplex with sum equal to one."""
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1
    active = np.nonzero(sorted_values - cumulative / np.arange(1, len(values) + 1) > 0)[0]
    threshold = cumulative[active[-1]] / (active[-1] + 1)
    return np.maximum(values - threshold, 0)


def fit_nonnegative_weights(predictions: pd.DataFrame, actual: pd.Series, *, iterations: int = 2_000) -> pd.Series:
    """Fit convex weights for the competition MAE objective without a new dependency."""
    matrix = predictions.to_numpy(dtype=float, copy=True)
    target = actual.to_numpy(dtype=float, copy=True)
    if len(matrix) != len(target) or not len(target):
        raise ValueError("Predictions and actual values must be non-empty and aligned.")
    scale = max(float(np.mean(np.abs(target))), 1e-8)
    matrix /= scale
    target /= scale
    weights = np.full(matrix.shape[1], 1 / matrix.shape[1])
    for iteration in range(iterations):
        gradient = matrix.T @ np.sign(matrix @ weights - target) / len(target)
        updated = _project_simplex(weights - 0.2 / np.sqrt(iteration + 1) * gradient)
        if np.allclose(updated, weights, atol=1e-9):
            break
        weights = updated
    return pd.Series(weights, index=predictions.columns)


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


def _trailing(base: ForecastFunction, window: int) -> ForecastFunction:
    return lambda history, horizon: base(history.iloc[-window:], horizon)


def competition_candidates() -> dict[str, ForecastFunction]:
    """A compact, diverse candidate pool suitable for one long univariate series."""
    return {
        "persistence": persistence_forecast,
        "seasonal_2": lambda history, horizon: seasonal_forecast(history, horizon, lag=2),
        "ets_full": ets_forecast,
        "ets_90": _trailing(ets_forecast, 90),
        "ets_182": _trailing(ets_forecast, 182),
        "ets_365": _trailing(ets_forecast, 365),
        "ets_log_diff_full": lambda history, horizon: transformed_forecast(ets_forecast, "log_diff", history, horizon),
        "ets_log_diff_365": lambda history, horizon: transformed_forecast(ets_forecast, "log_diff", history.iloc[-365:], horizon),
        "analog_365": lambda history, horizon: analog_forecast(history.iloc[-365:], horizon),
        "analog_730": lambda history, horizon: analog_forecast(history.iloc[-730:], horizon),
        "analog_full": analog_forecast,
    }


def oof_prediction_frame(
    candidates: dict[str, ForecastFunction], target: pd.Series, *, horizon: int, folds: int
) -> pd.DataFrame:
    """Generate aligned out-of-fold paths once for reporting and ensemble fitting."""
    if horizon <= 0 or folds < 2 or len(target) < horizon * (folds + 1):
        raise ValueError("Not enough observations for the requested horizon and folds.")
    frames: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(range(folds, 0, -1), start=1):
        cutoff = len(target) - fold * horizon
        history = target.iloc[:cutoff]
        actual = target.iloc[cutoff : cutoff + horizon].to_numpy()
        frame = pd.DataFrame(
            {"fold": fold_number, "cutoff": history.index[-1], "date": target.index[cutoff : cutoff + horizon], "step": np.arange(1, horizon + 1), "actual": actual}
        )
        for name, forecast in candidates.items():
            predicted = forecast(history, horizon).to_numpy(dtype=float)
            if len(predicted) != horizon or not np.isfinite(predicted).all():
                raise ValueError(f"{name} returned invalid predictions in fold {fold_number}.")
            frame[name] = predicted
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def select_oof_ensemble(
    oof: pd.DataFrame, *, baseline_model: str = "persistence", min_improvement: float = 0.03
) -> EnsemblePlan:
    """Fit weights on all but the most recent fold, then gate them on that fold."""
    folds = sorted(oof["fold"].unique())
    if len(folds) < 2:
        raise ValueError("At least two folds are required for ensemble calibration and confirmation.")
    candidates = [column for column in oof.columns if column not in {"fold", "cutoff", "date", "step", "actual"}]
    calibration = oof[oof["fold"] != folds[-1]]
    confirmation = oof[oof["fold"] == folds[-1]]
    weights = fit_nonnegative_weights(calibration[candidates], calibration["actual"])
    calibration_prediction = calibration[candidates].to_numpy() @ weights.to_numpy()
    confirmation_prediction = confirmation[candidates].to_numpy() @ weights.to_numpy()
    calibration_mae = float(np.mean(np.abs(calibration["actual"].to_numpy() - calibration_prediction)))
    calibration_baseline_mae = float(np.mean(np.abs(calibration["actual"].to_numpy() - calibration[baseline_model].to_numpy())))
    calibration_fold_wins = sum(
        np.mean(np.abs(group["actual"].to_numpy() - group[candidates].to_numpy() @ weights.to_numpy()))
        < np.mean(np.abs(group["actual"].to_numpy() - group[baseline_model].to_numpy()))
        for _, group in calibration.groupby("fold")
    )
    confirmation_mae = float(np.mean(np.abs(confirmation["actual"].to_numpy() - confirmation_prediction)))
    confirmation_baseline_mae = float(np.mean(np.abs(confirmation["actual"].to_numpy() - confirmation[baseline_model].to_numpy())))
    accepted = (
        calibration_mae <= calibration_baseline_mae * (1 - min_improvement)
        and calibration_fold_wins >= (len(folds) - 1 + 1) // 2
        and confirmation_mae <= confirmation_baseline_mae * (1 - min_improvement)
    )
    return EnsemblePlan(
        tuple((name, float(weight)) for name, weight in weights.items() if weight > 1e-8),
        bool(accepted),
        calibration_mae,
        calibration_baseline_mae,
        int(calibration_fold_wins),
        confirmation_mae,
        confirmation_baseline_mae,
    )


def ensemble_forecast(candidates: dict[str, ForecastFunction], weights: tuple[tuple[str, float], ...], history: pd.Series, horizon: int) -> pd.Series:
    """Produce the final convex blend from the models selected before the confirmation fold."""
    prediction = np.zeros(horizon)
    for name, weight in weights:
        prediction += weight * candidates[name](history, horizon).to_numpy(dtype=float)
    return pd.Series(prediction)


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
