import pandas as pd
import pytest

from stock_forecast.evaluation import BacktestResult, select_champion


def result(name: str, mae: float, fold_mae: tuple[float, ...]) -> BacktestResult:
    return BacktestResult(name, mae, mae + 1, fold_mae, fold_mae, fold_mae, tuple(range(len(fold_mae))))


def test_select_champion_requires_consistent_improvement_over_persistence() -> None:
    results = [
        result("persistence", 100.0, (100.0, 100.0, 100.0, 100.0)),
        result("ets", 95.0, (94.0, 95.0, 96.0, 95.0)),
    ]

    assert select_champion(results, min_improvement=0.03).model == "ets"


def test_select_champion_keeps_baseline_when_latest_fold_does_not_clear_threshold() -> None:
    results = [
        result("persistence", 100.0, (100.0, 100.0, 100.0, 100.0)),
        result("ets", 94.0, (90.0, 90.0, 90.0, 106.0)),
    ]

    assert select_champion(results, min_improvement=0.03).model == "persistence"


def test_log_transform_forecast_returns_to_price_scale() -> None:
    from stock_forecast.evaluation import transformed_forecast

    history = pd.Series([10.0, 20.0, 40.0])
    forecast = transformed_forecast(lambda values, horizon: pd.Series([values.iloc[-1]] * horizon), "log", history, 2)

    assert forecast.tolist() == [40.0, 40.0]


def test_log_difference_transform_compounds_predicted_returns() -> None:
    from stock_forecast.evaluation import transformed_forecast

    history = pd.Series([10.0, 20.0, 40.0])
    forecast = transformed_forecast(lambda values, horizon: pd.Series([values.iloc[-1]] * horizon), "log_diff", history, 2)

    assert forecast.tolist() == pytest.approx([80.0, 160.0])
