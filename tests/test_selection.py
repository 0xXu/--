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


def test_analog_forecast_reuses_matching_return_trajectory() -> None:
    from stock_forecast.evaluation import analog_forecast

    history = pd.Series([100.0, 110.0, 121.0, 100.0, 110.0, 121.0, 100.0, 110.0, 121.0])

    forecast = analog_forecast(history, horizon=2, context_length=2, neighbors=1, stride=1)

    assert forecast.tolist() == pytest.approx([100.0, 110.0])


def test_nonnegative_ensemble_weights_favor_perfect_candidate() -> None:
    from stock_forecast.evaluation import fit_nonnegative_weights

    predictions = pd.DataFrame({"good": [1.0, 2.0, 3.0], "bad": [4.0, 5.0, 6.0]})

    weights = fit_nonnegative_weights(predictions, pd.Series([1.0, 2.0, 3.0]))

    assert weights["good"] > 0.99
    assert weights["bad"] < 0.01


def test_oof_ensemble_requires_confirmation_improvement() -> None:
    from stock_forecast.evaluation import select_oof_ensemble

    oof = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "step": [1, 2, 1, 2],
            "actual": [10.0, 20.0, 30.0, 40.0],
            "persistence": [12.0, 22.0, 35.0, 45.0],
            "good": [10.0, 20.0, 30.0, 40.0],
        }
    )

    plan = select_oof_ensemble(oof, min_improvement=0.03)

    assert plan.accepted
    assert dict(plan.weights)["good"] > 0.99
