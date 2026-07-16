import pandas as pd

from stock_forecast.evaluation import evaluate


def test_evaluate_uses_expanding_windows_without_future_leakage() -> None:
    target = pd.Series(range(20), dtype=float)
    seen_lengths: list[int] = []

    def forecaster(history: pd.Series, horizon: int) -> pd.Series:
        seen_lengths.append(len(history))
        return pd.Series([history.iloc[-1]] * horizon)

    result = evaluate("test", forecaster, target, horizon=4, folds=3)

    assert seen_lengths == [8, 12, 16]
    assert result.model == "test"
    assert len(result.fold_mae) == 3
