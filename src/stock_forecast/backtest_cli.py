"""CLI for selecting a forecasting model without touching the hidden test set."""

import argparse
from collections.abc import Sequence

from .chronos import ChronosConfig, forecast_chronos
from .data import load_time_series, training_target
from .evaluation import arima_forecast, ets_forecast, evaluate, persistence_forecast, results_frame
from .pipeline import DEFAULT_TRAIN_URL


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free 182-day rolling backtests.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_URL)
    parser.add_argument("--target-column", default="price")
    parser.add_argument("--horizon", type=int, default=182)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--output", default="backtest-results.csv")
    parser.add_argument("--cache-dir", default=".cache/stock-forecast")
    args = parser.parse_args(argv)
    data = load_time_series(args.train_path, target_column=args.target_column, require_target=True, cache_dir=args.cache_dir)
    target = training_target(data, args.target_column)
    chronos_config = ChronosConfig(device=args.device)
    candidates = [
        ("persistence", persistence_forecast),
        ("damped_ets", ets_forecast),
        ("arima_7_1_2", arima_forecast),
        ("chronos_2", lambda history, horizon: forecast_chronos(history, horizon, chronos_config)),
    ]
    leaderboard = results_frame([evaluate(name, predictor, target, horizon=args.horizon, folds=args.folds) for name, predictor in candidates])
    leaderboard.to_csv(args.output, index=False)
    print(leaderboard.to_string(index=False))
    print(f"Wrote rolling-backtest results to {args.output}")
