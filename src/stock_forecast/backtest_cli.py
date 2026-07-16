"""CLI for selecting a forecasting model without touching the hidden test set."""

import argparse
from collections.abc import Sequence

from .chronos import ChronosConfig, forecast_chronos, forecast_chronos_lora
from .data import load_time_series, training_target
from .evaluation import evaluate, results_frame, statistical_candidates, write_backtest_reports
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
    parser.add_argument("--include-lora", action="store_true", help="Evaluate fresh Chronos-2 LoRA tuning in every fold.")
    parser.add_argument("--lora-steps", type=int, default=200)
    args = parser.parse_args(argv)
    data = load_time_series(args.train_path, target_column=args.target_column, require_target=True, cache_dir=args.cache_dir)
    target = training_target(data, args.target_column)
    chronos_config = ChronosConfig(device=args.device, local_files_only=True)
    candidates = list(statistical_candidates().items())
    candidates.append(("chronos_2", lambda history, horizon: forecast_chronos(history, horizon, chronos_config)))
    if args.include_lora:
        candidates.append(
            ("chronos_2_lora", lambda history, horizon: forecast_chronos_lora(history, horizon, chronos_config, steps=args.lora_steps))
        )
    results = [evaluate(name, predictor, target, horizon=args.horizon, folds=args.folds) for name, predictor in candidates]
    leaderboard_path, _ = write_backtest_reports(results, args.output)
    leaderboard = results_frame(results)
    print(leaderboard.to_string(index=False))
    print(f"Wrote rolling-backtest results to {leaderboard_path}")
