"""Command-line interface for the forecasting application."""

import argparse
from collections.abc import Sequence

from .pipeline import DEFAULT_TEST_URL, DEFAULT_TRAIN_URL, ForecastConfig, run_forecast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast stock prices with Chronos-2.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_URL, help="Training CSV path or URL.")
    parser.add_argument("--test-path", default=DEFAULT_TEST_URL, help="Test CSV path or URL.")
    parser.add_argument("--output", default="submission.csv", help="CSV output path.")
    parser.add_argument("--target-column", default="price", help="Column to forecast.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Chronos execution device (default: auto).")
    parser.add_argument("--chronos-model-id", default="amazon/chronos-2", help="Hugging Face Chronos model ID.")
    parser.add_argument("--model", default="auto", help="Forecast model: auto, chronos, or a statistical candidate.")
    parser.add_argument("--include-chronos", action="store_true", help="Include Chronos-2 in automatic rolling-backtest selection.")
    parser.add_argument("--cache-dir", default=".cache/stock-forecast", help="Persistent CSV download cache directory.")
    parser.add_argument("--plot", help="Optional PNG output path for a forecast chart.")
    parser.add_argument("--selection-output", help="Optional CSV path for automatic-selection backtest diagnostics.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    submission = run_forecast(
        ForecastConfig(
            train_path=args.train_path,
            test_path=args.test_path,
            output_path=args.output,
            target_column=args.target_column,
            device=args.device,
            model=args.model,
            include_chronos_in_auto=args.include_chronos,
            chronos_model_id=args.chronos_model_id,
            cache_dir=args.cache_dir,
            plot_path=args.plot,
            selection_output_path=args.selection_output,
        )
    )
    print(f"Wrote {len(submission)} {submission.attrs['model']} predictions to {args.output}")


if __name__ == "__main__":
    main()
