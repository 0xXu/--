"""Command-line interface for the forecasting application."""

import argparse
from collections.abc import Sequence

from .pipeline import DEFAULT_TEST_URL, DEFAULT_TRAIN_URL, ForecastConfig, run_forecast


def parse_order(value: str) -> tuple[int, int, int]:
    """Parse a comma-separated ARIMA order, such as ``7,1,2``."""
    try:
        order = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ARIMA order must be three integers, e.g. 7,1,2.") from error
    if len(order) != 3 or any(part < 0 for part in order):
        raise argparse.ArgumentTypeError("ARIMA order must be three non-negative integers, e.g. 7,1,2.")
    return order  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast stock prices with ARIMA.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_URL, help="Training CSV path or URL.")
    parser.add_argument("--test-path", default=DEFAULT_TEST_URL, help="Test CSV path or URL.")
    parser.add_argument("--output", default="submission.csv", help="CSV output path.")
    parser.add_argument("--target-column", default="price", help="Column to forecast.")
    parser.add_argument("--arima-order", type=parse_order, default=(7, 1, 2), help="ARIMA order p,d,q (default: 7,1,2).")
    parser.add_argument("--plot", help="Optional PNG output path for a forecast chart.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    submission = run_forecast(
        ForecastConfig(
            train_path=args.train_path,
            test_path=args.test_path,
            output_path=args.output,
            target_column=args.target_column,
            arima_order=args.arima_order,
            plot_path=args.plot,
        )
    )
    print(f"Wrote {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
