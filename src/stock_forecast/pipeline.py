"""Application orchestration for stock-price forecasting."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .chronos import ChronosConfig, Device, forecast_chronos
from .data import PathLike, load_time_series, training_target, validate_forecast_dates
from .evaluation import (
    ForecastFunction,
    competition_candidates,
    ensemble_forecast,
    evaluate,
    oof_prediction_frame,
    select_oof_ensemble,
    statistical_candidates,
    write_backtest_reports,
)
from .plotting import save_forecast_plot

DEFAULT_TRAIN_URL = "https://www.dropbox.com/scl/fi/1gzfa2otxd3vqruw4n98e/train.csv?rlkey=2e1kobduti6fqh4bu7r0srpo6&st=fhqqu5xz&dl=1"
DEFAULT_TEST_URL = "https://www.dropbox.com/scl/fi/k8gvgdwfmvang7ac9z3p4/test.csv?rlkey=zxpgjhlp6o7glnkek89nrx0vd&st=8eebzl17&dl=1"


@dataclass(frozen=True)
class ForecastConfig:
    """Inputs and model settings for a single forecast run."""

    train_path: PathLike = DEFAULT_TRAIN_URL
    test_path: PathLike = DEFAULT_TEST_URL
    output_path: PathLike = "submission.csv"
    target_column: str = "price"
    device: Device = "auto"
    model: str = "auto"
    chronos_model_id: str = "amazon/chronos-2"
    cache_dir: PathLike = ".cache/stock-forecast"
    plot_path: PathLike | None = None
    selection_output_path: PathLike | None = None
    backtest_folds: int = 4
    include_chronos_in_auto: bool = False


def run_forecast(config: ForecastConfig) -> pd.DataFrame:
    """Create and persist a submission DataFrame indexed like the test dataset."""
    train_data = load_time_series(config.train_path, target_column=config.target_column, require_target=True, cache_dir=config.cache_dir)
    test_data = load_time_series(config.test_path, target_column=config.target_column, require_target=False, cache_dir=config.cache_dir)
    target = training_target(train_data, config.target_column)
    validate_forecast_dates(train_data, test_data)

    candidates = statistical_candidates()
    if config.include_chronos_in_auto:
        candidates["chronos"] = lambda history, horizon: forecast_chronos(
            history, horizon, ChronosConfig(model_id=config.chronos_model_id, device=config.device)
        )
    if config.model == "chronos":
        predictor: ForecastFunction = lambda history, horizon: forecast_chronos(
            history, horizon, ChronosConfig(model_id=config.chronos_model_id, device=config.device)
        )
        selected_name = "chronos"
    elif config.model == "auto":
        candidates = competition_candidates() | ({"chronos": candidates["chronos"]} if "chronos" in candidates else {})
        oof = oof_prediction_frame(candidates, target, horizon=len(test_data), folds=config.backtest_folds)
        plan = select_oof_ensemble(oof)
        if plan.accepted:
            selected_name = "oof_ensemble"
            predictor = lambda history, horizon: ensemble_forecast(candidates, plan.weights, history, horizon)
        else:
            selected_name = "persistence"
            predictor = candidates[selected_name]
        if config.selection_output_path is not None:
            results = [evaluate(name, candidate, target, horizon=len(test_data), folds=config.backtest_folds) for name, candidate in candidates.items()]
            write_backtest_reports(results, config.selection_output_path)
            selection_path = Path(config.selection_output_path)
            oof.to_csv(selection_path.with_name(f"{selection_path.stem}-oof.csv"), index=False)
            selection_path.with_name(f"{selection_path.stem}-ensemble.json").write_text(
                json.dumps({
                    "objective": "MAE", "accepted": plan.accepted, "weights": dict(plan.weights),
                    "calibration_mae": plan.calibration_mae, "calibration_baseline_mae": plan.calibration_baseline_mae,
                    "calibration_fold_wins": plan.calibration_fold_wins, "confirmation_mae": plan.confirmation_mae,
                    "confirmation_baseline_mae": plan.confirmation_baseline_mae,
                    "confirmation_improvement": 1 - plan.confirmation_mae / plan.confirmation_baseline_mae,
                    "candidate_names": list(candidates),
                }, indent=2),
                encoding="utf-8",
            )
    elif config.model in candidates:
        selected_name = config.model
        predictor = candidates[selected_name]
    else:
        choices = ", ".join(["auto", "chronos", *candidates])
        raise ValueError(f"Unknown model '{config.model}'. Choose one of: {choices}.")

    predictions = predictor(target, len(test_data))
    submission = pd.DataFrame({config.target_column: predictions.to_numpy()}, index=test_data.index)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path)

    if config.plot_path is not None:
        save_forecast_plot(target, submission[config.target_column], config.plot_path)
    submission.attrs["model"] = selected_name
    return submission
