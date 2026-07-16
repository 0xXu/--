"""CSV loading and validation for forecasting datasets."""

from pathlib import Path
from typing import TypeAlias

import pandas as pd

PathLike: TypeAlias = str | Path


def load_time_series(path_or_url: PathLike, *, target_column: str, require_target: bool) -> pd.DataFrame:
    """Load a date-indexed CSV and validate the expected target column."""
    frame = pd.read_csv(path_or_url, index_col=0, parse_dates=True)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("The first CSV column must be a date index.")
    if frame.index.has_duplicates:
        raise ValueError("The date index must not contain duplicate values.")
    if require_target and target_column not in frame.columns:
        raise ValueError(f"Training data must contain a '{target_column}' column.")
    return frame.sort_index()


def training_target(training_data: pd.DataFrame, target_column: str) -> pd.Series:
    """Return a complete numeric target series suitable for model fitting."""
    target = pd.to_numeric(training_data[target_column], errors="raise")
    if target.empty:
        raise ValueError("Training data must contain at least one observation.")
    if target.isna().any():
        raise ValueError("Training target must not contain missing values.")
    return target
