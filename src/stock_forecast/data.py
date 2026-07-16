"""CSV loading and validation for forecasting datasets."""

from pathlib import Path
from typing import TypeAlias
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlretrieve

import hashlib
import time

import pandas as pd

PathLike: TypeAlias = str | Path


def _cache_path(url: str, cache_dir: Path) -> Path:
    """Use a stable, collision-resistant filename for a remote CSV."""
    source_name = Path(urlparse(url).path).name or "dataset.csv"
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return cache_dir / f"{Path(source_name).stem}-{digest}.csv"


def _download_with_cache(url: str, cache_dir: Path, attempts: int = 3) -> Path:
    destination = _cache_path(url, cache_dir)
    if destination.exists():
        return destination
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    for attempt in range(attempts):
        try:
            urlretrieve(url, temporary)
            temporary.replace(destination)
            return destination
        except URLError:
            temporary.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def load_time_series(
    path_or_url: PathLike, *, target_column: str, require_target: bool, cache_dir: PathLike | None = None
) -> pd.DataFrame:
    """Load a date-indexed CSV and validate the expected target column."""
    source = path_or_url
    if cache_dir is not None and isinstance(path_or_url, str) and urlparse(path_or_url).scheme in {"http", "https"}:
        source = _download_with_cache(path_or_url, Path(cache_dir))
    frame = pd.read_csv(source, index_col=0, parse_dates=True)
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
