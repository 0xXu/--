"""Data input and output helpers."""

from pathlib import Path

import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN, TITLE_COLUMN


def load_training_data(source: str, *, deduplicate: bool = True) -> pd.DataFrame:
    """Load labelled data, validate its schema, and remove duplicate text pairs.

    The title is deliberately retained: it is an informative part of the competition's news
    document. Exact duplicate `Title` + `Description` pairs are removed before a validation
    split, preventing a duplicate document from leaking into both splits.
    """
    frame = pd.read_csv(source)
    _require_columns(frame, [TITLE_COLUMN, TEXT_COLUMN, TARGET_COLUMN])
    if deduplicate:
        frame = drop_conflicting_text_pairs(frame)
        frame = frame.drop_duplicates(subset=[TITLE_COLUMN, TEXT_COLUMN])
    return frame.reset_index(drop=True)


def load_test_data(source: str) -> pd.DataFrame:
    """Load test data and validate the columns required for prediction/submission."""
    frame = pd.read_csv(source)
    _require_columns(frame, [ID_COLUMN, TITLE_COLUMN, TEXT_COLUMN])
    return frame


def join_text_fields(frame: pd.DataFrame) -> pd.Series:
    """Create one document per row for sparse-vector models."""
    _require_columns(frame, [TITLE_COLUMN, TEXT_COLUMN])
    return frame[TITLE_COLUMN].fillna("").str.strip() + "\n" + frame[TEXT_COLUMN].fillna("").str.strip()


def conflicting_text_pair_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify rows whose exact title/description pair has more than one label.

    A contradictory pair cannot be learned reliably and can leak inconsistent targets across a
    validation split. The raw mask is also used by profiling so quality reports never discard
    evidence of the problem.
    """
    _require_columns(frame, [TITLE_COLUMN, TEXT_COLUMN, TARGET_COLUMN])
    label_counts = frame.groupby([TITLE_COLUMN, TEXT_COLUMN], dropna=False)[TARGET_COLUMN].nunique()
    conflicting_pairs = label_counts[label_counts.gt(1)].index
    pair_index = pd.MultiIndex.from_frame(frame[[TITLE_COLUMN, TEXT_COLUMN]])
    return pd.Series(pair_index.isin(conflicting_pairs), index=frame.index)


def drop_conflicting_text_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop every row belonging to a contradictory title/description pair."""
    return frame.loc[~conflicting_text_pair_mask(frame)].copy()


def write_submission(ids: pd.Series, predictions, output_path: str | Path) -> Path:
    """Write predictions in the competition's required submission shape."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: predictions}).to_csv(path, index=False)
    return path


def _require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

