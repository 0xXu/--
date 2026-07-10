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
    _raise_on_conflicting_labels(frame)
    if deduplicate:
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


def _raise_on_conflicting_labels(frame: pd.DataFrame) -> None:
    label_counts = frame.groupby([TITLE_COLUMN, TEXT_COLUMN], dropna=False)[TARGET_COLUMN].nunique()
    if (label_counts > 1).any():
        raise ValueError("Identical title/description pairs have conflicting labels.")
