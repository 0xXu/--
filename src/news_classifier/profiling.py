"""Dataset profiling for transparent experiment setup."""

import json
from pathlib import Path

import pandas as pd

from .constants import TARGET_COLUMN, TEXT_COLUMN, TITLE_COLUMN
from .data import conflicting_text_pair_mask


def profile_frame(frame: pd.DataFrame, labelled: bool) -> dict:
    """Return JSON-serializable schema and quality statistics."""
    result = {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "missing_values": {name: int(value) for name, value in frame.isna().sum().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_text_pairs": int(frame.duplicated([TITLE_COLUMN, TEXT_COLUMN]).sum()),
        "lengths_in_words": {
            name: _length_quantiles(frame[name]) for name in (TITLE_COLUMN, TEXT_COLUMN)
        },
    }
    if labelled:
        conflicting_mask = conflicting_text_pair_mask(frame)
        result["class_counts"] = {
            str(label): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts().sort_index().items()
        }
        result["conflicting_text_pairs"] = int(
            frame.loc[conflicting_mask, [TITLE_COLUMN, TEXT_COLUMN]].drop_duplicates().shape[0]
        )
        result["rows_in_conflicting_text_pairs"] = int(conflicting_mask.sum())
    return result


def write_profile(train: pd.DataFrame, test: pd.DataFrame, output: str | Path) -> Path:
    """Profile both datasets and write a readable JSON artifact."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"train": profile_frame(train, labelled=True), "test": profile_frame(test, False)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def _length_quantiles(column: pd.Series) -> dict[str, int]:
    lengths = column.fillna("").str.split().str.len()
    return {str(key): int(value) for key, value in lengths.quantile([0, .5, .95, .99, 1]).items()}
