"""Fixed, leakage-aware holdout evaluation and artifacts."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .constants import TARGET_COLUMN


def stratified_split(frame: pd.DataFrame, validation_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split already-deduplicated labelled data while preserving class proportions."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be strictly between 0 and 1.")
    return train_test_split(
        frame,
        test_size=validation_size,
        random_state=seed,
        stratify=frame[TARGET_COLUMN],
    )


def write_evaluation(
    validation: pd.DataFrame, predictions, output_dir: str | Path, prefix: str
) -> dict:
    """Write metrics, confusion matrix, and row-level validation predictions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = sorted(validation[TARGET_COLUMN].unique())
    actual = validation[TARGET_COLUMN]
    metrics = {
        "accuracy": accuracy_score(actual, predictions),
        "macro_f1": f1_score(actual, predictions, average="macro"),
        "classification_report": classification_report(actual, predictions, output_dict=True),
        "labels": labels,
        "confusion_matrix": confusion_matrix(actual, predictions, labels=labels).tolist(),
    }
    (output / f"{prefix}-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    validation.assign(prediction=predictions).to_csv(output / f"{prefix}-validation-predictions.csv", index=False)
    return metrics
