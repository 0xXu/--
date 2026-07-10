import json

import pandas as pd

from news_classifier.evaluation import stratified_split, write_evaluation


def test_stratified_split_retains_each_class_in_both_partitions():
    frame = pd.DataFrame(
        {
            "Class Index": [1] * 10 + [2] * 10,
            "Title": [f"title-{index}" for index in range(20)],
            "Description": [f"description-{index}" for index in range(20)],
        }
    )
    train, validation = stratified_split(frame, validation_size=0.2, seed=42)
    assert train["Class Index"].value_counts().to_dict() == {1: 8, 2: 8}
    assert validation["Class Index"].value_counts().to_dict() == {1: 2, 2: 2}


def test_write_evaluation_serializes_pandas_integer_labels(tmp_path):
    validation = pd.DataFrame(
        {
            "Class Index": pd.Series([1, 2], dtype="int64"),
            "Title": ["first", "second"],
            "Description": ["a", "b"],
        }
    )

    metrics = write_evaluation(validation, [1, 2], tmp_path, "tfidf")

    assert metrics["labels"] == [1, 2]
    assert json.loads((tmp_path / "tfidf-metrics.json").read_text())["labels"] == [1, 2]
