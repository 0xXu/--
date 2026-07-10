import pandas as pd
import pytest

from news_classifier.data import write_submission


def test_write_submission_creates_competition_columns(tmp_path):
    output = write_submission(pd.Series([1, 2]), [4, 1], tmp_path / "submission.csv")
    assert pd.read_csv(output).to_dict("list") == {"id": [1, 2], "Class Index": [4, 1]}


def test_write_submission_creates_parent_directory(tmp_path):
    output = write_submission(pd.Series([1]), [2], tmp_path / "nested" / "submission.csv")
    assert output.exists()
