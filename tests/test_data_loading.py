import pandas as pd

from news_classifier.data import join_text_fields, load_training_data


def test_training_loader_deduplicates_title_and_description_pairs(tmp_path):
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Class Index": [1, 1],
            "Title": ["Shared title", "Shared title"],
            "Description": ["Shared description", "Shared description"],
        }
    ).to_csv(source, index=False)
    frame = load_training_data(source)
    assert len(frame) == 1
    assert join_text_fields(frame).iloc[0] == "Shared title\nShared description"


def test_training_loader_drops_every_row_from_conflicting_text_pairs(tmp_path):
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Class Index": [1, 3, 2, 2, 4],
            "Title": ["Ambiguous", "Ambiguous", "Repeated", "Repeated", "Clean"],
            "Description": ["Same text", "Same text", "Valid text", "Valid text", "Other text"],
        }
    ).to_csv(source, index=False)

    clean = load_training_data(source)
    raw = load_training_data(source, deduplicate=False)

    assert len(raw) == 5
    assert clean[["Title", "Description"]].to_dict("records") == [
        {"Title": "Repeated", "Description": "Valid text"},
        {"Title": "Clean", "Description": "Other text"},
    ]
