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
