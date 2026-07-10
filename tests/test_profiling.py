import pandas as pd

from news_classifier.profiling import profile_frame


def test_profile_reports_conflicting_text_pairs_without_failing():
    frame = pd.DataFrame(
        {
            "Class Index": [1, 3, 2],
            "Title": ["Ambiguous", "Ambiguous", "Clean"],
            "Description": ["Same text", "Same text", "Other text"],
        }
    )

    profile = profile_frame(frame, labelled=True)

    assert profile["conflicting_text_pairs"] == 1
    assert profile["rows_in_conflicting_text_pairs"] == 2
