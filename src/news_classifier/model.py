"""Fast, strong sparse baseline for text classification."""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class ModelConfig:
    """Settings for the word and character n-gram TF-IDF baseline."""

    word_max_features: int = 150_000
    char_max_features: int = 100_000
    c: float = 1.0


def build_model(config: ModelConfig = ModelConfig()) -> Pipeline:
    """Build an unfitted title+description TF-IDF + LinearSVC pipeline."""
    if config.word_max_features < 1 or config.char_max_features < 1 or config.c <= 0:
        raise ValueError("Feature limits and c must be greater than zero.")

    return Pipeline(
        [
            (
                "vectorizer",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                strip_accents="unicode",
                                sublinear_tf=True,
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=config.word_max_features,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                sublinear_tf=True,
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=config.char_max_features,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", LinearSVC(C=config.c)),
        ]
    )


def fit_model(texts, labels, config: ModelConfig = ModelConfig()) -> Pipeline:
    """Build and fit the baseline model."""
    model = build_model(config)
    return model.fit(texts, labels)
