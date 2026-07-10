import pytest

from news_classifier.model import ModelConfig, build_model


def test_build_model_uses_configured_ngram_range():
    model = build_model(ModelConfig(word_max_features=10, char_max_features=20, c=0.5))
    assert model.named_steps["vectorizer"].transformer_list[0][1].max_features == 10
    assert model.named_steps["vectorizer"].transformer_list[1][1].max_features == 20
    assert model.named_steps["classifier"].C == 0.5


def test_build_model_rejects_invalid_config():
    with pytest.raises(ValueError):
        build_model(ModelConfig(word_max_features=0))
