from news_classifier.transformer import TransformerConfig


def test_transformer_defaults_preserve_effective_batch_size_and_disable_compile():
    config = TransformerConfig()
    assert config.train_batch_size == 16
    assert config.gradient_accumulation_steps == 2
    assert config.train_batch_size * config.gradient_accumulation_steps == 32
    assert config.torch_compile is False
