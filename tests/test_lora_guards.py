import pandas as pd
import pytest

from stock_forecast.chronos import ChronosConfig, forecast_chronos_lora


def test_lora_rejects_history_that_cannot_form_training_and_validation_windows() -> None:
    history = pd.Series(range(600), index=pd.date_range("2020-01-01", periods=600))

    with pytest.raises(ValueError, match="too short"):
        forecast_chronos_lora(history, 182, ChronosConfig(device="cpu"))
