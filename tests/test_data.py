import pandas as pd
import pytest

from stock_forecast.data import training_target


def test_training_target_rejects_missing_values() -> None:
    data = pd.DataFrame({"price": [1.0, None]})

    with pytest.raises(ValueError, match="missing values"):
        training_target(data, "price")
