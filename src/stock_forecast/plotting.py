"""Visualizations kept separate from forecasting logic."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_forecast_plot(training: pd.Series, forecast: pd.Series, destination: str | Path) -> Path:
    """Save a chart showing historical observations and forecast values."""
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(15, 8), dpi=300)
    axis.plot(training.index, training, label="Training data", color="darkgreen", alpha=0.5)
    axis.plot(forecast.index, forecast, label="Forecast", color="purple", alpha=0.75)
    axis.grid(color="lightgrey")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
