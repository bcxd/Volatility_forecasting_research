"""Create diagnostic figures from reporting data."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _display_model_name(model_name: str) -> str:
    """Return a readable plot label for a stored model name."""
    if model_name == "persistence":
        return "Persistence"
    if model_name == "historical_variance":
        return "Historical variance"
    if model_name == "garch_1_1":
        return "GARCH(1,1)"
    if model_name.startswith("ewma_lambda_"):
        lambda_value = model_name.removeprefix(
            "ewma_lambda_"
        )
        return f"EWMA (lambda={lambda_value})"

    return model_name


def plot_test_volatility(
    panel: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Plot daily volatility percentages derived from variance data."""
    volatility_percent = (
        np.sqrt(panel) * 100.0
    )

    figure = Figure(
        figsize=(12, 6),
        layout="constrained",
    )
    axis = figure.subplots()

    axis.plot(
        volatility_percent.index,
        volatility_percent["variance_proxy"],
        label="Realized volatility proxy",
        color="black",
        linewidth=1.0,
        alpha=0.45,
    )

    for model_name in volatility_percent.columns[1:]:
        axis.plot(
            volatility_percent.index,
            volatility_percent[model_name],
            label=_display_model_name(model_name),
            linewidth=1.4,
        )

    axis.set_xlabel("Date")
    axis.set_ylabel("Daily volatility (%)")
    axis.set_title(
        "One-Day-Ahead Volatility Forecasts"
    )
    axis.grid(
        visible=True,
        alpha=0.25,
    )
    axis.legend()

    return figure, axis

def plot_cumulative_qlike(
    cumulative_differences: pd.DataFrame,
    benchmark_name: str,
) -> tuple[Figure, Axes]:
    """Plot cumulative model-minus-benchmark QLIKE differences."""
    figure = Figure(
        figsize=(12, 6),
        layout="constrained",
    )
    axis = figure.subplots()

    for model_name in cumulative_differences.columns:
        axis.plot(
            cumulative_differences.index,
            cumulative_differences[model_name],
            label=_display_model_name(model_name),
            linewidth=1.5,
        )

    axis.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        alpha=0.6,
    )
    axis.set_xlabel("Date")
    axis.set_ylabel(
        "Cumulative QLIKE difference"
    )
    axis.set_title(
        "Cumulative QLIKE Difference vs "
        f"{_display_model_name(benchmark_name)}"
    )
    axis.text(
        0.01,
        0.02,
        "Negative favors model",
        transform=axis.transAxes,
        alpha=0.7,
    )
    axis.grid(
        visible=True,
        alpha=0.25,
    )
    axis.legend()

    return figure, axis

