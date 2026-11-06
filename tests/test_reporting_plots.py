import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas.testing import assert_frame_equal

from src.reporting import (
    plot_cumulative_qlike,
    plot_test_volatility,
)


def _valid_test_forecast_artifacts():
    dates = pd.date_range(
        "2025-01-01",
        periods=2,
        freq="D",
        tz="UTC",
    )
    dates.name = "open_time"

    targets = pd.DataFrame(
        {
            "log_return": [0.01, -0.02],
            "variance_proxy": [0.0001, 0.0004],
        },
        index=dates,
    )

    forecasts = pd.DataFrame(
        {
            "persistence": [0.0002, 0.0001],
            "garch_1_1": [0.0003, 0.0002],
        },
        index=dates,
    )

    return targets, forecasts


def _valid_volatility_panel():
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )

    panel = pd.concat(
        [
            targets[["variance_proxy"]],
            forecasts,
        ],
        axis=1,
    )
    panel.index.name = "date"

    return panel


def test_plot_test_volatility_transforms_variance_to_percent():
    panel = _valid_volatility_panel()

    figure, axis = plot_test_volatility(
        panel
    )

    assert isinstance(figure, Figure)

    expected_columns = list(panel.columns)
    plotted_labels = [
        line.get_label()
        for line in axis.lines
    ]

    assert plotted_labels == [
        "Realized volatility proxy",
        "Persistence",
        "GARCH(1,1)",
    ]

    for line, column in zip(
        axis.lines,
        expected_columns,
        strict=True,
    ):
        expected_values = (
            np.sqrt(panel[column]) * 100.0
        )
        np.testing.assert_allclose(
            line.get_ydata(),
            expected_values,
        )


def test_plot_test_volatility_sets_labels():
    panel = _valid_volatility_panel()

    _, axis = plot_test_volatility(panel)

    assert axis.get_xlabel() == "Date"
    assert (
        axis.get_ylabel()
        == "Daily volatility (%)"
    )
    assert (
        axis.get_title()
        == "One-Day-Ahead Volatility Forecasts"
    )
    assert axis.get_legend() is not None


def test_plot_test_volatility_does_not_mutate_input():
    panel = _valid_volatility_panel()
    original = panel.copy(deep=True)

    plot_test_volatility(panel)

    assert_frame_equal(panel, original)


def _valid_cumulative_qlike():
    dates = pd.date_range(
        "2025-01-01",
        periods=3,
        freq="D",
        tz="UTC",
        name="date",
    )

    return pd.DataFrame(
        {
            "garch_1_1": [-0.1, -0.3, -0.2],
            "ewma_lambda_0.97": [-0.2, -0.4, -0.6],
        },
        index=dates,
    )


def test_plot_cumulative_qlike_plots_each_model_and_zero_line():
    cumulative = _valid_cumulative_qlike()

    figure, axis = plot_cumulative_qlike(
        cumulative,
        benchmark_name="historical_variance",
    )

    assert isinstance(figure, Figure)

    model_lines = {
        line.get_label(): line
        for line in axis.lines
        if not line.get_label().startswith("_")
    }

    expected_labels = {
        "GARCH(1,1)": "garch_1_1",
        "EWMA (lambda=0.97)": (
            "ewma_lambda_0.97"
        ),
    }
    assert set(model_lines) == set(expected_labels)

    for label, model_name in expected_labels.items():
        np.testing.assert_allclose(
            model_lines[label].get_ydata(),
            cumulative[model_name],
        )

    zero_lines = [
        line
        for line in axis.lines
        if line.get_label().startswith("_")
    ]
    assert len(zero_lines) == 1
    np.testing.assert_allclose(
        zero_lines[0].get_ydata(),
        [0.0, 0.0],
    )


def test_plot_cumulative_qlike_sets_interpretive_labels():
    cumulative = _valid_cumulative_qlike()

    _, axis = plot_cumulative_qlike(
        cumulative,
        benchmark_name="historical_variance",
    )

    assert axis.get_xlabel() == "Date"
    assert (
        axis.get_ylabel()
        == "Cumulative QLIKE difference"
    )
    assert (
        axis.get_title()
        == (
            "Cumulative QLIKE Difference vs "
            "Historical variance"
        )
    )
    assert [
        text.get_text()
        for text in axis.texts
    ] == [
        "Negative favors model",
    ]
    assert axis.get_legend() is not None


def test_plot_cumulative_qlike_does_not_mutate_input():
    cumulative = _valid_cumulative_qlike()
    original = cumulative.copy(deep=True)

    plot_cumulative_qlike(
        cumulative,
        benchmark_name="historical_variance",
    )

    assert_frame_equal(cumulative, original)
