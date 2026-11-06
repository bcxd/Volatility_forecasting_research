"""Public reporting API."""

from src.reporting.loaders import (
    load_dm_results,
    load_test_forecast_panel,
    load_test_losses,
    load_test_metrics,
)
from src.reporting.plots import (
    plot_cumulative_qlike,
    plot_test_volatility,
)
from src.reporting.tables import (
    build_cumulative_qlike_differences,
    build_dm_summary,
    build_metric_summary,
)

__all__ = [
    "build_cumulative_qlike_differences",
    "build_dm_summary",
    "build_metric_summary",
    "load_dm_results",
    "load_test_forecast_panel",
    "load_test_losses",
    "load_test_metrics",
    "plot_cumulative_qlike",
    "plot_test_volatility",
]
