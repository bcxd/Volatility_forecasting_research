"""Load saved experiment artifacts for reporting."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


_TEST_METRIC_COLUMNS = (
    "QLIKE",
    "MSE",
    "MAE",
)

_DM_RESULT_COLUMNS = (
    "model_a",
    "model_b",
    "n_observations",
    "bandwidth",
    "mean_loss_a",
    "mean_loss_b",
    "mean_difference",
    "dm_statistic",
    "raw_p_value",
    "holm_adjusted_p_value",
    "familywise_alpha",
    "reject_holm",
)

_DM_NUMERIC_COLUMNS = (
    "n_observations",
    "bandwidth",
    "mean_loss_a",
    "mean_loss_b",
    "mean_difference",
    "dm_statistic",
    "raw_p_value",
    "holm_adjusted_p_value",
    "familywise_alpha",
)

_TEST_TARGET_COLUMNS = (
    "log_return",
    "variance_proxy",
)


_TEST_LOSS_COLUMNS = (
    "QLIKE",
    "MSE",
    "MAE",
)


def _required_artifact_path(
    run_directory: Path,
    filename: str,
) -> Path:
    """Return a required artifact path or raise a clear error."""
    artifact_path = run_directory / filename

    if not artifact_path.is_file():
        raise FileNotFoundError(
            "Required artifact is missing: "
            f"{artifact_path}"
        )

    return artifact_path


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


def load_test_metrics(
    run_directory: Path,
) -> pd.DataFrame:
    """Load aggregate test metrics with model names as the index."""
    metrics_path = _required_artifact_path(
        run_directory,
        "test_metrics.csv",
    )

    metrics = pd.read_csv(
        metrics_path,
        index_col=0,
    )

    if tuple(metrics.columns) != (
        _TEST_METRIC_COLUMNS
    ):
        expected_columns = ", ".join(
            _TEST_METRIC_COLUMNS
        )
        raise ValueError(
            "test metrics columns must be exactly: "
            f"{expected_columns}"
        )

    if metrics.empty:
        raise ValueError(
            "test metrics must not be empty"
        )

    if not metrics.index.is_unique:
        raise ValueError(
            "model names must be unique"
        )

    try:
        metrics = metrics.apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "test metrics must be numeric"
        ) from error

    if not np.isfinite(
        metrics.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "test metrics must be finite"
        )

    metrics.index.name = "model"

    return metrics


def load_dm_results(
    run_directory: Path,
) -> pd.DataFrame:
    """Load saved Diebold-Mariano comparison results."""
    results_path = _required_artifact_path(
        run_directory,
        "dm_results.csv",
    )

    results = pd.read_csv(
        results_path,
        index_col=0,
    ).reset_index(drop=True)

    if tuple(results.columns) != (
        _DM_RESULT_COLUMNS
    ):
        expected_columns = ", ".join(
            _DM_RESULT_COLUMNS
        )
        raise ValueError(
            "DM result columns must be exactly: "
            f"{expected_columns}"
        )

    if results.empty:
        raise ValueError(
            "DM results must not be empty"
        )

    if results.duplicated(
        subset=["model_a", "model_b"]
    ).any():
        raise ValueError(
            "DM comparison pairs must be unique"
        )

    try:
        numeric_values = results.loc[
            :,
            _DM_NUMERIC_COLUMNS,
        ].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "DM numeric columns must be numeric"
        ) from error

    results.loc[
        :,
        _DM_NUMERIC_COLUMNS,
    ] = numeric_values

    if not np.isfinite(
        numeric_values.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "DM numeric columns must be finite"
        )
    p_values = numeric_values.loc[
        :,
        [
            "raw_p_value",
            "holm_adjusted_p_value",
        ],
    ].to_numpy(dtype=float)

    if (
        (p_values < 0.0)
        | (p_values > 1.0)
    ).any():
        raise ValueError(
            "DM p-values must be between 0 and 1"
        )

    alpha_values = numeric_values[
        "familywise_alpha"
    ].to_numpy(dtype=float)

    if (
        (alpha_values <= 0.0)
        | (alpha_values >= 1.0)
    ).any():
        raise ValueError(
            "familywise alpha must be "
            "strictly between 0 and 1"
        )

    n_observations = numeric_values[
        "n_observations"
    ].to_numpy(dtype=float)

    if (
        (n_observations < 2)
        | (
            n_observations
            != np.floor(n_observations)
        )
    ).any():
        raise ValueError(
            "n_observations must be integers "
            "of at least 2"
        )

    bandwidth = numeric_values[
        "bandwidth"
    ].to_numpy(dtype=float)

    if (
        (bandwidth < 0)
        | (bandwidth != np.floor(bandwidth))
        | (bandwidth >= n_observations)
    ).any():
        raise ValueError(
            "bandwidth must be an integer from "
            "0 to n_observations - 1"
        )

    if not pd.api.types.is_bool_dtype(
        results["reject_holm"]
    ):
        raise ValueError(
            "reject_holm must be boolean"
        )

    results["n_observations"] = (
        n_observations.astype(int)
    )
    results["bandwidth"] = (
        bandwidth.astype(int)
    )

    return results


def load_test_forecast_panel(
    run_directory: Path,
) -> pd.DataFrame:
    """Load and align test targets with model forecasts."""
    targets_path = _required_artifact_path(
        run_directory,
        "test_targets.csv",
    )
    forecasts_path = _required_artifact_path(
        run_directory,
        "test_forecasts.csv",
    )

    targets = pd.read_csv(
        targets_path,
        index_col=0,
        parse_dates=[0],
    )
    forecasts = pd.read_csv(
        forecasts_path,
        index_col=0,
        parse_dates=[0],
    )

    if tuple(targets.columns) != (
        _TEST_TARGET_COLUMNS
    ):
        expected_columns = ", ".join(
            _TEST_TARGET_COLUMNS
        )
        raise ValueError(
            "test target columns must be exactly: "
            f"{expected_columns}"
        )

    if (
        len(targets.index) == 0
        or len(forecasts.index) == 0
    ):
        raise ValueError(
            "test forecast panel must not be empty"
        )

    if forecasts.shape[1] == 0:
        raise ValueError(
            "test forecasts must contain "
            "at least one model"
        )

    if not targets.index.is_unique:
        raise ValueError(
            "test target dates must be unique"
        )

    if not forecasts.index.is_unique:
        raise ValueError(
            "test forecast dates must be unique"
        )

    if not targets.index.equals(
        forecasts.index
    ):
        raise ValueError(
            "target and forecast dates must match"
        )

    if (
        not isinstance(
            targets.index,
            pd.DatetimeIndex,
        )
        or not isinstance(
            forecasts.index,
            pd.DatetimeIndex,
        )
        or targets.index.tz is None
        or forecasts.index.tz is None
        or str(targets.index.tz) != "UTC"
        or str(forecasts.index.tz) != "UTC"
    ):
        raise ValueError(
            "test dates must be timezone-aware UTC"
        )

    if not targets.index.is_monotonic_increasing:
        raise ValueError(
            "test dates must be strictly increasing"
        )

    try:
        targets = targets.apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "test targets must be numeric"
        ) from error

    target_values = targets.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        target_values
    ).all():
        raise ValueError(
            "test targets must be finite"
        )

    if (
        targets["variance_proxy"] < 0.0
    ).any():
        raise ValueError(
            "variance proxy must be nonnegative"
        )

    try:
        forecasts = forecasts.apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "test forecasts must be numeric"
        ) from error

    forecast_values = forecasts.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        forecast_values
    ).all():
        raise ValueError(
            "test forecasts must be finite"
        )

    if (forecast_values <= 0.0).any():
        raise ValueError(
            "test forecasts must be positive"
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


def build_metric_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a QLIKE-sorted table with ranks for each loss metric."""
    summary = metrics.copy()

    for metric in _TEST_METRIC_COLUMNS:
        summary[f"{metric}_rank"] = (
            summary[metric]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

    summary = summary.loc[
        :,
        [
            "QLIKE",
            "QLIKE_rank",
            "MSE",
            "MSE_rank",
            "MAE",
            "MAE_rank",
        ],
    ]

    return summary.sort_values(
        by="QLIKE",
        ascending=True,
        kind="stable",
    )


def build_dm_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build an interpretable summary of DM comparisons."""
    summary = results.copy()

    mean_difference = summary[
        "mean_difference"
    ]

    summary["lower_mean_loss_model"] = np.where(
        mean_difference < 0.0,
        summary["model_a"],
        np.where(
            mean_difference > 0.0,
            summary["model_b"],
            "tie",
        ),
    )

    summary["significant_after_holm"] = (
        summary["reject_holm"].copy()
    )

    return summary.loc[
        :,
        [
            "model_a",
            "model_b",
            "n_observations",
            "bandwidth",
            "mean_difference",
            "lower_mean_loss_model",
            "dm_statistic",
            "raw_p_value",
            "holm_adjusted_p_value",
            "familywise_alpha",
            "significant_after_holm",
        ],
    ]


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


def load_test_losses(
    run_directory: Path,
) -> pd.DataFrame:
    """Load daily test losses indexed by date and model."""
    losses_path = _required_artifact_path(
        run_directory,
        "test_losses.csv",
    )

    losses = pd.read_csv(
        losses_path,
        index_col=["date", "model"],
        parse_dates=["date"],
    )

    if tuple(losses.columns) != (
        _TEST_LOSS_COLUMNS
    ):
        expected_columns = ", ".join(
            _TEST_LOSS_COLUMNS
        )
        raise ValueError(
            "test loss columns must be exactly: "
            f"{expected_columns}"
        )

    if losses.empty:
        raise ValueError(
            "test losses must not be empty"
        )

    if not losses.index.is_unique:
        raise ValueError(
            "date-model pairs must be unique"
        )

    dates = losses.index.get_level_values(
        "date"
    )

    if (
        not isinstance(
            dates,
            pd.DatetimeIndex,
        )
        or dates.tz is None
        or str(dates.tz) != "UTC"
    ):
        raise ValueError(
            "test loss dates must be "
            "timezone-aware UTC"
        )

    model_names = (
        losses.index
        .get_level_values("model")
        .unique()
    )

    reference_dates = None

    for model_name in model_names:
        model_dates = losses.xs(
            model_name,
            level="model",
        ).index

        if not model_dates.is_monotonic_increasing:
            raise ValueError(
                "test loss dates must be "
                "strictly increasing within each model"
            )

        if reference_dates is None:
            reference_dates = model_dates
        elif not model_dates.equals(
            reference_dates
        ):
            raise ValueError(
                "all models must share identical dates"
            )

    try:
        losses = losses.apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "test losses must be numeric"
        ) from error

    if not np.isfinite(
        losses.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "test losses must be finite"
        )

    if (
        losses[["MSE", "MAE"]] < 0.0
    ).any().any():
        raise ValueError(
            "MSE and MAE must be nonnegative"
        )

    return losses


def build_cumulative_qlike_differences(
    daily_losses: pd.DataFrame,
    benchmark_name: str,
) -> pd.DataFrame:
    """Accumulate model-minus-benchmark daily QLIKE differences."""
    qlike_by_model = daily_losses[
        "QLIKE"
    ].unstack(
        level="model",
        sort=False,
    )

    if benchmark_name not in qlike_by_model.columns:
        raise ValueError(
            "benchmark model not found: "
            f"{benchmark_name}"
        )

    comparison_models = [
        model_name
        for model_name in qlike_by_model.columns
        if model_name != benchmark_name
    ]

    if not comparison_models:
        raise ValueError(
            "at least one non-benchmark "
            "model is required"
        )

    daily_differences = qlike_by_model.loc[
        :,
        comparison_models,
    ].subtract(
        qlike_by_model[benchmark_name],
        axis="index",
    )

    return daily_differences.cumsum()


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
