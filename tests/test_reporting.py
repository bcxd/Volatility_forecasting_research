import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas.testing import assert_frame_equal

from src.reporting import (
    build_dm_summary,
    build_metric_summary,
    load_dm_results,
    load_test_forecast_panel,
    load_test_metrics,
    plot_test_volatility,
)


def test_load_test_metrics_restores_model_index(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    expected = pd.DataFrame(
        {
            "QLIKE": [-6.5, -6.3],
            "MSE": [0.1, 0.2],
            "MAE": [0.3, 0.4],
        },
        index=pd.Index(
            [
                "ewma_lambda_0.97",
                "garch_1_1",
            ],
            name="model",
        ),
    )

    expected.to_csv(
        run_directory / "test_metrics.csv"
    )

    actual = load_test_metrics(
        run_directory
    )

    assert_frame_equal(actual, expected)


def test_load_test_metrics_requires_saved_artifact(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Required artifact is missing: "
            ".*test_metrics\\.csv"
        ),
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_requires_expected_columns(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        {
            "QLIKE": [-6.5],
            "MSE": [0.1],
        },
        index=["garch_1_1"],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="test metrics columns must be exactly",
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_rejects_empty_table(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        columns=["QLIKE", "MSE", "MAE"],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="test metrics must not be empty",
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_requires_unique_models(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        {
            "QLIKE": [-6.5, -6.4],
            "MSE": [0.1, 0.2],
            "MAE": [0.3, 0.4],
        },
        index=[
            "garch_1_1",
            "garch_1_1",
        ],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="model names must be unique",
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_requires_numeric_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        {
            "QLIKE": ["not-a-number"],
            "MSE": [0.1],
            "MAE": [0.3],
        },
        index=["garch_1_1"],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="test metrics must be numeric",
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_requires_finite_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        {
            "QLIKE": [float("inf")],
            "MSE": [0.1],
            "MAE": [0.3],
        },
        index=["garch_1_1"],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="test metrics must be finite",
    ):
        load_test_metrics(run_directory)


def _valid_dm_results():
    return pd.DataFrame(
        {
            "model_a": ["garch_1_1"],
            "model_b": ["ewma_lambda_0.97"],
            "n_observations": [365],
            "bandwidth": [5],
            "mean_loss_a": [-6.61],
            "mean_loss_b": [-6.69],
            "mean_difference": [0.08],
            "dm_statistic": [1.41],
            "raw_p_value": [0.16],
            "holm_adjusted_p_value": [0.25],
            "familywise_alpha": [0.05],
            "reject_holm": [False],
        }
    )


def test_load_dm_results_restores_saved_table(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    expected = _valid_dm_results()
    expected.to_csv(
        run_directory / "dm_results.csv"
    )

    actual = load_dm_results(run_directory)

    assert_frame_equal(actual, expected)


def test_load_dm_results_requires_saved_artifact(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Required artifact is missing: "
            ".*dm_results\\.csv"
        ),
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_expected_columns(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results().drop(
        columns=["raw_p_value"]
    )
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM result columns must be exactly",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_rejects_empty_table(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results().iloc[0:0]
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM results must not be empty",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_unique_pairs(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = pd.concat(
        [
            _valid_dm_results(),
            _valid_dm_results(),
        ],
        ignore_index=True,
    )
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM comparison pairs must be unique",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_numeric_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results["raw_p_value"] = (
        results["raw_p_value"].astype(object)
    )
    results.loc[0, "raw_p_value"] = (
        "not-a-number"
    )

    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM numeric columns must be numeric",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_finite_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results.loc[0, "dm_statistic"] = float(
        "inf"
    )
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM numeric columns must be finite",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_valid_p_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results.loc[0, "raw_p_value"] = 1.1
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM p-values must be between 0 and 1",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_valid_alpha(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results.loc[0, "familywise_alpha"] = 0.0
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            "familywise alpha must be "
            "strictly between 0 and 1"
        ),
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_valid_sample_size(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results["n_observations"] = (
        results["n_observations"].astype(float)
    )
    results.loc[0, "n_observations"] = 1.5
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            "n_observations must be integers "
            "of at least 2"
        ),
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_valid_bandwidth(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results.loc[0, "bandwidth"] = 365
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            "bandwidth must be an integer from "
            "0 to n_observations - 1"
        ),
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_boolean_decisions(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results["reject_holm"] = (
        results["reject_holm"].astype(object)
    )
    results.loc[0, "reject_holm"] = (
        "not-a-boolean"
    )
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="reject_holm must be boolean",
    ):
        load_dm_results(run_directory)


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


def _write_test_forecast_artifacts(
    tmp_path,
    targets,
    forecasts,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    targets.to_csv(
        run_directory / "test_targets.csv"
    )
    forecasts.to_csv(
        run_directory / "test_forecasts.csv"
    )

    return run_directory


def test_load_test_forecast_panel_aligns_saved_data(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    expected = pd.concat(
        [
            targets[["variance_proxy"]],
            forecasts,
        ],
        axis=1,
    )
    expected.index.name = "date"

    actual = load_test_forecast_panel(
        run_directory
    )

    assert_frame_equal(
        actual,
        expected,
        check_freq=False,
    )


def test_load_test_forecast_panel_requires_targets(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Required artifact is missing: "
            ".*test_targets\\.csv"
        ),
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_forecasts(
    tmp_path,
):
    targets, _ = (
        _valid_test_forecast_artifacts()
    )
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    targets.to_csv(
        run_directory / "test_targets.csv"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Required artifact is missing: "
            ".*test_forecasts\\.csv"
        ),
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_target_columns(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    targets = targets.drop(
        columns=["log_return"]
    )
    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test target columns must be exactly",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_matching_dates(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    forecasts.index = forecasts.index.shift(
        1,
        freq="D",
    )
    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="target and forecast dates must match",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_unique_dates(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    forecasts.index = pd.DatetimeIndex(
        [
            forecasts.index[0],
            forecasts.index[0],
        ],
        name="open_time",
    )
    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test forecast dates must be unique",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_numeric_forecasts(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    forecasts["garch_1_1"] = (
        forecasts["garch_1_1"].astype(object)
    )
    forecasts.loc[
        forecasts.index[0],
        "garch_1_1",
    ] = "not-a-number"

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test forecasts must be numeric",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_finite_forecasts(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    forecasts.loc[
        forecasts.index[0],
        "garch_1_1",
    ] = float("inf")

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test forecasts must be finite",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_positive_forecasts(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    forecasts.loc[
        forecasts.index[0],
        "garch_1_1",
    ] = 0.0

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test forecasts must be positive",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_rejects_empty_data(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    targets = targets.iloc[0:0]
    forecasts = forecasts.iloc[0:0]

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test forecast panel must not be empty",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_model_column(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    forecasts = forecasts.iloc[:, 0:0]

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "test forecasts must contain "
            "at least one model"
        ),
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_numeric_targets(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    targets["log_return"] = (
        targets["log_return"].astype(object)
    )
    targets.loc[
        targets.index[0],
        "log_return",
    ] = "not-a-number"

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test targets must be numeric",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_finite_targets(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    targets.loc[
        targets.index[0],
        "variance_proxy",
    ] = float("inf")

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test targets must be finite",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_nonnegative_target(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    targets.loc[
        targets.index[0],
        "variance_proxy",
    ] = -0.0001

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="variance proxy must be nonnegative",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_chronological_dates(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    targets = targets.iloc[::-1]
    forecasts = forecasts.loc[targets.index]

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match="test dates must be strictly increasing",
    ):
        load_test_forecast_panel(run_directory)


def test_load_test_forecast_panel_requires_utc_dates(
    tmp_path,
):
    targets, forecasts = (
        _valid_test_forecast_artifacts()
    )
    targets.index = (
        targets.index.tz_localize(None)
    )
    forecasts.index = (
        forecasts.index.tz_localize(None)
    )

    run_directory = (
        _write_test_forecast_artifacts(
            tmp_path,
            targets,
            forecasts,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "test dates must be timezone-aware UTC"
        ),
    ):
        load_test_forecast_panel(run_directory)


def _valid_test_metrics():
    return pd.DataFrame(
        {
            "QLIKE": [10.0, 2.0, 3.0],
            "MSE": [3.0, 1.0, 2.0],
            "MAE": [1.0, 2.0, 0.5],
        },
        index=pd.Index(
            [
                "persistence",
                "ewma_lambda_0.97",
                "garch_1_1",
            ],
            name="model",
        ),
    )


def test_build_metric_summary_sorts_and_ranks_models():
    metrics = _valid_test_metrics()

    actual = build_metric_summary(metrics)

    expected = pd.DataFrame(
        {
            "QLIKE": [2.0, 3.0, 10.0],
            "QLIKE_rank": [1, 2, 3],
            "MSE": [1.0, 2.0, 3.0],
            "MSE_rank": [1, 2, 3],
            "MAE": [2.0, 0.5, 1.0],
            "MAE_rank": [3, 1, 2],
        },
        index=pd.Index(
            [
                "ewma_lambda_0.97",
                "garch_1_1",
                "persistence",
            ],
            name="model",
        ),
    )

    assert_frame_equal(actual, expected)


def test_build_metric_summary_gives_ties_same_rank():
    metrics = _valid_test_metrics()
    metrics.loc[
        "garch_1_1",
        "QLIKE",
    ] = 2.0

    summary = build_metric_summary(metrics)

    assert (
        summary.loc[
            "ewma_lambda_0.97",
            "QLIKE_rank",
        ]
        == 1
    )
    assert (
        summary.loc[
            "garch_1_1",
            "QLIKE_rank",
        ]
        == 1
    )


def test_build_metric_summary_does_not_mutate_input():
    metrics = _valid_test_metrics()
    original = metrics.copy(deep=True)

    build_metric_summary(metrics)

    assert_frame_equal(metrics, original)


def _dm_results_with_all_directions():
    negative = _valid_dm_results()
    negative.loc[0, "mean_difference"] = -0.2
    negative.loc[0, "model_a"] = "garch_1_1"
    negative.loc[0, "model_b"] = "persistence"

    positive = _valid_dm_results()
    positive.loc[0, "mean_difference"] = 0.1
    positive.loc[0, "model_a"] = "garch_1_1"
    positive.loc[0, "model_b"] = "ewma_lambda_0.97"

    tie = _valid_dm_results()
    tie.loc[0, "mean_difference"] = 0.0
    tie.loc[0, "model_a"] = "ewma_lambda_0.97"
    tie.loc[0, "model_b"] = "historical_variance"

    return pd.concat(
        [negative, positive, tie],
        ignore_index=True,
    )


def test_build_dm_summary_identifies_lower_loss_model():
    results = _dm_results_with_all_directions()

    summary = build_dm_summary(results)

    assert summary[
        "lower_mean_loss_model"
    ].tolist() == [
        "garch_1_1",
        "ewma_lambda_0.97",
        "tie",
    ]


def test_build_dm_summary_reports_holm_decision():
    results = _dm_results_with_all_directions()
    results["reject_holm"] = [
        True,
        False,
        False,
    ]

    summary = build_dm_summary(results)

    assert summary[
        "significant_after_holm"
    ].tolist() == [
        True,
        False,
        False,
    ]


def test_build_dm_summary_keeps_reporting_columns():
    results = _dm_results_with_all_directions()

    summary = build_dm_summary(results)

    assert list(summary.columns) == [
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
    ]


def test_build_dm_summary_does_not_mutate_input():
    results = _dm_results_with_all_directions()
    original = results.copy(deep=True)

    build_dm_summary(results)

    assert_frame_equal(results, original)


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
        *expected_columns[1:],
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
