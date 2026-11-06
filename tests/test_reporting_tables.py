import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.reporting import (
    build_cumulative_qlike_differences,
    build_dm_summary,
    build_metric_summary,
)


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


def _valid_test_losses():
    dates = pd.date_range(
        "2025-01-01",
        periods=2,
        freq="D",
        tz="UTC",
    )

    index = pd.MultiIndex.from_product(
        [
            dates,
            [
                "garch_1_1",
                "ewma_lambda_0.97",
            ],
        ],
        names=["date", "model"],
    )

    return pd.DataFrame(
        {
            "QLIKE": [-6.1, -6.2, -6.3, -6.4],
            "MSE": [0.4, 0.3, 0.2, 0.1],
            "MAE": [0.04, 0.03, 0.02, 0.01],
        },
        index=index,
    )


def test_build_cumulative_qlike_differences_matches_hand_calculation():
    losses = _valid_test_losses()

    cumulative = (
        build_cumulative_qlike_differences(
            losses,
            benchmark_name="ewma_lambda_0.97",
        )
    )

    assert cumulative.index.equals(
        losses.index
        .get_level_values("date")
        .unique()
    )
    assert cumulative.columns.tolist() == [
        "garch_1_1",
    ]

    np.testing.assert_allclose(
        cumulative["garch_1_1"],
        [0.1, 0.2],
    )


def test_build_cumulative_qlike_differences_uses_model_minus_benchmark():
    losses = _valid_test_losses()

    cumulative = (
        build_cumulative_qlike_differences(
            losses,
            benchmark_name="garch_1_1",
        )
    )

    np.testing.assert_allclose(
        cumulative["ewma_lambda_0.97"],
        [-0.1, -0.2],
    )


def test_build_cumulative_qlike_differences_requires_benchmark():
    losses = _valid_test_losses()

    with pytest.raises(
        ValueError,
        match=(
            "benchmark model not found: "
            "missing_model"
        ),
    ):
        build_cumulative_qlike_differences(
            losses,
            benchmark_name="missing_model",
        )


def test_build_cumulative_qlike_differences_requires_comparison_model():
    losses = _valid_test_losses().xs(
        "garch_1_1",
        level="model",
        drop_level=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "at least one non-benchmark "
            "model is required"
        ),
    ):
        build_cumulative_qlike_differences(
            losses,
            benchmark_name="garch_1_1",
        )


def test_build_cumulative_qlike_differences_does_not_mutate_input():
    losses = _valid_test_losses()
    original = losses.copy(deep=True)

    build_cumulative_qlike_differences(
        losses,
        benchmark_name="garch_1_1",
    )

    assert_frame_equal(losses, original)
