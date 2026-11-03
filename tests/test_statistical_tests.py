import numpy as np
import pandas as pd
import pytest

from src.statistical_tests import (
    compare_qlike_losses,
    diebold_mariano_test,
    holm_adjust_pvalues,
    newey_west_bandwidth,
    newey_west_long_run_variance,
)


@pytest.mark.parametrize(
    ("n_observations", "expected_bandwidth"),
    [
        (100, 4),
        (365, 5),
    ],
)
def test_newey_west_bandwidth_uses_prespecified_rule(
    n_observations,
    expected_bandwidth,
):
    bandwidth = newey_west_bandwidth(
        n_observations
    )

    assert bandwidth == expected_bandwidth


@pytest.mark.parametrize(
    "n_observations",
    [
        -1,
        0,
        1,
    ],
)
def test_newey_west_bandwidth_requires_two_observations(
    n_observations,
):
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        newey_west_bandwidth(
            n_observations
        )


def test_newey_west_long_run_variance_matches_hand_calculation():
    loss_differential = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    long_run_variance = (
        newey_west_long_run_variance(
            loss_differential,
            bandwidth=1,
        )
    )

    assert long_run_variance == pytest.approx(
        1.5625
    )


@pytest.mark.parametrize(
    "bandwidth",
    [
        -1,
        4,
    ],
)
def test_newey_west_long_run_variance_rejects_invalid_bandwidth(
    bandwidth,
):
    loss_differential = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="0 <= bandwidth",
    ):
        newey_west_long_run_variance(
            loss_differential,
            bandwidth=bandwidth,
        )


@pytest.mark.parametrize(
    "loss_differential",
    [
        np.array([]),
        np.array([1.0]),
    ],
)
def test_newey_west_long_run_variance_requires_two_observations(
    loss_differential,
):
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        newey_west_long_run_variance(
            loss_differential,
            bandwidth=0,
        )


def test_newey_west_long_run_variance_requires_1d_input():
    loss_differential = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="1D",
    ):
        newey_west_long_run_variance(
            loss_differential,
            bandwidth=0,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_newey_west_long_run_variance_requires_finite_values(
    invalid_value,
):
    loss_differential = np.array(
        [
            1.0,
            invalid_value,
        ]
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        newey_west_long_run_variance(
            loss_differential,
            bandwidth=0,
        )


def test_diebold_mariano_matches_hand_calculation():
    loss_a = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )
    loss_b = np.array(
        [
            2.0,
            4.0,
            6.0,
            8.0,
        ]
    )

    result = diebold_mariano_test(
        loss_a,
        loss_b,
    )

    assert result.n_observations == 4
    assert result.bandwidth == 1
    assert result.mean_loss_a == pytest.approx(
        2.5
    )
    assert result.mean_loss_b == pytest.approx(
        5.0
    )
    assert result.mean_difference == pytest.approx(
        -2.5
    )
    assert result.statistic == pytest.approx(
        -4.0
    )
    assert result.p_value == pytest.approx(
        0.0000633425,
    )


def test_diebold_mariano_requires_equal_length_losses():
    loss_a = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )
    loss_b = np.array(
        [
            1.0,
            2.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="same number",
    ):
        diebold_mariano_test(
            loss_a,
            loss_b,
        )


@pytest.mark.parametrize(
    ("loss_a", "loss_b"),
    [
        (
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
        ),
        (
            np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
        ),
    ],
)
def test_diebold_mariano_requires_1d_losses(
    loss_a,
    loss_b,
):
    with pytest.raises(
        ValueError,
        match="loss_a and loss_b must be 1D",
    ):
        diebold_mariano_test(
            loss_a,
            loss_b,
        )


@pytest.mark.parametrize(
    "n_observations",
    [
        0,
        1,
    ],
)
def test_diebold_mariano_requires_two_paired_observations(
    n_observations,
):
    loss_a = np.arange(
        n_observations,
        dtype=float,
    )
    loss_b = np.arange(
        n_observations,
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match=(
            "loss_a and loss_b must each "
            "contain at least 2 observations"
        ),
    ):
        diebold_mariano_test(
            loss_a,
            loss_b,
        )


@pytest.mark.parametrize(
    "invalid_side",
    [
        "a",
        "b",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_diebold_mariano_requires_finite_losses(
    invalid_side,
    invalid_value,
):
    loss_a = np.array(
        [
            1.0,
            2.0,
        ]
    )
    loss_b = np.array(
        [
            2.0,
            3.0,
        ]
    )

    if invalid_side == "a":
        loss_a[1] = invalid_value
    else:
        loss_b[1] = invalid_value

    with pytest.raises(
        ValueError,
        match=(
            "loss_a and loss_b must contain "
            "only finite values"
        ),
    ):
        diebold_mariano_test(
            loss_a,
            loss_b,
        )


@pytest.mark.parametrize(
    ("loss_a", "loss_b"),
    [
        (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            ),
        ),
        (
            np.array(
                [
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                ]
            ),
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            ),
        ),
    ],
)
def test_diebold_mariano_rejects_nonpositive_long_run_variance(
    loss_a,
    loss_b,
):
    with pytest.raises(
        ValueError,
        match="long-run variance must be positive",
    ):
        diebold_mariano_test(
            loss_a,
            loss_b,
        )


def test_holm_adjustment_matches_hand_calculation():
    raw_p_values = np.array(
        [
            0.200,
            0.004,
            0.018,
            0.012,
            0.600,
        ]
    )

    result = holm_adjust_pvalues(
        raw_p_values,
        alpha=0.05,
    )

    expected_adjusted = np.array(
        [
            0.400,
            0.020,
            0.054,
            0.048,
            0.600,
        ]
    )
    expected_reject = np.array(
        [
            False,
            True,
            False,
            True,
            False,
        ]
    )

    np.testing.assert_allclose(
        result.adjusted_p_values,
        expected_adjusted,
    )
    np.testing.assert_array_equal(
        result.reject,
        expected_reject,
    )


def test_holm_adjustment_requires_1d_p_values():
    p_values = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.04],
        ]
    )

    with pytest.raises(
        ValueError,
        match="p_values must be a 1D array",
    ):
        holm_adjust_pvalues(
            p_values
        )


def test_holm_adjustment_requires_at_least_one_p_value():
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        holm_adjust_pvalues(
            np.array([])
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_holm_adjustment_requires_finite_p_values(
    invalid_value,
):
    p_values = np.array(
        [
            0.01,
            invalid_value,
        ]
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        holm_adjust_pvalues(
            p_values
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
    ],
)
def test_holm_adjustment_requires_unit_interval_p_values(
    invalid_value,
):
    p_values = np.array(
        [
            0.01,
            invalid_value,
        ]
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        holm_adjust_pvalues(
            p_values
        )


@pytest.mark.parametrize(
    "alpha",
    [
        -0.01,
        0.0,
        1.0,
        1.01,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_holm_adjustment_requires_valid_alpha(
    alpha,
):
    p_values = np.array(
        [
            0.01,
            0.02,
        ]
    )

    with pytest.raises(
        ValueError,
        match="0 < alpha < 1",
    ):
        holm_adjust_pvalues(
            p_values,
            alpha=alpha,
        )


def test_compare_qlike_losses_runs_prespecified_family():
    dates = pd.date_range(
        "2025-01-01",
        periods=6,
        freq="D",
        tz="UTC",
        name="date",
    )

    wide_losses = pd.DataFrame(
        {
            "persistence": [
                5.0,
                7.0,
                4.0,
                9.0,
                6.0,
                8.0,
            ],
            "historical_variance": [
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            "ewma_lambda_0.97": [
                3.0,
                6.0,
                2.0,
                8.0,
                4.0,
                7.0,
            ],
            "garch_1_1": [
                4.0,
                4.0,
                3.0,
                6.0,
                5.0,
                5.0,
            ],
        },
        index=dates,
    )
    wide_losses.columns.name = "model"

    daily_losses = (
        wide_losses.stack()
        .to_frame(name="QLIKE")
    )

    results = compare_qlike_losses(
        daily_losses,
        selected_ewma_name=(
            "ewma_lambda_0.97"
        ),
        alpha=0.05,
    )

    expected_pairs = [
        (
            "garch_1_1",
            "ewma_lambda_0.97",
        ),
        (
            "garch_1_1",
            "persistence",
        ),
        (
            "garch_1_1",
            "historical_variance",
        ),
        (
            "ewma_lambda_0.97",
            "persistence",
        ),
        (
            "ewma_lambda_0.97",
            "historical_variance",
        ),
    ]

    actual_pairs = list(
        results[
            [
                "model_a",
                "model_b",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    assert actual_pairs == expected_pairs
    assert results[
        "n_observations"
    ].tolist() == [6] * 5
    assert results[
        "bandwidth"
    ].tolist() == [2] * 5

    assert np.isfinite(
        results["raw_p_value"]
    ).all()
    assert np.isfinite(
        results["holm_adjusted_p_value"]
    ).all()

    assert (
        results["holm_adjusted_p_value"]
        >= results["raw_p_value"]
    ).all()
    assert results[
        "familywise_alpha"
    ].tolist() == [0.05] * 5
    assert results[
        "reject_holm"
    ].dtype == bool


def test_compare_qlike_losses_requires_identical_forecast_dates():
    dates = pd.date_range(
        "2025-01-01",
        periods=6,
        freq="D",
        tz="UTC",
    )
    shifted_dates = (
        dates
        + pd.Timedelta(days=1)
    )

    model_losses = {
        "persistence": [
            5.0,
            7.0,
            4.0,
            9.0,
            6.0,
            8.0,
        ],
        "historical_variance": [
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
        ],
        "ewma_lambda_0.97": [
            3.0,
            6.0,
            2.0,
            8.0,
            4.0,
            7.0,
        ],
        "garch_1_1": [
            4.0,
            4.0,
            3.0,
            6.0,
            5.0,
            5.0,
        ],
    }

    frames = []

    for model, losses in model_losses.items():
        model_dates = dates

        if model == "ewma_lambda_0.97":
            model_dates = shifted_dates

        frames.append(
            pd.DataFrame(
                {
                    "date": model_dates,
                    "model": model,
                    "QLIKE": losses,
                }
            )
        )

    daily_losses = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .set_index(
            [
                "date",
                "model",
            ]
        )
        .sort_index()
    )

    with pytest.raises(
        ValueError,
        match="identical forecast dates",
    ):
        compare_qlike_losses(
            daily_losses,
            selected_ewma_name=(
                "ewma_lambda_0.97"
            ),
            alpha=0.05,
        )


def test_compare_qlike_losses_requires_multiindex():
    daily_losses = pd.DataFrame(
        {
            "QLIKE": [
                1.0,
                2.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="MultiIndex with date and model levels",
    ):
        compare_qlike_losses(
            daily_losses,
            selected_ewma_name=(
                "ewma_lambda_0.97"
            ),
            alpha=0.05,
        )


def test_compare_qlike_losses_requires_named_index_levels():
    index = pd.MultiIndex.from_product(
        [
            pd.date_range(
                "2025-01-01",
                periods=2,
                tz="UTC",
            ),
            [
                "garch_1_1",
                "ewma_lambda_0.97",
            ],
        ],
        names=[
            "timestamp",
            "asset",
        ],
    )
    daily_losses = pd.DataFrame(
        {
            "QLIKE": np.arange(
                len(index),
                dtype=float,
            ),
        },
        index=index,
    )

    with pytest.raises(
        ValueError,
        match="levels named date and model",
    ):
        compare_qlike_losses(
            daily_losses,
            selected_ewma_name=(
                "ewma_lambda_0.97"
            ),
            alpha=0.05,
        )


def test_compare_qlike_losses_requires_qlike_column():
    dates = pd.date_range(
        "2025-01-01",
        periods=2,
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
        names=[
            "date",
            "model",
        ],
    )
    daily_losses = pd.DataFrame(
        {
            "MSE": np.arange(
                len(index),
                dtype=float,
            ),
        },
        index=index,
    )

    with pytest.raises(
        ValueError,
        match="QLIKE column",
    ):
        compare_qlike_losses(
            daily_losses,
            selected_ewma_name=(
                "ewma_lambda_0.97"
            ),
            alpha=0.05,
        )


def test_compare_qlike_losses_rejects_duplicate_date_model_pairs():
    date = pd.Timestamp(
        "2025-01-01",
        tz="UTC",
    )
    index = pd.MultiIndex.from_tuples(
        [
            (
                date,
                "garch_1_1",
            ),
            (
                date,
                "garch_1_1",
            ),
            (
                date,
                "ewma_lambda_0.97",
            ),
        ],
        names=[
            "date",
            "model",
        ],
    )
    daily_losses = pd.DataFrame(
        {
            "QLIKE": [
                1.0,
                1.0,
                2.0,
            ],
        },
        index=index,
    )

    with pytest.raises(
        ValueError,
        match="duplicate date-model pairs",
    ):
        compare_qlike_losses(
            daily_losses,
            selected_ewma_name=(
                "ewma_lambda_0.97"
            ),
            alpha=0.05,
        )


def test_compare_qlike_losses_requires_all_models():
    dates = pd.date_range(
        "2025-01-01",
        periods=2,
        freq="D",
        tz="UTC",
        name="date",
    )
    wide_losses = pd.DataFrame(
        {
            "garch_1_1": [
                1.0,
                3.0,
            ],
            "ewma_lambda_0.97": [
                2.0,
                2.0,
            ],
            "persistence": [
                3.0,
                1.0,
            ],
        },
        index=dates,
    )
    wide_losses.columns.name = "model"

    daily_losses = (
        wide_losses.stack()
        .to_frame(name="QLIKE")
    )

    with pytest.raises(
        ValueError,
        match="missing required models",
    ):
        compare_qlike_losses(
            daily_losses,
            selected_ewma_name=(
                "ewma_lambda_0.97"
            ),
            alpha=0.05,
        )


def test_compare_qlike_losses_requires_chronological_dates():
    dates = pd.DatetimeIndex(
        [
            "2025-01-01",
            "2025-01-03",
            "2025-01-02",
            "2025-01-04",
            "2025-01-06",
            "2025-01-05",
        ],
        tz="UTC",
        name="date",
    )
    wide_losses = pd.DataFrame(
        {
            "persistence": [
                5.0,
                7.0,
                4.0,
                9.0,
                6.0,
                8.0,
            ],
            "historical_variance": [
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            "ewma_lambda_0.97": [
                3.0,
                6.0,
                2.0,
                8.0,
                4.0,
                7.0,
            ],
            "garch_1_1": [
                4.0,
                4.0,
                3.0,
                6.0,
                5.0,
                5.0,
            ],
        },
        index=dates,
    )
    wide_losses.columns.name = "model"

    daily_losses = (
        wide_losses.stack()
        .to_frame(name="QLIKE")
    )

    assert not daily_losses.xs(
        "garch_1_1",
        level="model",
    ).index.is_monotonic_increasing

    with pytest.raises(
        ValueError,
        match="chronological order",
    ):
        compare_qlike_losses(
            daily_losses,
            selected_ewma_name=(
                "ewma_lambda_0.97"
            ),
            alpha=0.05,
        )
