import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    apply_variance_floor,
    evaluate_forecasts,
    mae_loss,
    mse_loss,
    qlike_loss,
)


def test_qlike_loss_matches_known_values():
    dates = pd.date_range(
        start="2024-01-01",
        periods=2,
        freq="D",
        tz="UTC",
    )

    actual = pd.Series([1.0, 4.0], index=dates)
    forecast = pd.Series([1.0, 2.0], index=dates)

    result = qlike_loss(actual, forecast)

    expected = np.array(
        [
            1.0,
            np.log(2.0) + 2.0,
        ]
    )

    np.testing.assert_allclose(
        result.to_numpy(),
        expected,
    )
    assert result.index.equals(dates)


def test_mse_loss_matches_known_values():
    dates = pd.date_range(
        start="2024-01-01",
        periods=2,
        freq="D",
        tz="UTC",
    )

    actual = pd.Series([1.0, 4.0], index=dates)
    forecast = pd.Series([1.0, 2.0], index=dates)

    result = mse_loss(actual, forecast)

    expected = np.array(
        [
            (1.0 - 1.0) ** 2,
            (4.0 - 2.0) ** 2,
        ]
    )

    np.testing.assert_allclose(
        result.to_numpy(),
        expected,
    )
    assert result.index.equals(dates)


def test_mae_loss_matches_known_values():
    dates = pd.date_range(
        start="2024-01-01",
        periods=2,
        freq="D",
        tz="UTC",
    )

    actual = pd.Series([1.0, 4.0], index=dates)
    forecast = pd.Series([1.0, 2.0], index=dates)

    result = mae_loss(actual, forecast)

    expected = np.array(
        [
            np.abs(1.0 - 1.0),
            np.abs(4.0 - 2.0),
        ]
    )

    np.testing.assert_allclose(
        result.to_numpy(),
        expected,
    )
    assert result.index.equals(dates)


def test_apply_variance_floor_clips_small_values():
    forecast = pd.Series([0.0, 1e-14, 1e-6])
    floor = 1e-12

    result = apply_variance_floor(forecast, floor)

    expected = np.array([1e-12, 1e-12, 1e-6])

    np.testing.assert_allclose(
        result.to_numpy(),
        expected,
    )
    assert result.index.equals(forecast.index)
    assert forecast.iloc[0] == 0.0


def test_qlike_loss_rejects_mismatched_dates():
    actual = pd.Series([1.0, 4.0], index=["2024-01-01", "2024-01-02"])
    forecast = pd.Series([1.0, 2.0], index=["2024-01-02", "2024-01-03"])

    with pytest.raises(ValueError, match="matching dates"):
        qlike_loss(actual, forecast)


@pytest.mark.parametrize("invalid_forecast", [0.0, -0.1])
def test_qlike_loss_rejects_nonpositive_forecast(invalid_forecast):
    dates = pd.date_range(
        start="2024-01-01",
        periods=1,
        freq="D",
        tz="UTC",
    )

    actual = pd.Series([1.0], index=dates)
    forecast = pd.Series([invalid_forecast], index=dates)

    with pytest.raises(ValueError, match="must be positive"):
        qlike_loss(actual, forecast)


def test_evaluate_forecasts_returns_metrics_and_daily_losses():
    dates = pd.date_range(
        "2024-01-01",
        periods=2,
        freq="D",
        tz="UTC",
    )

    actual = pd.Series(
        [1.0, 4.0],
        index=dates,
    )
    forecasts = pd.DataFrame(
        {
            "model_a": [1.0, 2.0],
            "model_b": [2.0, 4.0],
        },
        index=dates,
    )

    (
        metrics,
        daily_losses,
        adjusted,
    ) = evaluate_forecasts(
        actual=actual,
        forecasts=forecasts,
        floor=1e-12,
    )

    assert metrics.index.tolist() == [
        "model_a",
        "model_b",
    ]
    assert metrics.columns.tolist() == [
        "QLIKE",
        "MSE",
        "MAE",
    ]

    assert daily_losses.index.names == [
        "date",
        "model",
    ]
    assert daily_losses.columns.tolist() == [
        "QLIKE",
        "MSE",
        "MAE",
    ]

    model_a_losses = daily_losses.xs(
        "model_a",
        level="model",
    )

    assert model_a_losses.index.equals(dates)

    expected_qlike = np.array(
        [
            1.0,
            np.log(2.0) + 2.0,
        ]
    )

    np.testing.assert_allclose(
        model_a_losses["QLIKE"].to_numpy(),
        expected_qlike,
    )
    np.testing.assert_allclose(
        model_a_losses["MSE"].to_numpy(),
        [0.0, 4.0],
    )
    np.testing.assert_allclose(
        model_a_losses["MAE"].to_numpy(),
        [0.0, 2.0],
    )

    np.testing.assert_allclose(
        metrics.loc["model_a"].to_numpy(),
        [
            expected_qlike.mean(),
            2.0,
            1.0,
        ],
    )

    pd.testing.assert_frame_equal(
        adjusted,
        forecasts,
    )
