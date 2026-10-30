from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.models as models
from src.models import (
    build_garch_constraints,
    calculate_ewma,
    calculate_garch_variance,
    forecast_garch,
    minimize_likelihood,
    negative_log_likelihood,
    valid_garch_params,
)


def test_calculate_ewma_uses_only_available_history():
    train_dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    train_df = pd.DataFrame(
        {"variance_proxy": [1.0, 3.0]},
        index=train_dates,
    )

    forecast_dates = pd.date_range("2024-01-03", periods=2, tz="UTC")
    forecast_df = pd.DataFrame(
        {"variance_proxy": [4.0, 9.0]},
        index=forecast_dates,
    )

    result = calculate_ewma(
        forecast_df=forecast_df,
        train_df=train_df,
        lamda_=0.5,
    )

    np.testing.assert_allclose(result.to_numpy(), [2.0, 3.0])
    assert result.index.equals(forecast_dates)
    assert result.name == "ewma_variance"


@pytest.mark.parametrize(
    ("lamda_", "expected"),
    [
        (0.0, [2.0, 4.0]),
        (1.0, [2.0, 2.0]),
    ],
)
def test_calculate_ewma_lambda_boundaries(lamda_, expected):
    train_dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    train_df = pd.DataFrame(
        {"variance_proxy": [1.0, 3.0]},
        index=train_dates,
    )

    forecast_dates = pd.date_range("2024-01-03", periods=2, tz="UTC")
    forecast_df = pd.DataFrame(
        {"variance_proxy": [4.0, 9.0]},
        index=forecast_dates,
    )

    result = calculate_ewma(
        forecast_df=forecast_df,
        train_df=train_df,
        lamda_=lamda_,
    )

    np.testing.assert_allclose(result.to_numpy(), expected)


@pytest.mark.parametrize("invalid_lamda", [-0.1, 1.1])
def test_calculate_ewma_rejects_lambda_outside_unit_interval(invalid_lamda):
    train_df = pd.DataFrame(
        {"variance_proxy": [1.0]},
        index=pd.to_datetime(["2024-01-01"], utc=True),
    )

    forecast_df = pd.DataFrame(
        {"variance_proxy": [2.0]},
        index=pd.to_datetime(["2024-01-02"], utc=True),
    )

    with pytest.raises(ValueError, match="0 <= lamda <= 1"):
        calculate_ewma(
            forecast_df=forecast_df,
            train_df=train_df,
            lamda_=invalid_lamda,
        )


def test_calculate_garch_variance_follows_garch_1_1_recursion():
    params = np.array([0.1, 0.2, 0.5])
    returns = np.array([1.0, 2.0])

    result = calculate_garch_variance(
        params=params,
        train_returns=returns,
    )

    np.testing.assert_allclose(result, [2.5, 1.55])


def test_valid_garch_params_accepts_stationary_parameters():
    assert valid_garch_params([0.1, 0.2, 0.7])


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ([0.0, 0.1, 0.8], "omega is zero"),
        ([0.1, -0.1, 0.8], "alpha is negative"),
        ([0.1, 0.1, -0.2], "beta is negative"),
        ([0.1, 0.4, 0.6], "alpha plus beta is one"),
        ([0.1, np.nan, 0.5], "parameter is not finite"),
        ([0.1, 0.2], "wrong parameter count"),
    ],
)
def test_valid_garch_params_rejects_invalid_parameters(params, reason):
    assert not valid_garch_params(params), reason


def test_negative_log_likelihood_penalizes_invalid_parameters():
    invalid_params = np.array([0.1, 0.6, 0.5])
    returns = np.array([0.01, -0.02, 0.015])

    result = negative_log_likelihood(
        params=invalid_params,
        train_returns=returns,
    )

    assert result == 1e100


def test_negative_log_likelihood_matches_known_value():
    params = np.array([0.1, 0.2, 0.5])
    returns = np.array([1.0, 2.0])

    expected = 0.5 * (
        np.log(2 * np.pi)
        + np.log(2.5)
        + 1.0 ** 2 / 2.5
        + np.log(2 * np.pi)
        + np.log(1.55)
        + 2.0 ** 2 / 1.55
    )

    result = negative_log_likelihood(
        params=params,
        train_returns=returns,
    )

    np.testing.assert_allclose(result, expected)


def test_forecast_garch_uses_latest_shock_and_variance():
    params = np.array([0.1, 0.2, 0.5])
    returns = np.array([1.0, 2.0])

    # omega + alpha × r[-1]² + beta × h[-1]
    # = 0.1 + 0.2 × 2² + 0.5 × 1.55
    # = 1.675

    result = forecast_garch(
        params=params,
        forecast_returns=returns,
    )

    np.testing.assert_allclose(result, 1.675)


def test_minimize_likelihood_returns_valid_finite_fit():
    returns = np.tile(
        np.array([0.01, -0.02, 0.015, -0.01, 0.025, -0.005]),
        10,
    )

    bounds = (
        (1e-12, None),
        (0.0, 1.0),
        (0.0, 1.0),
    )

    constraints = (
        {
            "type": "ineq",
            "fun": lambda params: 1.0 - params[1] - params[2] - 1e-8,
        },
    )

    result = minimize_likelihood(
        train_returns=returns,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    assert result.success
    assert result.x.shape == (3,)
    assert np.isfinite(result.x).all()
    assert np.isfinite(result.fun)
    assert valid_garch_params(result.x)


def test_walk_forward_garch_uses_only_past_returns(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")

    df = pd.DataFrame(
        {"log_return": [1.0, 2.0, 3.0, 999.0]},
        index=dates,
    )

    forecast_dates = dates[2:]

    def fake_minimize_likelihood(**kwargs):
        return SimpleNamespace(x=np.array([0.1, 0.2, 0.5]))

    def fake_forecast_garch(params, forecast_returns):
        return 2.0

    monkeypatch.setattr(
        models,
        "minimize_likelihood",
        fake_minimize_likelihood,
    )
    monkeypatch.setattr(
        models,
        "forecast_garch",
        fake_forecast_garch,
    )

    result = models.walk_forward_garch(
        df=df,
        forecast_dates=forecast_dates,
        method="SLSQP",
        bounds=(),
        constraints=(),
        refit_every=7,
    )

    np.testing.assert_allclose(result.to_numpy(), [2.0, 2.9])
    assert result.index.equals(forecast_dates)
    assert result.name == "garch_variance"


def test_walk_forward_garch_refits_on_schedule(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")

    df = pd.DataFrame(
        {"log_return": np.full(10, 0.1)},
        index=dates,
    )

    forecast_dates = dates[2:]

    fit_history_lengths = []

    def fake_minimize_likelihood(train_returns, **kwargs):
        fit_history_lengths.append(len(train_returns))
        return SimpleNamespace(x=np.array([0.1, 0.2, 0.5]))

    def fake_forecast_garch(params, forecast_returns):
        return 1.0

    monkeypatch.setattr(
        models,
        "minimize_likelihood",
        fake_minimize_likelihood,
    )
    monkeypatch.setattr(
        models,
        "forecast_garch",
        fake_forecast_garch,
    )

    models.walk_forward_garch(
        df=df,
        forecast_dates=forecast_dates,
        method="SLSQP",
        bounds=(),
        constraints=(),
        refit_every=7,
    )

    assert fit_history_lengths == [2, 9]


def test_build_garch_constraints_enforces_persistence_margin():
    margin = 1e-8

    constraints = build_garch_constraints(
        persistence_margin=margin,
    )

    assert len(constraints) == 1

    constraint = constraints[0]

    assert constraint["type"] == "ineq"

    valid_params = np.array([0.1, 0.2, 0.7])
    invalid_params = np.array([0.1, 0.6, 0.5])

    assert constraint["fun"](valid_params) > 0
    assert constraint["fun"](invalid_params) < 0


@pytest.mark.parametrize(
    "invalid_margin",
    [
        0.0,
        -1e-8,
        1.0,
        np.inf,
        np.nan,
    ],
)
def test_build_garch_constraints_rejects_invalid_margin(
    invalid_margin,
):
    with pytest.raises(
        ValueError,
        match="persistence_margin",
    ):
        build_garch_constraints(
            persistence_margin=invalid_margin,
        )