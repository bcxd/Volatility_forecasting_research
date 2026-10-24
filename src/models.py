# EWMA และ GARCH
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_ewma(forecast_df, train_df, lamda_):
    if train_df.empty:
        raise ValueError("train_df cannot be empty")

    if "variance_proxy" not in train_df.columns:
        raise ValueError("train_df requires a 'variance_proxy' column")

    if "variance_proxy" not in forecast_df.columns:
        raise ValueError("forecast_df requires a 'variance_proxy' column")

    if train_df["variance_proxy"].isna().any():
        raise ValueError("Found missing value in train_df")

    if forecast_df["variance_proxy"].isna().any():
        raise ValueError("Found missing value in forecast_df")

    train_values = train_df["variance_proxy"].to_numpy(dtype=float)
    forecast_values = forecast_df["variance_proxy"].to_numpy(dtype=float)

    if not np.isfinite(train_values).all():
        raise ValueError("train_df variance_proxy must contain finite numbers")

    if not np.isfinite(forecast_values).all():
        raise ValueError("forecast_df variance_proxy must contain finite numbers")

    if (train_values < 0).any() or (forecast_values < 0).any():
        raise ValueError("variance_proxy cannot be negative")

    if not train_df.index.is_monotonic_increasing:
        raise ValueError("train_df must be ordered by date")

    if not forecast_df.index.is_monotonic_increasing:
        raise ValueError("forecast_df must be ordered by date")

    if train_df.index.max() >= forecast_df.index.min():
        raise ValueError("forecast_df must start after train_df ends")

    if not (0 <= lamda_ <= 1):
        raise ValueError("Require 0 <= lamda <= 1")

    forecasts = []
    initial_variance = train_df["variance_proxy"].mean()
    h = initial_variance

    for squared_return in forecast_df["variance_proxy"]:
        forecasts.append(h)
        h = lamda_ * h + (1 - lamda_) * squared_return

    return pd.Series(
        forecasts,
        index=forecast_df.index,
        name="ewma_variance",
    )


def calculate_garch_variance(params, train_returns):
    params = np.asarray(params, dtype=float)
    if not params.shape == (3,):
        raise ValueError("params must contain exactly 3 values")

    train_returns = np.asarray(train_returns, dtype=float)
    omega, alpha, beta = params

    if train_returns.ndim != 1 or len(train_returns) == 0:
        raise ValueError("Returns must be a non-empty 1D array")

    if not np.isfinite(train_returns).all():
        raise ValueError("Returns must contain only finite values")

    params = np.array([omega, alpha, beta], dtype=float)

    if not np.isfinite(params).all():
        raise ValueError("Parameters must be finite")

    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError(
            "Require omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1"
        )

    h = np.empty(len(train_returns))

    h[0] = max(np.mean(train_returns ** 2), 1e-12)

    for t in range(1, len(train_returns)):
        h[t] = (
            omega
            + alpha * train_returns[t - 1] ** 2
            + beta * h[t - 1]
        )

    return h


def valid_garch_params(params):
    params = np.asarray(params, dtype=float)

    return (
        params.shape == (3,)
        and np.isfinite(params).all()
        and params[0] > 0
        and params[1] >= 0
        and params[2] >= 0
        and params[1] + params[2] < 1
    )


def negative_log_likelihood(params, train_returns):
    train_returns = np.asarray(train_returns, dtype=float)

    if not valid_garch_params(params):
        return 1e100

    h = calculate_garch_variance(params, train_returns)

    if not np.isfinite(h).all() or (h <= 0).any():
        return 1e100

    likelihood = 0.5 * np.sum(
        np.log(2 * np.pi)
        + np.log(h)
        + train_returns ** 2 / h
    )

    if not np.isfinite(likelihood):
        return 1e100

    return likelihood


def minimize_likelihood(
    train_returns,
    method,
    bounds,
    constraints,
):
    train_returns = np.asarray(train_returns, dtype=float)

    if train_returns.ndim != 1 or len(train_returns) == 0:
        raise ValueError("Returns must be a non-empty 1D array")

    if not np.isfinite(train_returns).all():
        raise ValueError("Returns must contain only finite values")

    variance_scale = max(float(np.mean(train_returns ** 2)), 1e-12)

    def scaled_negative_log_likelihood(scaled_params):
        params = np.array(
            [
                scaled_params[0] * variance_scale,
                scaled_params[1],
                scaled_params[2],
            ],
            dtype=float,
        )
        return negative_log_likelihood(params, train_returns) / len(train_returns)

    scaled_bounds = list(bounds)
    omega_lower, omega_upper = scaled_bounds[0]
    scaled_bounds[0] = (
        None if omega_lower is None else omega_lower / variance_scale,
        None if omega_upper is None else omega_upper / variance_scale,
    )

    result = minimize(
        scaled_negative_log_likelihood,
        x0=[0.05, 0.10, 0.85],
        method=method,
        bounds=scaled_bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"GARCH fitting failed: {result.message}")

    if not np.isfinite(result.x).all():
        raise RuntimeError("GARCH fitting returned non-finite parameters")

    result.x = np.asarray(result.x, dtype=float).copy()
    result.x[0] *= variance_scale

    if not valid_garch_params(result.x):
        raise RuntimeError("GARCH fitting returned invalid parameters")

    result.fun = negative_log_likelihood(result.x, train_returns)
    if not np.isfinite(result.fun):
        raise RuntimeError("GARCH fitting returned a non-finite likelihood")

    return result


def forecast_garch(params, forecast_returns):
    forecast_returns = np.asarray(forecast_returns, dtype=float)
    omega, alpha, beta = params
    h = calculate_garch_variance(params, forecast_returns)

    forecast = (
        omega
        + alpha * forecast_returns[-1] ** 2
        + beta * h[-1]
    )
    return forecast


def walk_forward_garch(
    df,
    forecast_dates,
    method,
    bounds,
    constraints,
    refit_every,
):
    forecasts = []
    variance_state = None
    params = None
    for i, forecast_date in enumerate(forecast_dates):
        history = df.loc[df.index < forecast_date, "log_return"]
        forecast_returns = history.to_numpy()

        if i % refit_every == 0:
            result = minimize_likelihood(
                train_returns=history.to_numpy(),
                method=method,
                bounds=bounds,
                constraints=constraints,
            )

            params = result.x
            variance_state = forecast_garch(params, forecast_returns)
        else:
            omega, alpha, beta = params

            variance_state = (
                omega
                + alpha * forecast_returns[-1] ** 2
                + beta * variance_state
            )

        forecasts.append(variance_state)

    return pd.Series(
        forecasts,
        index=forecast_dates,
        name="garch_variance",
    )
