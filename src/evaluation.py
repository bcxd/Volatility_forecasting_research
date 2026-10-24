# walk-forward และ metrics
import numpy as np
import pandas as pd


def apply_variance_floor(forecast, floor):
    if not isinstance(forecast, pd.Series):
        raise TypeError("forecast must be a pandas Series")

    if forecast.empty:
        raise ValueError("forecast cannot be empty")

    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be a finite positive number")

    values = forecast.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("forecast must contain only finite values")

    if (values < 0).any():
        raise ValueError("forecast variance cannot be negative")

    return forecast.clip(lower=floor)


def _validate_inputs(actual, forecast):
    if not isinstance(actual, pd.Series) or not isinstance(forecast, pd.Series):
        raise TypeError("actual and forecast must be pandas Series")

    if actual.empty or forecast.empty:
        raise ValueError("actual and forecast cannot be empty")

    if not actual.index.equals(forecast.index):
        raise ValueError("actual and forecast must have matching dates")

    actual_values = actual.to_numpy(dtype=float)
    forecast_values = forecast.to_numpy(dtype=float)

    if not np.isfinite(actual_values).all():
        raise ValueError("actual must contain only finite values")

    if not np.isfinite(forecast_values).all():
        raise ValueError("forecast must contain only finite values")

    if (actual_values < 0).any():
        raise ValueError("actual variance cannot be negative")

    if (forecast_values <= 0).any():
        raise ValueError(
            "forecast variance must be positive; apply the variance floor first"
        )

    return actual_values, forecast_values


def qlike_loss(actual, forecast):
    actual_values, forecast_values = _validate_inputs(actual, forecast)

    losses = np.log(forecast_values) + actual_values / forecast_values

    return pd.Series(losses, index=actual.index, name="qlike_loss")


def mse_loss(actual, forecast):
    actual_values, forecast_values = _validate_inputs(actual, forecast)

    losses = (actual_values - forecast_values) ** 2

    return pd.Series(losses, index=actual.index, name="mse_loss")


def mae_loss(actual, forecast):
    actual_values, forecast_values = _validate_inputs(actual, forecast)

    losses = np.abs(actual_values - forecast_values)

    return pd.Series(losses, index=actual.index, name="mae_loss")
