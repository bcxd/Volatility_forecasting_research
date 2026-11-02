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


def evaluate_forecasts(actual, forecasts, floor):
    if not isinstance(actual, pd.Series):
        raise TypeError("actual must be a pandas Series")

    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError("forecasts must be a pandas DataFrame")

    if forecasts.empty:
        raise ValueError("forecasts cannot be empty")

    if not forecasts.columns.is_unique:
        raise ValueError("forecast model names must be unique")

    if not actual.index.equals(forecasts.index):
        raise ValueError("actual and forecasts must have matching dates")

    adjusted_forecasts = forecasts.copy()
    daily_losses_by_model = {}
    metrics_by_model = {}

    for model_name in adjusted_forecasts.columns:
        adjusted_forecast = apply_variance_floor(
            adjusted_forecasts[model_name],
            floor=floor,
        )

        adjusted_forecasts[model_name] = adjusted_forecast

        qlike = qlike_loss(actual, adjusted_forecast)
        mse = mse_loss(actual, adjusted_forecast)
        mae = mae_loss(actual, adjusted_forecast)

        daily_losses_by_model[model_name] = pd.DataFrame(
            {
                "QLIKE": qlike,
                "MSE": mse,
                "MAE": mae,
            },
            index=actual.index,
        )

        metrics_by_model[model_name] = {
            "QLIKE": qlike.mean(),
            "MSE": mse.mean(),
            "MAE": mae.mean(),
        }

    metrics = pd.DataFrame.from_dict(
        metrics_by_model,
        orient="index",
    )

    daily_losses = pd.concat(
        daily_losses_by_model,
        names=[
            "model",
            "date",
        ],
    )

    daily_losses = daily_losses.reorder_levels(
        [
            "date",
            "model",
        ]
    ).sort_index()

    return metrics, daily_losses, adjusted_forecasts
