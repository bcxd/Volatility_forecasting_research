 # EWMA และ GARCH
import pandas as pd
import numpy as np
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
    h =  initial_variance

    for squared_return in forecast_df["variance_proxy"]:
        forecasts.append(h)  
        h = lamda_ * h + (1 - lamda_) * squared_return

    return pd.Series(
        forecasts,
        index=forecast_df.index,
        name="ewma_variance",
    )

