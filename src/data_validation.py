import pandas as pd
import numpy as np
from pathlib import Path 

def data_validation(raw_data):
    if not raw_data:
        raise ValueError("No data received")
        
    if not isinstance(raw_data, list):
        raise ValueError("Raw data must be a list of candles")

    if any(not isinstance(row, (list, tuple)) or len(row) != 12 for row in raw_data):
        raise ValueError("Each row must contain exactly 12 values")
    
    df = pd.DataFrame(raw_data)
        
    if df.shape[1] != 12:
        raise ValueError("Data must have exactly 12 columns")
        
    df.columns = [
        "open_time", "open", "high",
        "low", "close", "volume",
        "close_time", "quote_volume",
        "trade_count", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    
    ohlcv_columns = ["open", "high", "low", "close", "volume"]
     
    if df[ohlcv_columns].isna().any().any():
        raise ValueError(f"Found missing value in OHLCV")
    
    if df[["open_time", "close_time"]].isna().any().any():
        raise ValueError("Found missing timestamp")
        
    df["open_time"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True, errors="raise"
    )
    df["close_time"] = pd.to_datetime(
        df["close_time"], unit="ms", utc=True, errors="raise"
    )

    if df[["open_time", "close_time"]].isna().any().any():
        raise ValueError("Found invalid timestamp")
    
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    
    if df["open_time"].duplicated().any():
        raise ValueError(f"Found duplicate open_time")
    if df["close_time"].duplicated().any():
        raise ValueError(f"Found duplicate close_time")
        
    df = df.sort_values("open_time")
    df = df.set_index("open_time")
    
    expected_dates = pd.date_range(
        start="2019-01-01",
        end="2025-12-31",
        freq="D",
        tz="UTC",
    )
    missing_dates =  expected_dates.difference(df.index)
    if len(missing_dates) > 0:
        raise ValueError(f"Found missing dates: {missing_dates.tolist()}")
   
    unexpected_dates = df.index.difference(expected_dates)
    if len(unexpected_dates) > 0:
        raise ValueError(
            f"Found unexpected dates: {unexpected_dates.tolist()}"
        )
        
    # For Binance daily data. Daily candles close one millisecond before midnight.
    expected_close_time = (
        df.index
        + pd.Timedelta(days=1)
        - pd.Timedelta(milliseconds=1)
    )

    if (df["close_time"] != expected_close_time).any():
        raise ValueError("Invalid close_time for daily candle")    

    if (df["close_time"] >= pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("Found candle that has not closed yet")
        
    prices = df[["open", "high", "low", "close"]]
    checks = [
        (not np.isfinite(df[numeric_columns]).all().all(), "OHLCV must be finite"),
        ((prices <= 0).any().any(), "Prices must be positive"),
        ((df["volume"] < 0).any(), "Volume must be non-negative"),
        ((df["high"] < prices.max(axis=1)).any(), "High must be the highest price"),
        ((df["low"] > prices.min(axis=1)).any(), "Low must be the lowest price"),
    ]
    for invalid, message in checks:
        if invalid:
            raise ValueError(message)
    
    return df

def save_processed_data(df):
    path = Path("data/processed/btcusdt_1d.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=True, encoding="utf-8")
        
