import pandas as pd
from src.data_fetch import data_fetch, save_raw_data
from src.data_validation import data_validation, save_processed_data
from src.features import calculate_log_returns, variance_proxy

def main():
    raw_data = data_fetch(
            symbol="BTCUSDT",
            interval="1d",
            start="2019-01-01",
            end="2026-01-01",
    )
    save_raw_data(raw_data)
    processed_data = data_validation(raw_data)
    save_processed_data(processed_data)
    
    df = df = processed_data.copy()
    df = calculate_log_returns(df)
    df = variance_proxy(df)
    df = df.dropna(subset=["log_return", "variance_proxy"])


if __name__ == "__main__":
    main()