import requests
import json
import pandas as pd
from pathlib import Path 

def data_fetch(symbol, interval, start, end):

    url = "https://data-api.binance.vision/api/v3/klines"
    
    #Binance API want Unix timestamp
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    
    #Fetching data has 1000 limit per time
    all_data = []
    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms - 1,
            "limit": 1000,
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        all_data.extend(batch)

        current_start = batch[-1][0] + 86_400_000
    
    return all_data

def save_raw_data(raw_data):
    path = Path("data/raw/btcusdt_1d_2019-01-01_2025-12-31.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
