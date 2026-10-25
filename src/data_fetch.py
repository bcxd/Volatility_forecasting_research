import pandas as pd
import requests


BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


def data_fetch(symbol, interval, start, end):
    # Binance API expects Unix timestamps.
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    # Binance limits each response to 1,000 rows.
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
        response = requests.get(
            BINANCE_KLINES_URL,
            params=params,
            timeout=15,
        )
        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        all_data.extend(batch)

        if len(batch) < 1000:
            break

        # Start just after the last returned candle. Binance will return the
        # next available candle for any interval without duplicating a row.
        current_start = batch[-1][0] + 1

    return all_data
