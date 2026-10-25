import json
from pathlib import Path

import pandas as pd
import requests


def data_fetch(symbol, interval, start, end):

    url = "https://data-api.binance.vision/api/v3/klines"

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
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        all_data.extend(batch)

        current_start = batch[-1][0] + 86_400_000

    return all_data


def save_raw_data(
    raw_data,
    symbol,
    interval,
    start,
    end,
    output_dir=Path("data/raw"),
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{symbol.lower()}_{interval}_{start}_{end}.json"
    )
    path = output_dir / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(raw_data, file, indent=2)

    return path
