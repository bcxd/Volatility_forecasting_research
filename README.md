# Volatility Forecasting Research

This project compares one-day-ahead BTCUSDT variance forecasts from persistence, expanding historical variance, EWMA, and GARCH(1,1).

## Data workflow

Data acquisition and experiment execution are separate actions.

Download a new immutable raw snapshot only when the requested sample changes:

~~~powershell
python -m src.fetch_data `
    --symbol BTCUSDT `
    --interval 1d `
    --start 2019-01-01 `
    --end 2026-01-01
~~~

The end date is exclusive. This command creates a JSON data file and an adjacent `.metadata.json` file containing the request, retrieval time, row count, and SHA-256 checksum. It refuses to overwrite an existing snapshot.

Run the experiment from that explicit snapshot:

~~~powershell
python -m src.pipeline `
    --raw-path data/raw/btcusdt_1d_2019-01-01_2026-01-01.json
~~~

The experiment verifies the metadata and checksum before processing. It does not call Binance and does not modify the raw snapshot.

## Tests

~~~powershell
pytest -q
~~~
