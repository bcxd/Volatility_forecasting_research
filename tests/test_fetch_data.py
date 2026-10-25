from pathlib import Path
from unittest.mock import Mock

import src.fetch_data as fetch_data_module


def test_main_fetches_and_saves_requested_snapshot(monkeypatch, tmp_path):
    raw_data = [[1704067200000, "100.0"]]
    data_path = tmp_path / "snapshot.json"
    metadata_path = tmp_path / "snapshot.metadata.json"

    fetch_mock = Mock(return_value=raw_data)
    save_mock = Mock(return_value=(data_path, metadata_path))
    monkeypatch.setattr(fetch_data_module, "data_fetch", fetch_mock)
    monkeypatch.setattr(fetch_data_module, "save_raw_snapshot", save_mock)

    result = fetch_data_module.main(
        [
            "--symbol",
            "ETHUSDT",
            "--interval",
            "1d",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-03",
            "--output-dir",
            str(tmp_path),
        ]
    )

    fetch_mock.assert_called_once_with(
        symbol="ETHUSDT",
        interval="1d",
        start="2024-01-01",
        end="2024-01-03",
    )
    save_mock.assert_called_once_with(
        raw_data=raw_data,
        endpoint=fetch_data_module.BINANCE_KLINES_URL,
        symbol="ETHUSDT",
        interval="1d",
        start="2024-01-01",
        end="2024-01-03",
        output_dir=Path(tmp_path),
    )
    assert result == (data_path, metadata_path)
