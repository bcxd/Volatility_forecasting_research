from unittest.mock import Mock

import src.fetch_data as fetch_data_module


def test_acquire_snapshot_fetches_then_saves(monkeypatch, tmp_path):
    raw_data = [[1704067200000, "100.0"]]
    expected_path = tmp_path / "snapshot.json"

    fetch_mock = Mock(return_value=raw_data)
    save_mock = Mock(return_value=expected_path)

    monkeypatch.setattr(
        fetch_data_module,
        "data_fetch",
        fetch_mock,
    )
    monkeypatch.setattr(
        fetch_data_module,
        "save_raw_data",
        save_mock,
    )

    result = fetch_data_module.acquire_snapshot(
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        output_dir=tmp_path,
    )

    fetch_mock.assert_called_once_with(
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
    )
    save_mock.assert_called_once_with(
        raw_data=raw_data,
        endpoint=fetch_data_module.BINANCE_KLINES_URL,
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        output_dir=tmp_path,
    )
    assert result == expected_path


def test_main_parses_arguments_and_acquires_snapshot(
    monkeypatch,
    tmp_path,
):
    expected_path = tmp_path / "snapshot.json"
    acquire_mock = Mock(return_value=expected_path)

    monkeypatch.setattr(
        fetch_data_module,
        "acquire_snapshot",
        acquire_mock,
    )

    result = fetch_data_module.main(
        [
            "--symbol",
            "ETHUSDT",
            "--interval",
            "1d",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-01",
            "--output-dir",
            str(tmp_path),
        ]
    )

    acquire_mock.assert_called_once_with(
        symbol="ETHUSDT",
        interval="1d",
        start="2024-01-01",
        end="2024-02-01",
        output_dir=tmp_path,
    )
    assert result == expected_path