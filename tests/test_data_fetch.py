import json
from unittest.mock import Mock

import pandas as pd

import src.data_fetch as data_fetch_module
from src.data_fetch import data_fetch, save_raw_data


def test_save_raw_data_writes_expected_json(tmp_path):
    raw_data = [
        [1, "100.0", "110.0"],
        [2, "110.0", "120.0"],
    ]

    saved_path = save_raw_data(
        raw_data=raw_data,
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        output_dir=tmp_path,
    )

    expected_path = tmp_path / "btcusdt_1d_2019-01-01_2026-01-01.json"

    assert saved_path == expected_path
    assert saved_path.exists()

    with saved_path.open("r", encoding="utf-8") as file:
        saved_data = json.load(file)

    assert saved_data == raw_data


def test_data_fetch_returns_single_batch(monkeypatch):
    start_ms = int(
        pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000
    )

    batch = [
        [start_ms, "100.0"],
    ]

    response = Mock()
    response.json.return_value = batch

    get_mock = Mock(return_value=response)

    monkeypatch.setattr(
        data_fetch_module.requests,
        "get",
        get_mock,
    )

    result = data_fetch(
        symbol="BTCUSDT",
        interval="1d",
        start="2024-01-01",
        end="2024-01-02",
    )

    assert result == batch
    get_mock.assert_called_once()
    response.raise_for_status.assert_called_once_with()


def test_data_fetch_combines_multiple_batches(monkeypatch):
    day_ms = 86_400_000
    start_ms = int(
        pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000
    )

    first_batch = [
        [start_ms, "day-1"],
        [start_ms + day_ms, "day-2"],
    ]
    second_batch = [
        [start_ms + 2 * day_ms, "day-3"],
    ]

    first_response = Mock()
    first_response.json.return_value = first_batch

    second_response = Mock()
    second_response.json.return_value = second_batch

    get_mock = Mock(
        side_effect=[first_response, second_response]
    )

    monkeypatch.setattr(
        data_fetch_module.requests,
        "get",
        get_mock,
    )

    result = data_fetch(
        symbol="BTCUSDT",
        interval="1d",
        start="2024-01-01",
        end="2024-01-04",
    )

    assert result == first_batch + second_batch
    assert get_mock.call_count == 2

    second_request = get_mock.call_args_list[1]
    second_start = second_request.kwargs["params"]["startTime"]

    assert second_start == start_ms + 2 * day_ms


def test_data_fetch_stops_when_batch_is_empty(monkeypatch):
    response = Mock()
    response.json.return_value = []

    get_mock = Mock(return_value=response)

    monkeypatch.setattr(
        data_fetch_module.requests,
        "get",
        get_mock,
    )

    result = data_fetch(
        symbol="BTCUSDT",
        interval="1d",
        start="2024-01-01",
        end="2024-01-03",
    )

    assert result == []
    get_mock.assert_called_once()
    response.raise_for_status.assert_called_once_with()