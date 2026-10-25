from unittest.mock import Mock

import pandas as pd

import src.data_fetch as data_fetch_module
from src.data_fetch import data_fetch


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
        [start_ms + index * day_ms, f"day-{index + 1}"]
        for index in range(1000)
    ]
    second_batch = [
        [start_ms + 1000 * day_ms, "day-1001"],
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
        end="2027-01-01",
    )

    assert result == first_batch + second_batch
    assert get_mock.call_count == 2

    second_request = get_mock.call_args_list[1]
    second_start = second_request.kwargs["params"]["startTime"]

    assert second_start == start_ms + 999 * day_ms + 1


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
