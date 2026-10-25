import hashlib
import json
from datetime import datetime, timezone

import pytest

from src.data_io import (
    load_raw_snapshot,
    save_raw_snapshot,
    validate_snapshot_metadata,
)


RAW_DATA = [
    [1704067200000, "100.0"],
    [1704153600000, "110.0"],
]


def save_example_snapshot(tmp_path):
    return save_raw_snapshot(
        raw_data=RAW_DATA,
        endpoint="https://example.test/api/v3/klines",
        symbol="BTCUSDT",
        interval="1d",
        start="2024-01-01",
        end="2024-01-03",
        output_dir=tmp_path,
        retrieved_at=datetime(2024, 1, 4, tzinfo=timezone.utc),
    )


def test_save_raw_snapshot_writes_data_and_metadata(tmp_path):
    data_path, metadata_path = save_example_snapshot(tmp_path)

    assert data_path.name == "btcusdt_1d_2024-01-01_2024-01-03.json"
    assert metadata_path.name == (
        "btcusdt_1d_2024-01-01_2024-01-03.metadata.json"
    )

    raw_bytes = data_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert json.loads(raw_bytes) == RAW_DATA
    assert metadata["api_endpoint"] == "https://example.test/api/v3/klines"
    assert metadata["symbol"] == "BTCUSDT"
    assert metadata["interval"] == "1d"
    assert metadata["requested_start"] == "2024-01-01"
    assert metadata["requested_end_exclusive"] == "2024-01-03"
    assert metadata["retrieved_at_utc"] == "2024-01-04T00:00:00Z"
    assert metadata["first_candle_open_utc"] == "2024-01-01T00:00:00Z"
    assert metadata["last_candle_open_utc"] == "2024-01-02T00:00:00Z"
    assert metadata["row_count"] == 2
    assert metadata["data_filename"] == data_path.name
    assert metadata["sha256"] == hashlib.sha256(raw_bytes).hexdigest()


def test_save_raw_snapshot_refuses_to_overwrite(tmp_path):
    save_example_snapshot(tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        save_example_snapshot(tmp_path)


def test_load_raw_snapshot_verifies_checksum(tmp_path):
    data_path, _ = save_example_snapshot(tmp_path)
    original_bytes = data_path.read_bytes()

    raw_data, metadata = load_raw_snapshot(data_path)
    second_raw_data, second_metadata = load_raw_snapshot(data_path)

    assert raw_data == RAW_DATA
    assert metadata["row_count"] == len(RAW_DATA)
    assert second_raw_data == raw_data
    assert second_metadata == metadata
    assert data_path.read_bytes() == original_bytes

    data_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_raw_snapshot(data_path)


def test_load_raw_snapshot_requires_metadata(tmp_path):
    data_path = tmp_path / "legacy.json"
    data_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="metadata"):
        load_raw_snapshot(data_path)


def test_validate_snapshot_metadata_rejects_wrong_request(tmp_path):
    data_path, _ = save_example_snapshot(tmp_path)
    _, metadata = load_raw_snapshot(data_path)

    with pytest.raises(ValueError, match="symbol"):
        validate_snapshot_metadata(
            metadata,
            symbol="ETHUSDT",
            interval="1d",
            start="2024-01-01",
            end="2024-01-03",
        )
