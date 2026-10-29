import hashlib
import json
from datetime import datetime, timezone

import pytest

from src.data_io import (
    load_raw_snapshot,
    save_raw_data,
)

TEST_ENDPOINT = (
    "https://data-api.binance.vision/api/v3/klines"
)


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
        endpoint=TEST_ENDPOINT,
    )

    expected_path = tmp_path / "btcusdt_1d_2019-01-01_2026-01-01.json"

    assert saved_path == expected_path
    assert saved_path.exists()

    with saved_path.open("r", encoding="utf-8") as file:
        saved_data = json.load(file)

    assert saved_data == raw_data


def test_save_raw_data_refuses_to_overwrite_existing_file(tmp_path):
    first_data = [[1, "original"]]
    replacement_data = [[2, "replacement"]]

    saved_path = save_raw_data(
        raw_data=first_data,
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        output_dir=tmp_path,
        endpoint=TEST_ENDPOINT,
    )
    original_content = saved_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_raw_data(
            raw_data=replacement_data,
            symbol="BTCUSDT",
            interval="1d",
            start="2019-01-01",
            end="2026-01-01",
            output_dir=tmp_path,
            endpoint=TEST_ENDPOINT,
        )

    assert saved_path.read_text(encoding="utf-8") == original_content


def test_save_raw_data_writes_metadata_file(tmp_path):
    retrieved_at = datetime(
        2026,
        10,
        25,
        12,
        30,
        tzinfo=timezone.utc,
    )
    raw_data = [
        [1704067200000, "100.0"],
        [1704153600000, "110.0"],
    ]

    data_path = save_raw_data(
        raw_data=raw_data,
        endpoint=TEST_ENDPOINT,
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        output_dir=tmp_path,
        retrieved_at=retrieved_at,
    )
    metadata_path = data_path.with_suffix(".metadata.json")

    assert metadata_path.exists()

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    expected_checksum = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest()

    assert metadata["symbol"] == "BTCUSDT"
    assert metadata["interval"] == "1d"
    assert metadata["requested_start"] == "2019-01-01"
    assert metadata["requested_end_exclusive"] == "2026-01-01"
    assert metadata["row_count"] == 2
    assert metadata["data_filename"] == data_path.name
    assert metadata["sha256"] == expected_checksum
    assert metadata["api_endpoint"] == TEST_ENDPOINT
    assert metadata["retrieved_at_utc"] == "2026-10-25T12:30:00Z"
    assert (
        metadata["first_candle_open_utc"]
        == "2024-01-01T00:00:00Z"
    )
    assert (
        metadata["last_candle_open_utc"]
        == "2024-01-02T00:00:00Z"
    )


def test_save_raw_data_rejects_empty_data_before_writing(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        save_raw_data(
            raw_data=[],
            endpoint=TEST_ENDPOINT,
            symbol="BTCUSDT",
            interval="1d",
            start="2019-01-01",
            end="2026-01-01",
            output_dir=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_save_raw_data_checks_metadata_before_writing_raw_file(tmp_path):
    data_path = (
        tmp_path
        / "btcusdt_1d_2019-01-01_2026-01-01.json"
    )
    metadata_path = data_path.with_suffix(".metadata.json")
    metadata_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_raw_data(
            raw_data=[[1704067200000, "100.0"]],
            endpoint=TEST_ENDPOINT,
            symbol="BTCUSDT",
            interval="1d",
            start="2019-01-01",
            end="2026-01-01",
            output_dir=tmp_path,
        )

    assert not data_path.exists()
    assert metadata_path.read_text(encoding="utf-8") == "{}"


def test_load_raw_snapshot_returns_data_and_metadata(tmp_path):
    raw_data = [
        [1704067200000, "100.0"],
        [1704153600000, "110.0"],
    ]
    data_path = save_raw_data(
        raw_data=raw_data,
        symbol="ETHUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        output_dir=tmp_path,
        endpoint=TEST_ENDPOINT,
    )
    loaded_data, metadata = load_raw_snapshot(path=data_path)

    assert loaded_data == raw_data
    assert metadata["data_filename"] == data_path.name
    assert metadata["row_count"] == len(raw_data)
    assert metadata["symbol"] == "ETHUSDT"


def test_save_raw_data_requires_endpoint(tmp_path):
    with pytest.raises(TypeError, match="endpoint"):
        save_raw_data(
            raw_data=[[1704067200000, "100.0"]],
            symbol="BTCUSDT",
            interval="1d",
            start="2019-01-01",
            end="2026-01-01",
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "wrong_value", "error_message"),
    [
        ("data_filename", "wrong_file.json", "filename"),
        ("row_count", 999, "rows"),
    ],
)
def test_load_raw_snapshot_rejects_wrong_metadata(
    tmp_path,
    field,
    wrong_value,
    error_message,
):
    raw_data = [
        [1704067200000, "100.0"],
        [1704153600000, "110.0"],
    ]
    data_path = save_raw_data(
        raw_data=raw_data,
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        endpoint=TEST_ENDPOINT,
        output_dir=tmp_path,
    )
    metadata_path = data_path.with_suffix(".metadata.json")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    metadata[field] = wrong_value

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with pytest.raises(ValueError, match=error_message):
        load_raw_snapshot(data_path)


def test_load_raw_snapshot_rejects_changed_raw_data(tmp_path):
    raw_data = [
        [1704067200000, "100.0"],
        [1704153600000, "110.0"],
    ]
    data_path = save_raw_data(
        raw_data=raw_data,
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        endpoint=TEST_ENDPOINT,
        output_dir=tmp_path,
    )

    changed_data = [
        [1704067200000, "999.0"],
        [1704153600000, "110.0"],
    ]
    data_path.write_text(
        json.dumps(changed_data, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        load_raw_snapshot(data_path)
