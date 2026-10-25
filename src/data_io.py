import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from secrets import compare_digest


def _metadata_path(data_path):
    return data_path.with_suffix(".metadata.json")


def _utc_iso(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")

    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _milliseconds_to_utc_iso(value):
    timestamp = datetime.fromtimestamp(
        int(value) / 1000,
        tz=timezone.utc,
    )
    return _utc_iso(timestamp)


def save_raw_snapshot(
    raw_data,
    endpoint,
    symbol,
    interval,
    start,
    end,
    output_dir=Path("data/raw"),
    retrieved_at=None,
):
    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("raw_data must be a non-empty list")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol.lower()}_{interval}_{start}_{end}.json"
    data_path = output_dir / filename
    metadata_path = _metadata_path(data_path)

    existing_paths = [
        path for path in (data_path, metadata_path) if path.exists()
    ]
    if existing_paths:
        raise FileExistsError(
            f"Snapshot file already exists: {existing_paths[0]}"
        )

    data_text = json.dumps(raw_data, indent=2) + "\n"
    data_bytes = data_text.encode("utf-8")
    checksum = hashlib.sha256(data_bytes).hexdigest()

    if retrieved_at is None:
        retrieved_at = datetime.now(timezone.utc)

    metadata = {
        "schema_version": 1,
        "source": "Binance Spot",
        "api_endpoint": endpoint,
        "symbol": symbol,
        "interval": interval,
        "requested_start": start,
        "requested_end_exclusive": end,
        "retrieved_at_utc": _utc_iso(retrieved_at),
        "first_candle_open_utc": _milliseconds_to_utc_iso(
            raw_data[0][0]
        ),
        "last_candle_open_utc": _milliseconds_to_utc_iso(
            raw_data[-1][0]
        ),
        "row_count": len(raw_data),
        "data_filename": data_path.name,
        "sha256": checksum,
    }

    with data_path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(data_text)

    with metadata_path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    return data_path, metadata_path


def load_raw_snapshot(data_path):
    data_path = Path(data_path)
    metadata_path = _metadata_path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"Raw snapshot not found: {data_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Raw snapshot metadata not found: {metadata_path}"
        )

    data_bytes = data_path.read_bytes()
    raw_data = json.loads(data_bytes.decode("utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(raw_data, list):
        raise ValueError("Raw snapshot must contain a JSON list")

    expected_checksum = metadata.get("sha256")
    actual_checksum = hashlib.sha256(data_bytes).hexdigest()
    if not isinstance(expected_checksum, str) or not compare_digest(
        expected_checksum,
        actual_checksum,
    ):
        raise ValueError("Raw snapshot checksum does not match metadata")

    if metadata.get("data_filename") != data_path.name:
        raise ValueError("Raw snapshot filename does not match metadata")
    if metadata.get("row_count") != len(raw_data):
        raise ValueError("Raw snapshot row count does not match metadata")

    return raw_data, metadata


def validate_snapshot_metadata(
    metadata,
    symbol,
    interval,
    start,
    end,
):
    expected_values = {
        "symbol": symbol,
        "interval": interval,
        "requested_start": start,
        "requested_end_exclusive": end,
    }

    for field, expected in expected_values.items():
        observed = metadata.get(field)
        if observed != expected:
            raise ValueError(
                f"Snapshot {field} is {observed!r}; expected {expected!r}"
            )
