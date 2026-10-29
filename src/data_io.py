import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _milliseconds_to_utc_iso(milliseconds):
    timestamp = datetime.fromtimestamp(
        int(milliseconds) / 1000,
        tz=timezone.utc,
    )

    return (
        timestamp.isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def save_raw_data(
    raw_data,
    symbol,
    interval,
    start,
    end,
    *,
    endpoint,
    output_dir=Path("data/raw"),
    retrieved_at=None,
):
    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("raw_data must be a non-empty list")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{symbol.lower()}_{interval}_{start}_{end}.json"
    )
    path = output_dir / filename
    metadata_path = path.with_suffix(".metadata.json")

    for candidate_path in (path, metadata_path):
        if candidate_path.exists():
            raise FileExistsError(
                f"Snapshot file already exists: {candidate_path}"
            )
    with path.open("x", encoding="utf-8") as file:
        json.dump(raw_data, file, indent=2)

    checksum = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if retrieved_at is None:
        retrieved_at = datetime.now(timezone.utc)

    retrieved_at_utc = (
        retrieved_at.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    metadata = {
        "symbol": symbol,
        "interval": interval,
        "requested_start": start,
        "requested_end_exclusive": end,
        "row_count": len(raw_data),
        "data_filename": path.name,
        "sha256": checksum,
        "api_endpoint": endpoint,
        "retrieved_at_utc": retrieved_at_utc,
        "first_candle_open_utc": _milliseconds_to_utc_iso(
            raw_data[0][0]
        ),
        "last_candle_open_utc": _milliseconds_to_utc_iso(
            raw_data[-1][0]
        ),
    }
    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return path


def load_raw_snapshot(path):
    path = Path(path)
    metadata_path = path.with_suffix(".metadata.json")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    expected_filename = metadata["data_filename"]
    expected_checksum = metadata["sha256"]
    expected_row = metadata["row_count"]

    actual_filename = path.name
    actual_checksum = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    actual_row = len(raw_data)

    if actual_filename != expected_filename:
        raise ValueError("Raw data filename does not match metadata")
    if actual_checksum != expected_checksum:
        raise ValueError("Raw data checksum does not match metadata")

    if expected_row != actual_row:
        raise ValueError("Raw data rows does not match metadata")

    return raw_data, metadata
