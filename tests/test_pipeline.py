from pathlib import Path
from unittest.mock import Mock

import pytest

import src.pipeline as pipeline_module
from src.pipeline import build_parser


def test_pipeline_parser_reads_raw_path():
    args = build_parser().parse_args(
        [
            "--raw-path",
            "data/raw/snapshot.json",
        ]
    )

    assert args.raw_path == Path(
        "data/raw/snapshot.json"
    )


class StopAfterLoad(Exception):
    pass


class StopAfterValidation(Exception):
    pass


def test_pipeline_main_loads_selected_snapshot(
    monkeypatch,
    tmp_path,
):
    raw_path = tmp_path / "snapshot.json"
    load_mock = Mock(side_effect=StopAfterLoad)

    monkeypatch.setattr(
        pipeline_module,
        "load_raw_snapshot",
        load_mock,
    )

    with pytest.raises(StopAfterLoad):
        pipeline_module.main(raw_path)

    load_mock.assert_called_once_with(raw_path)


def test_pipeline_accepts_matching_snapshot_metadata(
    monkeypatch,
    tmp_path,
):
    raw_path = tmp_path / "snapshot.json"
    raw_data = [[1704067200000, "100.0"]]
    metadata = {
        "symbol": pipeline_module.SYMBOL,
        "interval": pipeline_module.INTERVAL,
        "requested_start": pipeline_module.START_DATE,
        "requested_end_exclusive": pipeline_module.END_DATE,
    }

    load_mock = Mock(return_value=(raw_data, metadata))
    validation_mock = Mock(side_effect=StopAfterValidation)

    monkeypatch.setattr(
        pipeline_module,
        "load_raw_snapshot",
        load_mock,
    )
    monkeypatch.setattr(
        pipeline_module,
        "data_validation",
        validation_mock,
    )

    with pytest.raises(StopAfterValidation):
        pipeline_module.main(raw_path)

    load_mock.assert_called_once_with(raw_path)
    validation_mock.assert_called_once_with(
        raw_data,
        start=pipeline_module.START_DATE,
        end=pipeline_module.END_DATE,
    )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("symbol", "ETHUSDT"),
        ("interval", "1h"),
        ("requested_start", "2020-01-01"),
        ("requested_end_exclusive", "2025-01-01"),
    ],
)
def test_pipeline_rejects_wrong_snapshot_metadata(
    monkeypatch,
    tmp_path,
    field,
    wrong_value,
):
    raw_path = tmp_path / "snapshot.json"

    metadata = {
        "symbol": pipeline_module.SYMBOL,
        "interval": pipeline_module.INTERVAL,
        "requested_start": pipeline_module.START_DATE,
        "requested_end_exclusive": pipeline_module.END_DATE,
    }
    metadata[field] = wrong_value

    load_mock = Mock(
        return_value=([], metadata)
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_raw_snapshot",
        load_mock,
    )

    with pytest.raises(ValueError, match=field):
        pipeline_module.main(raw_path)
