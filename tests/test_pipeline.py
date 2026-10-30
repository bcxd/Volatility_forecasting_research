from pathlib import Path
from unittest.mock import Mock

import pytest
import pandas as pd

import src.pipeline as pipeline_module
from src.pipeline import build_parser
from src.config import ExperimentConfig


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


class StopAfterValidation(Exception):
    pass


def test_pipeline_accepts_matching_snapshot_metadata(
    monkeypatch,
    tmp_path,
):
    raw_path = tmp_path / "snapshot.json"
    config = ExperimentConfig(raw_path=raw_path)
    raw_data = [[1704067200000, "100.0"]]
    metadata = {
        "symbol": config.symbol,
        "interval": config.interval,
        "requested_start": config.sample_start,
        "requested_end_exclusive": (
            config.sample_end_exclusive
        ),
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
        pipeline_module.main(config)

    load_mock.assert_called_once_with(raw_path)
    validation_mock.assert_called_once_with(
        raw_data,
        start=config.sample_start,
        end=config.sample_end_exclusive,
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
    config = ExperimentConfig(raw_path=raw_path)

    metadata = {
        "symbol": config.symbol,
        "interval": config.interval,
        "requested_start": config.sample_start,
        "requested_end_exclusive": (
            config.sample_end_exclusive
        ),
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
        pipeline_module.main(config)


class StopAfterLoad(Exception):
    pass


def test_pipeline_main_loads_selected_snapshot(
    monkeypatch,
    tmp_path,
):
    raw_path = tmp_path / "snapshot.json"
    config = ExperimentConfig(raw_path=raw_path)

    load_mock = Mock(side_effect=StopAfterLoad)

    monkeypatch.setattr(
        pipeline_module,
        "load_raw_snapshot",
        load_mock,
    )

    with pytest.raises(StopAfterLoad):
        pipeline_module.main(config)

    load_mock.assert_called_once_with(raw_path)


def test_prepare_data_creates_features_and_splits(
    tmp_path,
):
    dates = pd.date_range(
        "2024-01-01",
        periods=4,
        freq="D",
        tz="UTC",
    )
    processed_data = pd.DataFrame(
        {
            "close": [
                100.0,
                110.0,
                121.0,
                133.1,
            ],
        },
        index=dates,
    )

    config = ExperimentConfig(
        raw_path=tmp_path / "snapshot.json",
        sample_start="2024-01-01",
        sample_end_exclusive="2024-01-05",
        train_end="2024-01-03",
        validation_end="2024-01-04",
        test_end="2024-01-05",
    )

    train_df, val_df, test_df = (
        pipeline_module.prepare_data(
            processed_data,
            config,
        )
    )

    assert train_df.index.tolist() == [dates[1]]
    assert val_df.index.tolist() == [dates[2]]
    assert test_df.index.tolist() == [dates[3]]

    for split in (train_df, val_df, test_df):
        assert "log_return" in split.columns
        assert "variance_proxy" in split.columns

    assert "log_return" not in processed_data.columns
    assert "variance_proxy" not in processed_data.columns