from datetime import datetime, timezone
import json

import pytest
import pandas as pd
from unittest.mock import Mock, call

import src.artifacts as artifacts_module
from src.config import ExperimentConfig
from src.artifacts import (
    build_run_id,
    build_run_metadata,
    collect_git_state,
    collect_package_versions,
    create_run_bundle,
    create_run_directory,
    save_dataframe_csv,
    save_metadata_json,
    save_run_artifacts,
)


def test_create_run_directory_refuses_existing_run(
    tmp_path,
):
    results_root = tmp_path / "results" / "runs"
    run_id = "20261031T120000Z_abc1234"

    run_directory = create_run_directory(
        results_root,
        run_id,
    )

    assert run_directory == results_root / run_id
    assert run_directory.is_dir()

    with pytest.raises(FileExistsError):
        create_run_directory(
            results_root,
            run_id,
        )


def test_build_run_id_uses_utc_time_and_short_commit():
    created_at = datetime(
        2026,
        10,
        31,
        12,
        34,
        56,
        tzinfo=timezone.utc,
    )
    commit_hash = "abcdef1234567890"

    run_id = build_run_id(
        created_at,
        commit_hash,
    )

    assert run_id == "20261031T123456Z_abcdef1"


def test_build_run_id_rejects_datetime_without_timezone():
    created_at = datetime(
        2026,
        10,
        31,
        12,
        34,
        56,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        build_run_id(
            created_at,
            "abcdef1234567890",
        )


def test_save_dataframe_csv_preserves_datetime_index(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    dates = pd.date_range(
        "2025-01-01",
        periods=2,
        freq="D",
        tz="UTC",
        name="date",
    )
    dataframe = pd.DataFrame(
        {
            "garch_1_1": [1.5, 2.5],
        },
        index=dates,
    )

    output_path = save_dataframe_csv(
        run_directory,
        "test_forecasts.csv",
        dataframe,
    )

    assert output_path == (
        run_directory / "test_forecasts.csv"
    )
    assert output_path.is_file()

    loaded = pd.read_csv(
        output_path,
        index_col="date",
        parse_dates=["date"],
    )

    pd.testing.assert_frame_equal(
        loaded,
        dataframe,
        check_freq=False,
    )

    with pytest.raises(FileExistsError):
        save_dataframe_csv(
            run_directory,
            "test_forecasts.csv",
            dataframe,
        )


def test_save_metadata_json_preserves_content(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metadata = {
        "run_id": "20261031T123456Z_abcdef1",
        "code_commit": "abcdef1234567890",
        "source_snapshot": {
            "path": "data/raw/btcusdt.json",
            "sha256": "abc123",
        },
        "selected_ewma_lambda": 0.94,
    }

    output_path = save_metadata_json(
        run_directory,
        metadata,
    )

    assert output_path == (
        run_directory / "metadata.json"
    )

    with output_path.open(
        encoding="utf-8",
    ) as metadata_file:
        loaded = json.load(metadata_file)

    assert loaded == metadata

    with pytest.raises(FileExistsError):
        save_metadata_json(
            run_directory,
            metadata,
        )


def test_save_metadata_json_does_not_create_partial_file(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metadata = {
        "raw_path": tmp_path / "raw.json",
    }

    with pytest.raises(
        TypeError,
    ):
        save_metadata_json(
            run_directory,
            metadata,
        )

    assert not (
        run_directory / "metadata.json"
    ).exists()


def test_save_run_artifacts_writes_standard_files(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    dataframe = pd.DataFrame(
        {
            "value": [1.0],
        },
        index=pd.Index(
            ["row_1"],
            name="id",
        ),
    )

    metadata = {
        "run_id": "20261031T123456Z_abcdef1",
    }

    targets = pd.DataFrame(
        {
            "log_return": [0.1],
            "variance_proxy": [0.01],
        },
        index=pd.Index(
            ["2025-01-01"],
            name="date",
        ),
    )

    saved_paths = save_run_artifacts(
        run_directory,
        metadata=metadata,
        validation_forecasts=dataframe,
        validation_losses=dataframe,
        validation_metrics=dataframe,
        test_forecasts=dataframe,
        test_losses=dataframe,
        test_metrics=dataframe,
        garch_refit_log=dataframe,
        dm_results=dataframe,
        validation_targets=targets,
        test_targets=targets,
    )

    assert set(saved_paths) == {
        "metadata",
        "validation_forecasts",
        "validation_losses",
        "validation_metrics",
        "test_forecasts",
        "test_losses",
        "test_metrics",
        "garch_refit_log",
        "dm_results",
        "validation_targets",
        "test_targets",
    }

    assert {
        path.name
        for path in saved_paths.values()
    } == {
        "metadata.json",
        "validation_forecasts.csv",
        "validation_losses.csv",
        "validation_metrics.csv",
        "test_forecasts.csv",
        "test_losses.csv",
        "test_metrics.csv",
        "garch_refit_log.csv",
        "dm_results.csv",
        "validation_targets.csv",
        "test_targets.csv",
    }

    assert all(
        path.is_file()
        for path in saved_paths.values()
    )


def test_build_run_metadata_records_provenance(
    tmp_path,
):
    created_at = datetime(
        2026,
        11,
        1,
        12,
        30,
        tzinfo=timezone.utc,
    )
    config = ExperimentConfig(
        raw_path=tmp_path / "snapshot.json",
    )
    snapshot_metadata = {
        "data_filename": "snapshot.json",
        "sha256": "abc123",
        "row_count": 2557,
        "api_endpoint": "https://example.test/klines",
    }
    package_versions = {
        "numpy": "2.3.0",
        "pandas": "2.3.0",
        "scipy": "1.16.0",
    }

    metadata = build_run_metadata(
        run_id="20261101T123000Z_abcdef1",
        created_at=created_at,
        code_commit="abcdef1234567890",
        working_tree_dirty=False,
        config=config,
        snapshot_metadata=snapshot_metadata,
        selected_ewma_name="ewma_lambda_0.94",
        selected_ewma_lambda=0.94,
        package_versions=package_versions,
    )

    assert metadata["run_id"] == (
        "20261101T123000Z_abcdef1"
    )
    assert metadata["created_at_utc"] == (
        "2026-11-01T12:30:00Z"
    )

    assert metadata["code"] == {
        "commit": "abcdef1234567890",
        "working_tree_dirty": False,
    }

    assert metadata["source_snapshot"] == {
        "path": config.raw_path.as_posix(),
        **snapshot_metadata,
    }

    assert metadata["experiment"]["sample"] == {
        "symbol": config.symbol,
        "interval": config.interval,
        "start": config.sample_start,
        "end_exclusive": (
            config.sample_end_exclusive
        ),
    }

    assert metadata["experiment"]["splits"] == {
        "train_end": config.train_end,
        "validation_end": config.validation_end,
        "test_end": config.test_end,
    }

    assert metadata["experiment"]["evaluation"] == {
        "variance_floor": config.variance_floor,
        "familywise_alpha": (
            config.familywise_alpha
        ),
    }

    assert metadata["experiment"]["candidates"][
        "ewma_lambdas"
    ] == list(config.ewma_lambdas)

    assert metadata["model_selection"] == {
        "selected_ewma_name": "ewma_lambda_0.94",
        "selected_ewma_lambda": 0.94,
    }

    assert metadata["environment"] == {
        "package_versions": package_versions,
    }

    assert metadata["protocol"] == {
        "status": "exploratory",
        "amendments_path": (
            "docs/amendments.md"
        ),
    }

    json.dumps(
        metadata,
        allow_nan=False,
    )


def test_collect_git_state_returns_commit_and_dirty_status(
    monkeypatch,
    tmp_path,
):
    run_mock = Mock(
        side_effect=[
            Mock(stdout="abcdef1234567890\n"),
            Mock(stdout=" M src/pipeline.py\n"),
        ]
    )

    monkeypatch.setattr(
        artifacts_module.subprocess,
        "run",
        run_mock,
    )

    git_state = collect_git_state(tmp_path)

    assert git_state == {
        "commit": "abcdef1234567890",
        "working_tree_dirty": True,
    }

    expected_options = {
        "cwd": tmp_path,
        "check": True,
        "capture_output": True,
        "text": True,
    }

    assert run_mock.call_args_list == [
        call(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            **expected_options,
        ),
        call(
            [
                "git",
                "status",
                "--porcelain",
            ],
            **expected_options,
        ),
    ]


def test_collect_package_versions_uses_distribution_metadata(
    monkeypatch,
):
    version_mock = Mock(
        side_effect=[
            "2.3.0",
            "2.3.1",
        ]
    )

    monkeypatch.setattr(
        artifacts_module,
        "package_version",
        version_mock,
    )

    versions = collect_package_versions(
        (
            "numpy",
            "pandas",
        )
    )

    assert versions == {
        "numpy": "2.3.0",
        "pandas": "2.3.1",
    }

    assert version_mock.call_args_list == [
        call("numpy"),
        call("pandas"),
    ]


def test_collect_package_versions_uses_runtime_defaults(
    monkeypatch,
):
    version_mock = Mock(
        side_effect=[
            "2.3.0",
            "2.3.1",
            "1.16.0",
            "2.32.0",
        ]
    )

    monkeypatch.setattr(
        artifacts_module,
        "package_version",
        version_mock,
    )

    versions = collect_package_versions()

    assert versions == {
        "numpy": "2.3.0",
        "pandas": "2.3.1",
        "scipy": "1.16.0",
        "requests": "2.32.0",
    }

    assert version_mock.call_args_list == [
        call("numpy"),
        call("pandas"),
        call("scipy"),
        call("requests"),
    ]


def test_create_run_bundle_writes_versioned_artifacts(
    monkeypatch,
    tmp_path,
):
    config = ExperimentConfig(
        raw_path=tmp_path / "snapshot.json",
        results_root=tmp_path / "results" / "runs",
    )
    created_at = datetime(
        2026,
        11,
        2,
        1,
        2,
        3,
        tzinfo=timezone.utc,
    )

    dataframe = pd.DataFrame(
        {
            "value": [1.0],
        },
        index=pd.Index(
            ["row_1"],
            name="id",
        ),
    )

    git_mock = Mock(
        return_value={
            "commit": "abcdef1234567890",
            "working_tree_dirty": True,
        }
    )
    package_mock = Mock(
        return_value={
            "numpy": "2.3.0",
            "pandas": "2.3.1",
        }
    )

    monkeypatch.setattr(
        artifacts_module,
        "collect_git_state",
        git_mock,
    )
    monkeypatch.setattr(
        artifacts_module,
        "collect_package_versions",
        package_mock,
    )

    run_directory = create_run_bundle(
        repository_path=tmp_path,
        created_at=created_at,
        config=config,
        snapshot_metadata={
            "sha256": "snapshot-checksum",
        },
        selected_ewma_name="ewma_lambda_0.94",
        selected_ewma_lambda=0.94,
        validation_targets=dataframe,
        validation_forecasts=dataframe,
        validation_losses=dataframe,
        validation_metrics=dataframe,
        test_targets=dataframe,
        test_forecasts=dataframe,
        test_losses=dataframe,
        test_metrics=dataframe,
        garch_refit_log=dataframe,
        dm_results=dataframe,
    )

    assert run_directory == (
        config.results_root
        / "20261102T010203Z_abcdef1"
    )
    assert not (
        config.results_root
        / ".20261102T010203Z_abcdef1.incomplete"
    ).exists()

    assert {
        path.name
        for path in run_directory.iterdir()
    } == {
        "metadata.json",
        "validation_targets.csv",
        "validation_forecasts.csv",
        "validation_losses.csv",
        "validation_metrics.csv",
        "test_targets.csv",
        "test_forecasts.csv",
        "test_losses.csv",
        "test_metrics.csv",
        "garch_refit_log.csv",
        "dm_results.csv",
    }

    git_mock.assert_called_once_with(tmp_path)
    package_mock.assert_called_once_with()

    with (
        run_directory / "metadata.json"
    ).open(
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(metadata_file)

    assert metadata["code"] == {
        "commit": "abcdef1234567890",
        "working_tree_dirty": True,
    }
    assert metadata["source_snapshot"][
        "sha256"
    ] == "snapshot-checksum"


def test_create_run_bundle_does_not_publish_partial_run(
    monkeypatch,
    tmp_path,
):
    config = ExperimentConfig(
        raw_path=tmp_path / "snapshot.json",
        results_root=tmp_path / "results" / "runs",
    )
    created_at = datetime(
        2026,
        11,
        2,
        12,
        0,
        tzinfo=timezone.utc,
    )
    dataframe = pd.DataFrame(
        {
            "value": [1.0],
        },
        index=pd.Index(
            ["row_1"],
            name="id",
        ),
    )

    monkeypatch.setattr(
        artifacts_module,
        "collect_git_state",
        Mock(
            return_value={
                "commit": "abcdef1234567890",
                "working_tree_dirty": False,
            }
        ),
    )
    monkeypatch.setattr(
        artifacts_module,
        "collect_package_versions",
        Mock(return_value={}),
    )
    monkeypatch.setattr(
        artifacts_module,
        "save_run_artifacts",
        Mock(
            side_effect=RuntimeError(
                "simulated write failure"
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="simulated write failure",
    ):
        create_run_bundle(
            repository_path=tmp_path,
            created_at=created_at,
            config=config,
            snapshot_metadata={
                "sha256": "snapshot-checksum",
            },
            selected_ewma_name=(
                "ewma_lambda_0.94"
            ),
            selected_ewma_lambda=0.94,
            validation_targets=dataframe,
            validation_forecasts=dataframe,
            validation_losses=dataframe,
            validation_metrics=dataframe,
            test_targets=dataframe,
            test_forecasts=dataframe,
            test_losses=dataframe,
            test_metrics=dataframe,
            garch_refit_log=dataframe,
            dm_results=dataframe,
        )

    run_id = "20261102T120000Z_abcdef1"

    final_directory = (
        config.results_root / run_id
    )
    incomplete_directory = (
        config.results_root
        / f".{run_id}.incomplete"
    )

    assert not final_directory.exists()
    assert incomplete_directory.is_dir()
