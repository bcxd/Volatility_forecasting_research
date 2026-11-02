from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import src.pipeline as pipeline_module
from src.pipeline import build_parser
from src.config import ExperimentConfig
from src.models import GarchWalkForwardResult


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


def test_generate_validation_forecasts_builds_candidates(
    monkeypatch,
    tmp_path,
):
    train_dates = pd.date_range(
        "2024-01-01",
        periods=2,
        freq="D",
        tz="UTC",
    )
    val_dates = pd.date_range(
        "2024-01-03",
        periods=2,
        freq="D",
        tz="UTC",
    )

    train_df = pd.DataFrame(
        {
            "log_return": [0.1, 0.2],
            "variance_proxy": [1.0, 4.0],
        },
        index=train_dates,
    )
    val_df = pd.DataFrame(
        {
            "log_return": [0.3, 0.4],
            "variance_proxy": [9.0, 16.0],
        },
        index=val_dates,
    )

    config = ExperimentConfig(
        raw_path=tmp_path / "snapshot.json",
        ewma_lambdas=(0.50, 0.90),
    )
    constraints = ("constraint",)

    garch_result = pd.Series(
        [2.0, 3.0],
        index=val_dates,
        name="garch_variance",
    )

    expected_refit_log = pd.DataFrame(
        {
            "omega": [0.1],
            "alpha": [0.2],
            "beta": [0.5],
        },
        index=pd.DatetimeIndex(
            [val_dates[0]],
            name="refit_date",
        ),
    )

    garch_mock = Mock(
        return_value=GarchWalkForwardResult(
            forecasts=garch_result,
            refit_log=expected_refit_log,
        )
    )

    monkeypatch.setattr(
        pipeline_module,
        "walk_forward_garch",
        garch_mock,
    )

    (
        forecasts,
        candidate_lambdas,
        refit_log,
    ) = pipeline_module.generate_validation_forecasts(
        train_df,
        val_df,
        config,
        constraints,
    )

    assert forecasts.index.equals(val_dates)
    assert forecasts.columns.tolist() == [
        "persistence",
        "historical_variance",
        "ewma_lambda_0.50",
        "ewma_lambda_0.90",
        "garch_1_1",
    ]

    assert candidate_lambdas == {
        "ewma_lambda_0.50": 0.50,
        "ewma_lambda_0.90": 0.90,
    }

    pd.testing.assert_frame_equal(
        refit_log,
        expected_refit_log,
    )

    pd.testing.assert_series_equal(
        forecasts["garch_1_1"],
        garch_result.rename("garch_1_1"),
    )

    call_arguments = garch_mock.call_args.kwargs

    assert call_arguments["forecast_dates"].equals(
        val_dates
    )
    assert call_arguments["method"] == config.garch_method
    assert call_arguments["bounds"] == config.garch_bounds
    assert call_arguments["constraints"] is constraints
    assert (
        call_arguments["refit_every"]
        == config.garch_refit_every
    )


def test_select_best_ewma_uses_lowest_candidate_qlike():
    validation_results = pd.DataFrame(
        {
            "QLIKE": {
                "persistence": -10.0,
                "ewma_lambda_0.90": -6.0,
                "ewma_lambda_0.94": -7.0,
            },
        }
    )

    candidate_lambdas = {
        "ewma_lambda_0.90": 0.90,
        "ewma_lambda_0.94": 0.94,
    }

    best_name, best_lambda = (
        pipeline_module.select_best_ewma(
            validation_results,
            candidate_lambdas,
        )
    )

    assert best_name == "ewma_lambda_0.94"
    assert best_lambda == 0.94


def test_generate_test_forecasts_uses_validation_history(
    monkeypatch,
    tmp_path,
):
    train_dates = pd.date_range(
        "2024-01-01",
        periods=2,
        freq="D",
        tz="UTC",
    )
    val_dates = pd.date_range(
        "2024-01-03",
        periods=1,
        freq="D",
        tz="UTC",
    )
    test_dates = pd.date_range(
        "2024-01-04",
        periods=2,
        freq="D",
        tz="UTC",
    )

    train_df = pd.DataFrame(
        {
            "log_return": [0.1, 0.2],
            "variance_proxy": [1.0, 4.0],
        },
        index=train_dates,
    )
    val_df = pd.DataFrame(
        {
            "log_return": [0.3],
            "variance_proxy": [9.0],
        },
        index=val_dates,
    )
    test_df = pd.DataFrame(
        {
            "log_return": [0.4, 0.5],
            "variance_proxy": [16.0, 25.0],
        },
        index=test_dates,
    )

    config = ExperimentConfig(
        raw_path=tmp_path / "snapshot.json",
    )
    constraints = ("constraint",)

    ewma_dates = val_dates.append(test_dates)
    ewma_result = pd.Series(
        [10.0, 11.0, 12.0],
        index=ewma_dates,
    )
    garch_result = pd.Series(
        [20.0, 21.0],
        index=test_dates,
    )
    expected_refit_log = pd.DataFrame(
        {
            "omega": [0.1],
            "alpha": [0.2],
            "beta": [0.5],
        },
        index=pd.DatetimeIndex(
            [test_dates[0]],
            name="refit_date",
        ),
    )

    ewma_mock = Mock(return_value=ewma_result)
    garch_mock = Mock(
        return_value=GarchWalkForwardResult(
            forecasts=garch_result,
            refit_log=expected_refit_log,
        )
    )
    monkeypatch.setattr(
        pipeline_module,
        "calculate_ewma",
        ewma_mock,
    )
    monkeypatch.setattr(
        pipeline_module,
        "walk_forward_garch",
        garch_mock,
    )

    (
        forecasts,
        refit_log,
    ) = pipeline_module.generate_test_forecasts(
        train_df,
        val_df,
        test_df,
        best_lambda=0.94,
        config=config,
        garch_constraints=constraints,
    )

    assert forecasts.index.equals(test_dates)
    assert forecasts.columns.tolist() == [
        "persistence",
        "historical_variance",
        "ewma_lambda_0.94",
        "garch_1_1",
    ]

    pd.testing.assert_frame_equal(
        refit_log,
        expected_refit_log,
    )

    pd.testing.assert_series_equal(
        forecasts["ewma_lambda_0.94"],
        ewma_result.loc[test_dates].rename(
            "ewma_lambda_0.94"
        ),
    )

    ewma_arguments = ewma_mock.call_args.kwargs

    assert ewma_arguments["forecast_df"].index.equals(
        ewma_dates
    )
    assert ewma_arguments["train_df"] is train_df
    assert ewma_arguments["lamda_"] == 0.94

    garch_arguments = garch_mock.call_args.kwargs

    assert garch_arguments["df"].index.equals(
        train_dates.append(val_dates).append(test_dates)
    )
    assert garch_arguments["forecast_dates"].equals(
        test_dates
    )
    assert garch_arguments["constraints"] is constraints


def test_pipeline_main_persists_complete_run_bundle(
    monkeypatch,
    tmp_path,
):
    created_at = datetime(
        2026,
        11,
        2,
        12,
        30,
        tzinfo=timezone.utc,
    )
    repository_path = tmp_path / "repository"
    raw_path = tmp_path / "snapshot.json"
    run_directory = tmp_path / "results" / "run"

    config = ExperimentConfig(
        raw_path=raw_path,
        results_root=tmp_path / "results",
        ewma_lambdas=(0.94,),
    )

    snapshot_metadata = {
        "symbol": config.symbol,
        "interval": config.interval,
        "requested_start": config.sample_start,
        "requested_end_exclusive": (
            config.sample_end_exclusive
        ),
        "sha256": "snapshot-checksum",
    }
    raw_data = [[1704067200000, "100.0"]]
    processed_data = pd.DataFrame()

    train_dates = pd.date_range(
        "2023-12-30",
        periods=2,
        freq="D",
        tz="UTC",
    )
    validation_dates = pd.date_range(
        "2024-01-01",
        periods=1,
        freq="D",
        tz="UTC",
    )
    test_dates = pd.date_range(
        "2025-01-01",
        periods=1,
        freq="D",
        tz="UTC",
    )

    train_df = pd.DataFrame(
        {
            "variance_proxy": [1.0, 2.0],
        },
        index=train_dates,
    )
    validation_df = pd.DataFrame(
        {
            "log_return": [0.3],
            "variance_proxy": [3.0],
        },
        index=validation_dates,
    )
    test_df = pd.DataFrame(
        {
            "log_return": [0.4],
            "variance_proxy": [4.0],
        },
        index=test_dates,
    )

    validation_forecasts = pd.DataFrame(
        {
            "ewma_lambda_0.94": [2.5],
        },
        index=validation_dates,
    )
    validation_metrics = pd.DataFrame(
        {
            "QLIKE": [-6.0],
            "MSE": [0.25],
            "MAE": [0.5],
        },
        index=["ewma_lambda_0.94"],
    )
    validation_losses = pd.DataFrame(
        {
            "QLIKE": [-6.0],
            "MSE": [0.25],
            "MAE": [0.5],
        },
        index=pd.MultiIndex.from_product(
            [
                validation_dates,
                ["ewma_lambda_0.94"],
            ],
            names=["date", "model"],
        ),
    )
    validation_refit_log = pd.DataFrame(
        {
            "omega": [0.1],
        },
        index=pd.DatetimeIndex(
            [validation_dates[0]],
            name="refit_date",
        ),
    )

    test_forecasts = pd.DataFrame(
        {
            "ewma_lambda_0.94": [3.5],
        },
        index=test_dates,
    )
    test_metrics = pd.DataFrame(
        {
            "QLIKE": [-5.0],
            "MSE": [0.25],
            "MAE": [0.5],
        },
        index=["ewma_lambda_0.94"],
    )
    test_losses = pd.DataFrame(
        {
            "QLIKE": [-5.0],
            "MSE": [0.25],
            "MAE": [0.5],
        },
        index=pd.MultiIndex.from_product(
            [
                test_dates,
                ["ewma_lambda_0.94"],
            ],
            names=["date", "model"],
        ),
    )
    test_refit_log = pd.DataFrame(
        {
            "omega": [0.2],
        },
        index=pd.DatetimeIndex(
            [test_dates[0]],
            name="refit_date",
        ),
    )

    monkeypatch.setattr(
        pipeline_module,
        "load_raw_snapshot",
        Mock(
            return_value=(
                raw_data,
                snapshot_metadata,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "data_validation",
        Mock(return_value=processed_data),
    )
    monkeypatch.setattr(
        pipeline_module,
        "save_processed_data",
        Mock(
            return_value=tmp_path / "processed.csv"
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "prepare_data",
        Mock(
            return_value=(
                train_df,
                validation_df,
                test_df,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_validation_forecasts",
        Mock(
            return_value=(
                validation_forecasts,
                {
                    "ewma_lambda_0.94": 0.94,
                },
                validation_refit_log,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "evaluate_forecasts",
        Mock(
            side_effect=[
                (
                    validation_metrics,
                    validation_losses,
                    validation_forecasts,
                ),
                (
                    test_metrics,
                    test_losses,
                    test_forecasts,
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_test_forecasts",
        Mock(
            return_value=(
                test_forecasts,
                test_refit_log,
            )
        ),
    )

    bundle_mock = Mock(
        return_value=run_directory
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_run_bundle",
        bundle_mock,
        raising=False,
    )

    result = pipeline_module.main(
        config,
        created_at=created_at,
        repository_path=repository_path,
    )

    assert result == run_directory

    bundle_mock.assert_called_once()

    call_arguments = bundle_mock.call_args.kwargs

    assert call_arguments[
        "repository_path"
    ] == repository_path
    assert call_arguments["created_at"] == created_at
    assert call_arguments["config"] is config
    assert (
        call_arguments["snapshot_metadata"]
        is snapshot_metadata
    )
    assert (
        call_arguments["selected_ewma_name"]
        == "ewma_lambda_0.94"
    )
    assert (
        call_arguments["selected_ewma_lambda"]
        == 0.94
    )

    expected_validation_targets = (
        validation_df.loc[
            validation_forecasts.index,
            [
                "log_return",
                "variance_proxy",
            ],
        ]
    )

    pd.testing.assert_frame_equal(
        call_arguments["validation_targets"],
        expected_validation_targets,
    )

    assert (
        call_arguments["validation_forecasts"]
        is validation_forecasts
    )
    assert (
        call_arguments["validation_losses"]
        is validation_losses
    )
    assert (
        call_arguments["validation_metrics"]
        is validation_metrics
    )
    expected_test_targets = (
        test_df.loc[
            test_forecasts.index,
            [
                "log_return",
                "variance_proxy",
            ],
        ]
    )

    pd.testing.assert_frame_equal(
        call_arguments["test_targets"],
        expected_test_targets,
    )

    assert (
        call_arguments["test_forecasts"]
        is test_forecasts
    )
    assert (
        call_arguments["test_losses"]
        is test_losses
    )
    assert (
        call_arguments["test_metrics"]
        is test_metrics
    )

    expected_refit_log = (
        pipeline_module.combine_garch_refit_logs(
            validation_refit_log,
            test_refit_log,
        )
    )

    pd.testing.assert_frame_equal(
        call_arguments["garch_refit_log"],
        expected_refit_log,
    )


def test_combine_garch_refit_logs_labels_periods():
    validation_date = pd.Timestamp(
        "2024-01-01",
        tz="UTC",
    )
    test_date = pd.Timestamp(
        "2025-01-01",
        tz="UTC",
    )

    validation_log = pd.DataFrame(
        {
            "omega": [0.1],
            "alpha": [0.2],
            "beta": [0.5],
        },
        index=pd.DatetimeIndex(
            [validation_date],
            name="refit_date",
        ),
    )
    test_log = pd.DataFrame(
        {
            "omega": [0.2],
            "alpha": [0.3],
            "beta": [0.4],
        },
        index=pd.DatetimeIndex(
            [test_date],
            name="refit_date",
        ),
    )

    combined = (
        pipeline_module.combine_garch_refit_logs(
            validation_log,
            test_log,
        )
    )

    assert combined.index.tolist() == [
        validation_date,
        test_date,
    ]
    assert combined.index.name == "refit_date"
    assert combined["period"].tolist() == [
        "validation",
        "test",
    ]

    assert "period" not in validation_log.columns
    assert "period" not in test_log.columns
