from unittest.mock import Mock

import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas.testing import assert_frame_equal

import src.generate_report as report_module
from src.generate_report import (
    DEFAULT_REPORT_BENCHMARK,
    generate_report,
    main,
)


def _report_dataframes():
    dates = pd.date_range(
        "2025-01-01",
        periods=2,
        freq="D",
        tz="UTC",
        name="date",
    )
    metrics = pd.DataFrame(
        {"QLIKE": [-6.0], "MSE": [0.1], "MAE": [0.2]},
        index=pd.Index(["garch_1_1"], name="model"),
    )
    dm_results = pd.DataFrame(
        {"model_a": ["garch_1_1"], "model_b": ["benchmark"]}
    )
    forecast_panel = pd.DataFrame(
        {
            "variance_proxy": [0.01, 0.02],
            "garch_1_1": [0.02, 0.03],
        },
        index=dates,
    )
    daily_losses = pd.DataFrame(
        {"QLIKE": [-6.0], "MSE": [0.1], "MAE": [0.2]}
    )
    metric_summary = pd.DataFrame(
        {"QLIKE": [-6.0]},
        index=pd.Index(["garch_1_1"], name="model"),
    )
    dm_summary = pd.DataFrame(
        {"model_a": ["garch_1_1"], "model_b": ["benchmark"]}
    )
    cumulative = pd.DataFrame(
        {
            "garch_1_1": [-0.1, -0.2],
            "persistence": [10.0, 20.0],
        },
        index=dates,
    )

    return {
        "metrics": metrics,
        "dm_results": dm_results,
        "forecast_panel": forecast_panel,
        "daily_losses": daily_losses,
        "metric_summary": metric_summary,
        "dm_summary": dm_summary,
        "cumulative": cumulative,
    }


def _patch_report_dependencies(monkeypatch, data):
    monkeypatch.setattr(
        report_module,
        "load_test_metrics",
        Mock(return_value=data["metrics"]),
    )
    monkeypatch.setattr(
        report_module,
        "load_dm_results",
        Mock(return_value=data["dm_results"]),
    )
    monkeypatch.setattr(
        report_module,
        "load_test_forecast_panel",
        Mock(return_value=data["forecast_panel"]),
    )
    monkeypatch.setattr(
        report_module,
        "load_test_losses",
        Mock(return_value=data["daily_losses"]),
    )
    monkeypatch.setattr(
        report_module,
        "build_metric_summary",
        Mock(return_value=data["metric_summary"]),
    )
    monkeypatch.setattr(
        report_module,
        "build_dm_summary",
        Mock(return_value=data["dm_summary"]),
    )
    cumulative_mock = Mock(
        return_value=data["cumulative"]
    )
    monkeypatch.setattr(
        report_module,
        "build_cumulative_qlike_differences",
        cumulative_mock,
    )

    volatility_figure = Figure()
    volatility_figure.subplots()
    cumulative_figure = Figure()
    cumulative_figure.subplots()
    volatility_plot_mock = Mock(
        return_value=(
            volatility_figure,
            volatility_figure.axes[0],
        )
    )
    cumulative_plot_mock = Mock(
        return_value=(
            cumulative_figure,
            cumulative_figure.axes[0],
        )
    )
    monkeypatch.setattr(
        report_module,
        "plot_test_volatility",
        volatility_plot_mock,
    )
    monkeypatch.setattr(
        report_module,
        "plot_cumulative_qlike",
        cumulative_plot_mock,
    )

    return cumulative_mock, cumulative_plot_mock


def test_generate_report_writes_atomic_bundle(
    monkeypatch,
    tmp_path,
):
    data = _report_dataframes()
    cumulative_mock, plot_mock = (
        _patch_report_dependencies(
            monkeypatch,
            data,
        )
    )
    run_directory = tmp_path / "run_001"
    output_root = tmp_path / "reports"

    report_directory = generate_report(
        run_directory=run_directory,
        output_root=output_root,
        benchmark_name="historical_variance",
    )

    assert report_directory == output_root / "run_001"
    assert not (
        output_root / ".run_001.incomplete"
    ).exists()
    assert {
        path.name
        for path in report_directory.iterdir()
    } == {
        "metric_summary.csv",
        "dm_summary.csv",
        "cumulative_qlike.csv",
        "volatility_forecasts.png",
        "cumulative_qlike.png",
    }

    cumulative_mock.assert_called_once_with(
        data["daily_losses"],
        benchmark_name="historical_variance",
    )
    plotted_data = plot_mock.call_args.args[0]
    assert_frame_equal(
        plotted_data,
        data["cumulative"][["garch_1_1"]],
    )


def test_generate_report_refuses_existing_directory(
    tmp_path,
):
    run_directory = tmp_path / "run_001"
    output_root = tmp_path / "reports"
    final_directory = output_root / "run_001"
    final_directory.mkdir(parents=True)

    with pytest.raises(
        FileExistsError,
        match="Report directory already exists",
    ):
        generate_report(
            run_directory=run_directory,
            output_root=output_root,
        )


def test_generate_report_does_not_publish_partial_bundle(
    monkeypatch,
    tmp_path,
):
    data = _report_dataframes()
    _patch_report_dependencies(
        monkeypatch,
        data,
    )
    monkeypatch.setattr(
        report_module,
        "build_metric_summary",
        Mock(side_effect=RuntimeError("simulated failure")),
    )
    run_directory = tmp_path / "run_001"
    output_root = tmp_path / "reports"

    with pytest.raises(
        RuntimeError,
        match="simulated failure",
    ):
        generate_report(
            run_directory=run_directory,
            output_root=output_root,
        )

    assert not (output_root / "run_001").exists()
    assert (
        output_root / ".run_001.incomplete"
    ).is_dir()


def test_report_main_forwards_cli_arguments(
    monkeypatch,
    tmp_path,
):
    run_directory = tmp_path / "run_001"
    output_root = tmp_path / "reports"
    final_directory = output_root / "run_001"
    generate_mock = Mock(return_value=final_directory)
    monkeypatch.setattr(
        report_module,
        "generate_report",
        generate_mock,
    )

    result = main(
        [
            "--run-directory",
            str(run_directory),
            "--output-root",
            str(output_root),
            "--benchmark",
            "historical_variance",
        ]
    )

    assert result == final_directory
    generate_mock.assert_called_once_with(
        run_directory=run_directory,
        output_root=output_root,
        benchmark_name="historical_variance",
    )


def test_report_default_benchmark_is_historical_variance():
    assert DEFAULT_REPORT_BENCHMARK == "historical_variance"
