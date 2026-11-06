"""Generate tables and figures from a saved experiment run."""

import argparse
from pathlib import Path

from src.reporting import (
    build_cumulative_qlike_differences,
    build_dm_summary,
    build_metric_summary,
    load_dm_results,
    load_test_forecast_panel,
    load_test_losses,
    load_test_metrics,
    plot_cumulative_qlike,
    plot_test_volatility,
)


DEFAULT_REPORT_BENCHMARK = "historical_variance"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate report tables and figures from a saved run."
        )
    )
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="Path to one immutable experiment run directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/reports"),
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_REPORT_BENCHMARK,
        help="Benchmark used for cumulative QLIKE differences.",
    )

    return parser


def generate_report(
    run_directory: Path,
    output_root: Path = Path("results/reports"),
    benchmark_name: str = DEFAULT_REPORT_BENCHMARK,
) -> Path:
    """Create one atomic report bundle without refitting models."""
    report_id = run_directory.name
    final_directory = output_root / report_id

    if final_directory.exists():
        raise FileExistsError(
            "Report directory already exists: "
            f"{final_directory}"
        )

    incomplete_directory = (
        output_root / f".{report_id}.incomplete"
    )
    incomplete_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics = load_test_metrics(run_directory)
    dm_results = load_dm_results(run_directory)
    forecast_panel = load_test_forecast_panel(
        run_directory
    )
    daily_losses = load_test_losses(run_directory)

    metric_summary = build_metric_summary(metrics)
    dm_summary = build_dm_summary(dm_results)
    cumulative_qlike = (
        build_cumulative_qlike_differences(
            daily_losses,
            benchmark_name=benchmark_name,
        )
    )

    metric_summary.to_csv(
        incomplete_directory / "metric_summary.csv"
    )
    dm_summary.to_csv(
        incomplete_directory / "dm_summary.csv",
        index=False,
    )
    cumulative_qlike.to_csv(
        incomplete_directory / "cumulative_qlike.csv"
    )

    volatility_figure, _ = plot_test_volatility(
        forecast_panel
    )
    volatility_figure.savefig(
        incomplete_directory / "volatility_forecasts.png",
        dpi=160,
    )

    plot_columns = [
        model_name
        for model_name in cumulative_qlike.columns
        if model_name != "persistence"
    ]
    cumulative_plot_data = cumulative_qlike.loc[
        :,
        plot_columns or list(cumulative_qlike.columns),
    ]
    cumulative_figure, _ = plot_cumulative_qlike(
        cumulative_plot_data,
        benchmark_name=benchmark_name,
    )
    cumulative_figure.savefig(
        incomplete_directory / "cumulative_qlike.png",
        dpi=160,
    )

    incomplete_directory.rename(final_directory)

    return final_directory


def main(argv=None):
    args = build_parser().parse_args(argv)

    report_directory = generate_report(
        run_directory=args.run_directory,
        output_root=args.output_root,
        benchmark_name=args.benchmark,
    )

    print(
        "Report saved to: "
        f"{report_directory}"
    )

    return report_directory


if __name__ == "__main__":
    main()
