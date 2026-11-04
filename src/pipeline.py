import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ExperimentConfig
from src.artifacts import create_run_bundle
from src.data_io import load_raw_snapshot
from src.data_validation import data_validation, save_processed_data
from src.features import calculate_log_returns, variance_proxy
from src.splits import split_data
from src.statistical_tests import compare_qlike_losses
from src.benchmarks import persistence_var, historical_var
from src.models import (
    build_garch_constraints,
    calculate_ewma,
    walk_forward_garch,
)
from src.evaluation import evaluate_forecasts


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the forecasting experiment from a raw snapshot."
        )
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        required=True,
        help=(
            "Path to a raw JSON snapshot with adjacent metadata."
        ),
    )

    return parser


def validate_snapshot_for_experiment(
    metadata,
    config: ExperimentConfig,
):
    expected_metadata = {
        "symbol": config.symbol,
        "interval": config.interval,
        "requested_start": config.sample_start,
        "requested_end_exclusive": (
            config.sample_end_exclusive
        ),
    }

    for field, expected_value in expected_metadata.items():
        actual_value = metadata.get(field)

        if actual_value != expected_value:
            raise ValueError(
                f"Snapshot metadata mismatch for {field}: "
                f"expected {expected_value!r}, "
                f"got {actual_value!r}"
            )


def prepare_data(
    processed_data: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    df = processed_data.copy()
    df = calculate_log_returns(df)
    df = variance_proxy(df)
    df = df.dropna(
        subset=[
            "log_return",
            "variance_proxy",
        ]
    )

    return split_data(
        df,
        config.train_end,
        config.validation_end,
        config.test_end,
    )


def generate_validation_forecasts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
    garch_constraints,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
    pd.DataFrame,
]:
    train_val_df = pd.concat(
        [train_df, val_df]
    ).sort_index()

    forecasts = {}

    forecasts["persistence"] = persistence_var(
        train_val_df
    ).loc[
        val_df.index,
        "persistence_var",
    ]

    forecasts["historical_variance"] = historical_var(
        train_val_df
    ).loc[
        val_df.index,
        "historical_var",
    ]

    candidate_lambdas = {}

    for lambda_ in config.ewma_lambdas:
        column_name = f"ewma_lambda_{lambda_:.2f}"

        forecasts[column_name] = calculate_ewma(
            forecast_df=val_df,
            train_df=train_df,
            lamda_=lambda_,
        )

        candidate_lambdas[column_name] = lambda_

    forecasts = pd.DataFrame(
        forecasts,
        index=val_df.index,
    )

    garch_result = walk_forward_garch(
        df=train_val_df,
        forecast_dates=val_df.index,
        method=config.garch_method,
        bounds=config.garch_bounds,
        constraints=garch_constraints,
        refit_every=config.garch_refit_every,
    )

    forecasts["garch_1_1"] = (
        garch_result.forecasts
    )

    return (
        forecasts,
        candidate_lambdas,
        garch_result.refit_log,
    )


def select_best_ewma(
    validation_results: pd.DataFrame,
    candidate_lambdas: dict[str, float],
) -> tuple[str, float]:
    best_name = min(
        candidate_lambdas,
        key=lambda name: validation_results.loc[
            name,
            "QLIKE",
        ],
    )

    best_lambda = candidate_lambdas[best_name]

    return best_name, best_lambda


def generate_test_forecasts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_lambda: float,
    config: ExperimentConfig,
    garch_constraints,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    all_df = pd.concat(
        [train_df, val_df, test_df]
    ).sort_index()

    validation_and_test_df = pd.concat(
        [val_df, test_df]
    ).sort_index()

    ewma_after_train = calculate_ewma(
        forecast_df=validation_and_test_df,
        train_df=train_df,
        lamda_=best_lambda,
    )

    persistence_all = persistence_var(all_df)
    historical_all = historical_var(all_df)

    garch_result = walk_forward_garch(
        df=all_df,
        forecast_dates=test_df.index,
        method=config.garch_method,
        bounds=config.garch_bounds,
        constraints=garch_constraints,
        refit_every=config.garch_refit_every,
    )

    forecasts = pd.DataFrame(index=test_df.index)

    forecasts["persistence"] = persistence_all.loc[
        test_df.index,
        "persistence_var",
    ]

    forecasts["historical_variance"] = historical_all.loc[
        test_df.index,
        "historical_var",
    ]

    forecasts[
        f"ewma_lambda_{best_lambda:.2f}"
    ] = ewma_after_train.loc[test_df.index]

    forecasts["garch_1_1"] = (
        garch_result.forecasts
    )

    return (
        forecasts,
        garch_result.refit_log,
    )


def combine_garch_refit_logs(
    validation_log: pd.DataFrame,
    test_log: pd.DataFrame,
) -> pd.DataFrame:
    validation_labeled = (
        validation_log.copy()
    )
    test_labeled = test_log.copy()

    validation_labeled.insert(
        0,
        "period",
        "validation",
    )
    test_labeled.insert(
        0,
        "period",
        "test",
    )

    combined = pd.concat(
        [
            validation_labeled,
            test_labeled,
        ]
    ).sort_index()

    combined.index.name = "refit_date"

    return combined


def main(
    config: ExperimentConfig,
    *,
    created_at: datetime | None = None,
    repository_path: Path | None = None,
):
    if created_at is None:
        created_at = datetime.now(
            timezone.utc
        )

    if repository_path is None:
        repository_path = (
            Path(__file__).resolve().parents[1]
        )

    garch_constraints = build_garch_constraints(
        persistence_margin=(
            config.garch_persistence_margin
        ),
    )

    raw_data, metadata = load_raw_snapshot(
        config.raw_path
    )
    validate_snapshot_for_experiment(
        metadata,
        config,
    )

    print(
        f"Raw snapshot loaded from: "
        f"{config.raw_path}"
    )

    processed_data = data_validation(
        raw_data,
        start=config.sample_start,
        end=config.sample_end_exclusive,
    )
    processed_path = save_processed_data(
        df=processed_data,
        symbol=config.symbol,
        interval=config.interval,
        start=config.sample_start,
        end=config.sample_end_exclusive,
    )

    print(f"Processed data saved to: {processed_path}")

    train_df, val_df, test_df = prepare_data(
        processed_data,
        config,
    )

    (
        validation_forecasts,
        ewma_candidate_lambdas,
        validation_garch_refit_log,
    ) = generate_validation_forecasts(
        train_df,
        val_df,
        config,
        garch_constraints,
    )

    validation_targets = val_df.loc[
        validation_forecasts.index,
        [
            "log_return",
            "variance_proxy",
        ],
    ].copy()

    actual_validation = (
        validation_targets["variance_proxy"]
    )

    (
        validation_results,
        validation_daily_losses,
        validation_forecasts,
    ) = evaluate_forecasts(
        actual=actual_validation,
        forecasts=validation_forecasts,
        floor=config.variance_floor,
    )

    best_name, best_lambda = select_best_ewma(
        validation_results,
        ewma_candidate_lambdas,
    )

    print("\nValidation metrics:")
    print(validation_results)
    print(f"\nSelected EWMA lambda: {best_lambda:.2f}")

    (
        test_forecasts,
        test_garch_refit_log,
    ) = generate_test_forecasts(
        train_df,
        val_df,
        test_df,
        best_lambda=best_lambda,
        config=config,
        garch_constraints=garch_constraints,
    )

    test_targets = test_df.loc[
        test_forecasts.index,
        [
            "log_return",
            "variance_proxy",
        ],
    ].copy()

    actual_test = test_targets["variance_proxy"]

    (
        test_results,
        test_daily_losses,
        test_forecasts,
    ) = evaluate_forecasts(
        actual=actual_test,
        forecasts=test_forecasts,
        floor=config.variance_floor,
    )

    print("\nTest metrics:")
    print(test_results)

    dm_results = compare_qlike_losses(
        test_daily_losses,
        selected_ewma_name=best_name,
        alpha=config.familywise_alpha,
    )

    garch_refit_log = combine_garch_refit_logs(
        validation_garch_refit_log,
        test_garch_refit_log,
    )

    run_directory = create_run_bundle(
        repository_path=repository_path,
        created_at=created_at,
        config=config,
        snapshot_metadata=metadata,
        selected_ewma_name=best_name,
        selected_ewma_lambda=best_lambda,
        validation_targets=validation_targets,
        validation_forecasts=(
            validation_forecasts
        ),
        validation_losses=(
            validation_daily_losses
        ),
        validation_metrics=validation_results,
        test_targets=test_targets,
        test_forecasts=test_forecasts,
        test_losses=test_daily_losses,
        test_metrics=test_results,
        garch_refit_log=garch_refit_log,
        dm_results=dm_results,
    )

    print(
        f"\nRun artifacts saved to: "
        f"{run_directory}"
    )

    return run_directory


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    config = ExperimentConfig(
        raw_path=arguments.raw_path,
    )

    main(config)
