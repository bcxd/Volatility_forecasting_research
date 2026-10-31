import argparse
from pathlib import Path

import pandas as pd

from src.config import ExperimentConfig
from src.data_io import load_raw_snapshot
from src.data_validation import data_validation, save_processed_data
from src.features import calculate_log_returns, variance_proxy
from src.splits import split_data
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
) -> tuple[pd.DataFrame, dict[str, float]]:
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

    forecasts["garch_1_1"] = walk_forward_garch(
        df=train_val_df,
        forecast_dates=val_df.index,
        method=config.garch_method,
        bounds=config.garch_bounds,
        constraints=garch_constraints,
        refit_every=config.garch_refit_every,
    )

    return forecasts, candidate_lambdas


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
) -> pd.DataFrame:
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

    garch_test = walk_forward_garch(
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

    forecasts["garch_1_1"] = garch_test

    return forecasts


def main(config: ExperimentConfig):
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
    ) = generate_validation_forecasts(
        train_df,
        val_df,
        config,
        garch_constraints,
    )

    actual_validation = val_df["variance_proxy"]

    (
        validation_results,
        validation_qlike_losses,
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

    test_forecasts = generate_test_forecasts(
        train_df,
        val_df,
        test_df,
        best_lambda=best_lambda,
        config=config,
        garch_constraints=garch_constraints,
    )

    actual_test = test_df["variance_proxy"].reindex(test_forecasts.index)

    (
        test_results,
        test_qlike_losses,
        test_forecasts,
    ) = evaluate_forecasts(
        actual=actual_test,
        forecasts=test_forecasts,
        floor=config.variance_floor,
    )

    print("\nTest metrics:")
    print(test_results)

    plot_df = test_forecasts.copy()
    plot_df["actual_proxy"] = actual_test.mask(actual_test <= 0)

    ax = plot_df.plot(figsize=(12, 5), logy=True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Variance")
    ax.set_title("Test-period variance forecasts")


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    config = ExperimentConfig(
        raw_path=arguments.raw_path,
    )

    main(config)