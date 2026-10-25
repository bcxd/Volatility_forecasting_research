import argparse
from pathlib import Path

import pandas as pd

from src.data_io import load_raw_snapshot, validate_snapshot_metadata
from src.data_validation import data_validation, save_processed_data
from src.features import calculate_log_returns, variance_proxy
from src.splits import split_data
from src.benchmarks import persistence_var, historical_var
from src.models import calculate_ewma, walk_forward_garch
from src.evaluation import evaluate_forecasts


SYMBOL = "BTCUSDT"
INTERVAL = "1d"
START_DATE = "2019-01-01"
END_DATE = "2026-01-01"

TRAIN_END = "2024-01-01"
VALIDATION_END = "2025-01-01"
TEST_END = "2026-01-01"

VARIANCE_FLOOR = 1e-12
LAMBDA_CANDIDATES = (0.90, 0.94, 0.97, 0.99)

GARCH_METHOD = "SLSQP"
GARCH_BOUNDS = (
    (1e-12, None),  # omega > 0
    (0.0, 1.0),     # alpha >= 0
    (0.0, 1.0),     # beta >= 0
)
GARCH_CONSTRAINTS = (
    {
        "type": "ineq",
        "fun": lambda params: 1.0 - params[1] - params[2] - 1e-8,
    },
)
GARCH_REFIT_EVERY = 7


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the forecasting experiment from a raw snapshot."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        required=True,
        help="Path to raw JSON snapshot with adjacent metadata file.",
    )
    return parser


def main(raw_path):
    raw_data, snapshot_metadata = load_raw_snapshot(raw_path)
    validate_snapshot_metadata(
        snapshot_metadata,
        symbol=SYMBOL,
        interval=INTERVAL,
        start=START_DATE,
        end=END_DATE,
    )
    print(f"Raw snapshot loaded from: {raw_path}")
    processed_data = data_validation(
        raw_data,
        start=START_DATE,
        end=END_DATE,
    )
    processed_path = save_processed_data(
        df=processed_data,
        symbol=SYMBOL,
        interval=INTERVAL,
        start=START_DATE,
        end=END_DATE,
    )
    print(f"Processed data saved to: {processed_path}")

    df = processed_data.copy()
    df = calculate_log_returns(df)
    df = variance_proxy(df)
    df = df.dropna(subset=["log_return", "variance_proxy"])

    train_df, val_df, test_df = split_data(
        df,
        TRAIN_END,
        VALIDATION_END,
        TEST_END,
    )

    validation_forecasts = {}
    train_val_df = pd.concat([train_df, val_df]).sort_index()
    validation_forecasts["persistence"] = persistence_var(train_val_df).loc[
        val_df.index, "persistence_var"
    ]
    validation_forecasts["historical_variance"] = historical_var(train_val_df).loc[
        val_df.index, "historical_var"
    ]
    ewma_candidate_lambdas = {}
    for lambda_ in LAMBDA_CANDIDATES:
        column_name = f"ewma_lambda_{lambda_:.2f}"

        validation_forecasts[column_name] = calculate_ewma(
            forecast_df=val_df,
            train_df=train_df,
            lamda_=lambda_,
        )

        ewma_candidate_lambdas[column_name] = lambda_

    validation_forecasts = pd.DataFrame(
        validation_forecasts,
        index=val_df.index,
    )

    validation_forecasts["garch_1_1"] = walk_forward_garch(
        df=train_val_df,
        forecast_dates=val_df.index,
        method=GARCH_METHOD,
        bounds=GARCH_BOUNDS,
        constraints=GARCH_CONSTRAINTS,
        refit_every=GARCH_REFIT_EVERY,
    )

    actual_validation = val_df["variance_proxy"]

    (
        validation_results,
        validation_qlike_losses,
        validation_forecasts,
    ) = evaluate_forecasts(
        actual=actual_validation,
        forecasts=validation_forecasts,
        floor=VARIANCE_FLOOR,
    )

    best_name = min(
        ewma_candidate_lambdas,
        key=lambda name: validation_results.loc[name, "QLIKE"],
    )
    best_lambda = ewma_candidate_lambdas[best_name]

    print("\nValidation metrics:")
    print(validation_results)
    print(f"\nSelected EWMA lambda: {best_lambda:.2f}")

    all_df = pd.concat([train_df, val_df, test_df]).sort_index()

    ewma_after_train = calculate_ewma(
        forecast_df=pd.concat([val_df, test_df]),
        train_df=train_df,
        lamda_=best_lambda,
    )

    ewma_test = ewma_after_train.loc[test_df.index]
    persistence_all = persistence_var(all_df)
    historical_all = historical_var(all_df)
    garch_test = walk_forward_garch(
        df=all_df,
        forecast_dates=test_df.index,
        method=GARCH_METHOD,
        bounds=GARCH_BOUNDS,
        constraints=GARCH_CONSTRAINTS,
        refit_every=GARCH_REFIT_EVERY,
    )

    test_forecasts = pd.DataFrame(index=test_df.index)
    test_forecasts["persistence"] = persistence_all.loc[
        test_df.index, "persistence_var"
    ]
    test_forecasts["historical_variance"] = historical_all.loc[
        test_df.index, "historical_var"
    ]
    test_forecasts[f"ewma_lambda_{best_lambda:.2f}"] = ewma_test
    test_forecasts["garch_1_1"] = garch_test

    actual_test = test_df["variance_proxy"].reindex(test_forecasts.index)

    (
        test_results,
        test_qlike_losses,
        test_forecasts,
    ) = evaluate_forecasts(
        actual=actual_test,
        forecasts=test_forecasts,
        floor=VARIANCE_FLOOR,
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
    main(arguments.raw_path)
