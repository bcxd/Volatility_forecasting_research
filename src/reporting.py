"""Load saved experiment artifacts for reporting."""

from pathlib import Path

import numpy as np
import pandas as pd


_TEST_METRIC_COLUMNS = (
    "QLIKE",
    "MSE",
    "MAE",
)

_DM_RESULT_COLUMNS = (
    "model_a",
    "model_b",
    "n_observations",
    "bandwidth",
    "mean_loss_a",
    "mean_loss_b",
    "mean_difference",
    "dm_statistic",
    "raw_p_value",
    "holm_adjusted_p_value",
    "familywise_alpha",
    "reject_holm",
)

_DM_NUMERIC_COLUMNS = (
    "n_observations",
    "bandwidth",
    "mean_loss_a",
    "mean_loss_b",
    "mean_difference",
    "dm_statistic",
    "raw_p_value",
    "holm_adjusted_p_value",
    "familywise_alpha",
)


def load_test_metrics(
    run_directory: Path,
) -> pd.DataFrame:
    """Load aggregate test metrics with model names as the index."""
    metrics_path = (
        run_directory / "test_metrics.csv"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            "Required artifact is missing: "
            f"{metrics_path}"
        )

    metrics = pd.read_csv(
        metrics_path,
        index_col=0,
    )

    if tuple(metrics.columns) != (
        _TEST_METRIC_COLUMNS
    ):
        expected_columns = ", ".join(
            _TEST_METRIC_COLUMNS
        )
        raise ValueError(
            "test metrics columns must be exactly: "
            f"{expected_columns}"
        )

    if metrics.empty:
        raise ValueError(
            "test metrics must not be empty"
        )

    if not metrics.index.is_unique:
        raise ValueError(
            "model names must be unique"
        )

    try:
        metrics = metrics.apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "test metrics must be numeric"
        ) from error

    if not np.isfinite(
        metrics.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "test metrics must be finite"
        )

    metrics.index.name = "model"

    return metrics


def load_dm_results(
    run_directory: Path,
) -> pd.DataFrame:
    """Load saved Diebold-Mariano comparison results."""
    results_path = (
        run_directory / "dm_results.csv"
    )

    if not results_path.is_file():
        raise FileNotFoundError(
            "Required artifact is missing: "
            f"{results_path}"
        )

    results = pd.read_csv(
        results_path,
        index_col=0,
    ).reset_index(drop=True)

    if tuple(results.columns) != (
        _DM_RESULT_COLUMNS
    ):
        expected_columns = ", ".join(
            _DM_RESULT_COLUMNS
        )
        raise ValueError(
            "DM result columns must be exactly: "
            f"{expected_columns}"
        )

    if results.empty:
        raise ValueError(
            "DM results must not be empty"
        )

    if results.duplicated(
        subset=["model_a", "model_b"]
    ).any():
        raise ValueError(
            "DM comparison pairs must be unique"
        )

    try:
        numeric_values = results.loc[
            :,
            _DM_NUMERIC_COLUMNS,
        ].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "DM numeric columns must be numeric"
        ) from error

    results.loc[
        :,
        _DM_NUMERIC_COLUMNS,
    ] = numeric_values

    if not np.isfinite(
        numeric_values.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "DM numeric columns must be finite"
        )
    p_values = numeric_values.loc[
        :,
        [
            "raw_p_value",
            "holm_adjusted_p_value",
        ],
    ].to_numpy(dtype=float)

    if (
        (p_values < 0.0)
        | (p_values > 1.0)
    ).any():
        raise ValueError(
            "DM p-values must be between 0 and 1"
        )

    alpha_values = numeric_values[
        "familywise_alpha"
    ].to_numpy(dtype=float)

    if (
        (alpha_values <= 0.0)
        | (alpha_values >= 1.0)
    ).any():
        raise ValueError(
            "familywise alpha must be "
            "strictly between 0 and 1"
        )

    n_observations = numeric_values[
        "n_observations"
    ].to_numpy(dtype=float)

    if (
        (n_observations < 2)
        | (
            n_observations
            != np.floor(n_observations)
        )
    ).any():
        raise ValueError(
            "n_observations must be integers "
            "of at least 2"
        )

    bandwidth = numeric_values[
        "bandwidth"
    ].to_numpy(dtype=float)

    if (
        (bandwidth < 0)
        | (bandwidth != np.floor(bandwidth))
        | (bandwidth >= n_observations)
    ).any():
        raise ValueError(
            "bandwidth must be an integer from "
            "0 to n_observations - 1"
        )

    if not pd.api.types.is_bool_dtype(
        results["reject_holm"]
    ):
        raise ValueError(
            "reject_holm must be boolean"
        )

    results["n_observations"] = (
        n_observations.astype(int)
    )
    results["bandwidth"] = (
        bandwidth.astype(int)
    )

    return results