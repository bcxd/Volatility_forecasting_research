"""Statistical tests for forecast-loss comparisons."""

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd
from scipy.stats import norm


def newey_west_bandwidth(
    n_observations: int,
) -> int:
    """Return the pre-specified Newey-West bandwidth."""
    if n_observations < 2:
        raise ValueError(
            "n_observations must be at least 2"
        )

    bandwidth = 4 * (
        n_observations / 100
    ) ** (2 / 9)

    return floor(bandwidth)


def newey_west_long_run_variance(
    loss_differential: np.ndarray,
    bandwidth: int,
) -> float:
    """Estimate the Bartlett-kernel Newey-West long-run variance."""
    loss_differential = np.asarray(
        loss_differential,
        dtype=float,
    )

    if loss_differential.ndim != 1:
        raise ValueError(
            "loss_differential must be a 1D array"
        )

    n_observations = len(
        loss_differential
    )

    if n_observations < 2:
        raise ValueError(
            "loss_differential must contain at least 2 observations"
        )

    if not np.isfinite(
        loss_differential
    ).all():
        raise ValueError(
            "loss_differential must contain only finite values"
        )

    if not (
        0
        <= bandwidth
        < n_observations
    ):
        raise ValueError(
            "bandwidth must satisfy "
            "0 <= bandwidth < n_observations"
        )

    centered = (
        loss_differential
        - loss_differential.mean()
    )

    long_run_variance = (
        np.dot(centered, centered)
        / n_observations
    )

    for lag in range(
        1,
        bandwidth + 1,
    ):
        weight = (
            1
            - lag / (bandwidth + 1)
        )
        autocovariance = (
            np.dot(
                centered[lag:],
                centered[:-lag],
            )
            / n_observations
        )

        long_run_variance += (
            2
            * weight
            * autocovariance
        )

    return float(long_run_variance)


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Store one Diebold-Mariano test result."""

    n_observations: int
    bandwidth: int
    mean_loss_a: float
    mean_loss_b: float
    mean_difference: float
    statistic: float
    p_value: float


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
) -> DieboldMarianoResult:
    """Compare two forecast-loss series using a two-sided DM test."""
    loss_a = np.asarray(
        loss_a,
        dtype=float,
    )
    loss_b = np.asarray(
        loss_b,
        dtype=float,
    )

    if (
        loss_a.ndim != 1
        or loss_b.ndim != 1
    ):
        raise ValueError(
            "loss_a and loss_b must be 1D arrays"
        )
    if len(loss_a) != len(loss_b):
        raise ValueError(
            "loss_a and loss_b must contain "
            "the same number of observations"
        )

    n_observations = len(loss_a)

    if n_observations < 2:
        raise ValueError(
            "loss_a and loss_b must each "
            "contain at least 2 observations"
        )

    if (
        not np.isfinite(loss_a).all()
        or not np.isfinite(loss_b).all()
    ):
        raise ValueError(
            "loss_a and loss_b must contain "
            "only finite values"
        )

    bandwidth = newey_west_bandwidth(
        n_observations
    )

    loss_differential = (
        loss_a - loss_b
    )
    mean_loss_a = float(
        loss_a.mean()
    )
    mean_loss_b = float(
        loss_b.mean()
    )
    mean_difference = float(
        loss_differential.mean()
    )

    long_run_variance = (
        newey_west_long_run_variance(
            loss_differential,
            bandwidth,
        )
    )

    if (
        not np.isfinite(
            long_run_variance
        )
        or long_run_variance <= 0
    ):
        raise ValueError(
            "Newey-West long-run variance must be positive "
            "and finite"
        )

    standard_error = np.sqrt(
        long_run_variance
        / n_observations
    )

    statistic = float(
        mean_difference
        / standard_error
    )
    p_value = float(
        2
        * norm.sf(
            abs(statistic)
        )
    )

    return DieboldMarianoResult(
        n_observations=n_observations,
        bandwidth=bandwidth,
        mean_loss_a=mean_loss_a,
        mean_loss_b=mean_loss_b,
        mean_difference=mean_difference,
        statistic=statistic,
        p_value=p_value,
    )


@dataclass(frozen=True)
class HolmAdjustmentResult:
    """Store Holm-adjusted p-values and rejection decisions."""

    adjusted_p_values: np.ndarray
    reject: np.ndarray


def holm_adjust_pvalues(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> HolmAdjustmentResult:
    """Adjust a family of p-values using Holm's step-down method."""
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    if p_values.ndim != 1:
        raise ValueError(
            "p_values must be a 1D array"
        )

    if p_values.size == 0:
        raise ValueError(
            "p_values must contain at least one value"
        )

    if not np.isfinite(
        p_values
    ).all():
        raise ValueError(
            "p_values must contain only finite values"
        )

    if np.any(
        (p_values < 0)
        | (p_values > 1)
    ):
        raise ValueError(
            "p_values must be between 0 and 1"
        )

    # P-values may equal 0 or 1, but alpha is a decision
    # threshold, so its boundary values are excluded.
    if (
        not np.isfinite(alpha)
        or not (
            0 < alpha < 1
        )
    ):
        raise ValueError(
            "alpha must satisfy 0 < alpha < 1"
        )

    n_hypotheses = len(p_values)

    order = np.argsort(
        p_values,
        kind="stable",
    )
    sorted_p_values = p_values[
        order
    ]

    multipliers = np.arange(
        n_hypotheses,
        0,
        -1,
        dtype=float,
    )
    scaled_p_values = np.minimum(
        sorted_p_values * multipliers,
        1.0,
    )
    adjusted_sorted = (
        np.maximum.accumulate(
            scaled_p_values
        )
    )

    adjusted_p_values = np.empty_like(
        adjusted_sorted
    )
    adjusted_p_values[order] = (
        adjusted_sorted
    )

    reject = (
        adjusted_p_values <= alpha
    )

    return HolmAdjustmentResult(
        adjusted_p_values=adjusted_p_values,
        reject=reject,
    )


def compare_qlike_losses(
    daily_losses: pd.DataFrame,
    selected_ewma_name: str,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Run the five pre-specified QLIKE DM comparisons."""
    if not isinstance(
        daily_losses.index,
        pd.MultiIndex,
    ):
        raise ValueError(
            "daily_losses must use a MultiIndex "
            "with date and model levels"
        )

    required_index_levels = {
        "date",
        "model",
    }

    if (
        daily_losses.index.nlevels != 2
        or set(
            daily_losses.index.names
        )
        != required_index_levels
    ):
        raise ValueError(
            "daily_losses MultiIndex must have "
            "exactly two levels named date and model"
        )

    if "QLIKE" not in daily_losses.columns:
        raise ValueError(
            "daily_losses must contain a QLIKE column"
        )

    if daily_losses.index.has_duplicates:
        raise ValueError(
            "daily_losses must not contain "
            "duplicate date-model pairs"
        )

    required_models = {
        "garch_1_1",
        "persistence",
        "historical_variance",
        selected_ewma_name,
    }
    available_models = set(
        daily_losses.index.get_level_values(
            "model"
        )
    )
    missing_models = sorted(
        required_models
        - available_models
    )

    if missing_models:
        raise ValueError(
            "daily_losses is missing required models: "
            + ", ".join(missing_models)
        )

    for model in sorted(required_models):
        model_dates = daily_losses.xs(
            model,
            level="model",
        ).index

        if not model_dates.is_monotonic_increasing:
            raise ValueError(
                f"{model} forecast dates must be in "
                "chronological order"
            )

    comparisons = [
        (
            "garch_1_1",
            selected_ewma_name,
        ),
        (
            "garch_1_1",
            "persistence",
        ),
        (
            "garch_1_1",
            "historical_variance",
        ),
        (
            selected_ewma_name,
            "persistence",
        ),
        (
            selected_ewma_name,
            "historical_variance",
        ),
    ]

    rows = []

    for model_a, model_b in comparisons:
        loss_a = daily_losses.xs(
            model_a,
            level="model",
        )["QLIKE"]
        loss_b = daily_losses.xs(
            model_b,
            level="model",
        )["QLIKE"]

        if not loss_a.index.equals(
            loss_b.index
        ):
            raise ValueError(
                f"{model_a} and {model_b} must have "
                "identical forecast dates"
            )

        dm_result = diebold_mariano_test(
            loss_a.to_numpy(
                dtype=float
            ),
            loss_b.to_numpy(
                dtype=float
            ),
        )

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_observations": (
                    dm_result.n_observations
                ),
                "bandwidth": (
                    dm_result.bandwidth
                ),
                "mean_loss_a": (
                    dm_result.mean_loss_a
                ),
                "mean_loss_b": (
                    dm_result.mean_loss_b
                ),
                "mean_difference": (
                    dm_result.mean_difference
                ),
                "dm_statistic": (
                    dm_result.statistic
                ),
                "raw_p_value": (
                    dm_result.p_value
                ),
            }
        )

    raw_p_values = np.array(
        [
            row["raw_p_value"]
            for row in rows
        ],
        dtype=float,
    )
    holm_result = holm_adjust_pvalues(
        raw_p_values,
        alpha=alpha,
    )

    for index, row in enumerate(rows):
        row["holm_adjusted_p_value"] = float(
            holm_result.adjusted_p_values[
                index
            ]
        )
        row["familywise_alpha"] = float(
            alpha
        )
        row["reject_holm"] = bool(
            holm_result.reject[index]
        )

    return pd.DataFrame(rows)
