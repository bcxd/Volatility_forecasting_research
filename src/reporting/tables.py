"""Build reporting tables from validated experiment artifacts."""

import numpy as np
import pandas as pd


_METRIC_COLUMNS = (
    "QLIKE",
    "MSE",
    "MAE",
)


def build_metric_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a QLIKE-sorted table with ranks for each loss metric."""
    summary = metrics.copy()

    for metric in _METRIC_COLUMNS:
        summary[f"{metric}_rank"] = (
            summary[metric]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

    summary = summary.loc[
        :,
        [
            "QLIKE",
            "QLIKE_rank",
            "MSE",
            "MSE_rank",
            "MAE",
            "MAE_rank",
        ],
    ]

    return summary.sort_values(
        by="QLIKE",
        ascending=True,
        kind="stable",
    )


def build_dm_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build an interpretable summary of DM comparisons."""
    summary = results.copy()

    mean_difference = summary[
        "mean_difference"
    ]

    summary["lower_mean_loss_model"] = np.where(
        mean_difference < 0.0,
        summary["model_a"],
        np.where(
            mean_difference > 0.0,
            summary["model_b"],
            "tie",
        ),
    )

    summary["significant_after_holm"] = (
        summary["reject_holm"].copy()
    )

    return summary.loc[
        :,
        [
            "model_a",
            "model_b",
            "n_observations",
            "bandwidth",
            "mean_difference",
            "lower_mean_loss_model",
            "dm_statistic",
            "raw_p_value",
            "holm_adjusted_p_value",
            "familywise_alpha",
            "significant_after_holm",
        ],
    ]


def build_cumulative_qlike_differences(
    daily_losses: pd.DataFrame,
    benchmark_name: str,
) -> pd.DataFrame:
    """Accumulate model-minus-benchmark daily QLIKE differences."""
    qlike_by_model = daily_losses[
        "QLIKE"
    ].unstack(
        level="model",
        sort=False,
    )

    if benchmark_name not in qlike_by_model.columns:
        raise ValueError(
            "benchmark model not found: "
            f"{benchmark_name}"
        )

    comparison_models = [
        model_name
        for model_name in qlike_by_model.columns
        if model_name != benchmark_name
    ]

    if not comparison_models:
        raise ValueError(
            "at least one non-benchmark "
            "model is required"
        )

    daily_differences = qlike_by_model.loc[
        :,
        comparison_models,
    ].subtract(
        qlike_by_model[benchmark_name],
        axis="index",
    )

    return daily_differences.cumsum()
