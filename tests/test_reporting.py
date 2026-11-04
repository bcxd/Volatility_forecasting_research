import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.reporting import (
    load_dm_results,
    load_test_metrics,
)


def test_load_test_metrics_restores_model_index(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    expected = pd.DataFrame(
        {
            "QLIKE": [-6.5, -6.3],
            "MSE": [0.1, 0.2],
            "MAE": [0.3, 0.4],
        },
        index=pd.Index(
            [
                "ewma_lambda_0.97",
                "garch_1_1",
            ],
            name="model",
        ),
    )

    expected.to_csv(
        run_directory / "test_metrics.csv"
    )

    actual = load_test_metrics(
        run_directory
    )

    assert_frame_equal(actual, expected)


def test_load_test_metrics_requires_saved_artifact(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Required artifact is missing: "
            ".*test_metrics\\.csv"
        ),
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_requires_expected_columns(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        {
            "QLIKE": [-6.5],
            "MSE": [0.1],
        },
        index=["garch_1_1"],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="test metrics columns must be exactly",
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_rejects_empty_table(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        columns=["QLIKE", "MSE", "MAE"],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="test metrics must not be empty",
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_requires_unique_models(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        {
            "QLIKE": [-6.5, -6.4],
            "MSE": [0.1, 0.2],
            "MAE": [0.3, 0.4],
        },
        index=[
            "garch_1_1",
            "garch_1_1",
        ],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="model names must be unique",
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_requires_numeric_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        {
            "QLIKE": ["not-a-number"],
            "MSE": [0.1],
            "MAE": [0.3],
        },
        index=["garch_1_1"],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="test metrics must be numeric",
    ):
        load_test_metrics(run_directory)


def test_load_test_metrics_requires_finite_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    metrics = pd.DataFrame(
        {
            "QLIKE": [float("inf")],
            "MSE": [0.1],
            "MAE": [0.3],
        },
        index=["garch_1_1"],
    )
    metrics.to_csv(
        run_directory / "test_metrics.csv"
    )

    with pytest.raises(
        ValueError,
        match="test metrics must be finite",
    ):
        load_test_metrics(run_directory)


def _valid_dm_results():
    return pd.DataFrame(
        {
            "model_a": ["garch_1_1"],
            "model_b": ["ewma_lambda_0.97"],
            "n_observations": [365],
            "bandwidth": [5],
            "mean_loss_a": [-6.61],
            "mean_loss_b": [-6.69],
            "mean_difference": [0.08],
            "dm_statistic": [1.41],
            "raw_p_value": [0.16],
            "holm_adjusted_p_value": [0.25],
            "familywise_alpha": [0.05],
            "reject_holm": [False],
        }
    )


def test_load_dm_results_restores_saved_table(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    expected = _valid_dm_results()
    expected.to_csv(
        run_directory / "dm_results.csv"
    )

    actual = load_dm_results(run_directory)

    assert_frame_equal(actual, expected)


def test_load_dm_results_requires_saved_artifact(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Required artifact is missing: "
            ".*dm_results\\.csv"
        ),
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_expected_columns(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results().drop(
        columns=["raw_p_value"]
    )
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM result columns must be exactly",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_rejects_empty_table(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results().iloc[0:0]
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM results must not be empty",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_unique_pairs(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = pd.concat(
        [
            _valid_dm_results(),
            _valid_dm_results(),
        ],
        ignore_index=True,
    )
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM comparison pairs must be unique",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_numeric_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results["raw_p_value"] = (
        results["raw_p_value"].astype(object)
    )
    results.loc[0, "raw_p_value"] = (
        "not-a-number"
    )

    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM numeric columns must be numeric",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_finite_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results.loc[0, "dm_statistic"] = float(
        "inf"
    )
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM numeric columns must be finite",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_valid_p_values(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results.loc[0, "raw_p_value"] = 1.1
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="DM p-values must be between 0 and 1",
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_valid_alpha(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results.loc[0, "familywise_alpha"] = 0.0
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            "familywise alpha must be "
            "strictly between 0 and 1"
        ),
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_valid_sample_size(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results["n_observations"] = (
        results["n_observations"].astype(float)
    )
    results.loc[0, "n_observations"] = 1.5
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            "n_observations must be integers "
            "of at least 2"
        ),
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_valid_bandwidth(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results.loc[0, "bandwidth"] = 365
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match=(
            "bandwidth must be an integer from "
            "0 to n_observations - 1"
        ),
    ):
        load_dm_results(run_directory)


def test_load_dm_results_requires_boolean_decisions(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    results = _valid_dm_results()
    results["reject_holm"] = (
        results["reject_holm"].astype(object)
    )
    results.loc[0, "reject_holm"] = (
        "not-a-boolean"
    )
    results.to_csv(
        run_directory / "dm_results.csv"
    )

    with pytest.raises(
        ValueError,
        match="reject_holm must be boolean",
    ):
        load_dm_results(run_directory)