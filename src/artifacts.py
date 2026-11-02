"""Build and persist immutable experiment run artifacts."""

import json
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from importlib.metadata import version as package_version

from src.config import ExperimentConfig


RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "requests",
)


def create_run_directory(
    results_root: Path,
    run_id: str,
) -> Path:
    """Create a new run directory without overwriting an existing run."""
    run_directory = results_root / run_id

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory



def build_run_id(
    created_at: datetime,
    commit_hash: str,
) -> str:
    """Build a run identifier from a UTC timestamp and short commit hash."""
    if (
        created_at.tzinfo is None
        or created_at.utcoffset() is None
    ):
        raise ValueError(
            "created_at must be timezone-aware"
        )

    created_at_utc = created_at.astimezone(
        timezone.utc
    )

    timestamp = created_at_utc.strftime(
        "%Y%m%dT%H%M%SZ"
    )
    short_commit = commit_hash[:7]

    return f"{timestamp}_{short_commit}"


def build_run_metadata(
    *,
    run_id: str,
    created_at: datetime,
    code_commit: str,
    working_tree_dirty: bool,
    config: ExperimentConfig,
    snapshot_metadata: Mapping[str, Any],
    selected_ewma_name: str,
    selected_ewma_lambda: float,
    package_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Build JSON-safe provenance metadata for one experiment run."""
    if (
        created_at.tzinfo is None
        or created_at.utcoffset() is None
    ):
        raise ValueError(
            "created_at must be timezone-aware"
        )

    created_at_utc = (
        created_at.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    source_snapshot = {
        **dict(snapshot_metadata),
        "path": config.raw_path.as_posix(),
    }

    metadata = {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "code": {
            "commit": code_commit,
            "working_tree_dirty": (
                working_tree_dirty
            ),
        },
        "source_snapshot": source_snapshot,
        "experiment": {
            "sample": {
                "symbol": config.symbol,
                "interval": config.interval,
                "start": config.sample_start,
                "end_exclusive": (
                    config.sample_end_exclusive
                ),
            },
            "splits": {
                "train_end": config.train_end,
                "validation_end": (
                    config.validation_end
                ),
                "test_end": config.test_end,
            },
            "evaluation": {
                "variance_floor": (
                    config.variance_floor
                ),
            },
            "candidates": {
                "ewma_lambdas": list(
                    config.ewma_lambdas
                ),
                "garch": {
                    "specification": "GARCH(1,1)",
                    "method": config.garch_method,
                    "bounds": [
                        list(bound)
                        for bound in config.garch_bounds
                    ],
                    "persistence_margin": (
                        config.garch_persistence_margin
                    ),
                    "refit_every": (
                        config.garch_refit_every
                    ),
                },
            },
        },
        "model_selection": {
            "selected_ewma_name": (
                selected_ewma_name
            ),
            "selected_ewma_lambda": (
                selected_ewma_lambda
            ),
        },
        "environment": {
            "package_versions": dict(
                package_versions
            ),
        },
        "protocol": {
            "status": config.protocol_status,
            "amendments_path": (
                config.amendments_path.as_posix()
            ),
        },
    }

    return metadata


def save_dataframe_csv(
    run_directory: Path,
    filename: str,
    dataframe: pd.DataFrame,
) -> Path:
    """Write a DataFrame to a new CSV file and return the created path."""
    output_path = run_directory / filename

    with output_path.open(
        mode="x",
        encoding="utf-8",
        newline="",
    ) as output_file:
        dataframe.to_csv(output_file)

    return output_path


def save_metadata_json(
    run_directory: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Serialize metadata to a new JSON file and return its path."""
    output_path = run_directory / "metadata.json"

    serialized = json.dumps(
        metadata,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )

    with output_path.open(
        mode="x",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        output_file.write(serialized)
        output_file.write("\n")

    return output_path


def save_run_artifacts(
    run_directory: Path,
    *,
    metadata: Mapping[str, Any],
    validation_forecasts: pd.DataFrame,
    validation_losses: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_forecasts: pd.DataFrame,
    test_losses: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> dict[str, Path]:
    """Persist the standard forecast, loss, metric, and metadata bundle."""
    dataframes = {
        "validation_forecasts": validation_forecasts,
        "validation_losses": validation_losses,
        "validation_metrics": validation_metrics,
        "test_forecasts": test_forecasts,
        "test_losses": test_losses,
        "test_metrics": test_metrics,
    }

    saved_paths = {}

    for artifact_name, dataframe in dataframes.items():
        saved_paths[artifact_name] = (
            save_dataframe_csv(
                run_directory,
                f"{artifact_name}.csv",
                dataframe,
            )
        )

    saved_paths["metadata"] = save_metadata_json(
        run_directory,
        metadata,
    )

    return saved_paths


def collect_git_state(
    repository_path: Path,
) -> dict[str, str | bool]:
    """Return the current commit and dirty status for a Git repository."""
    options = {
        "cwd": repository_path,
        "check": True,
        "capture_output": True,
        "text": True,
    }

    commit_result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        **options,
    )

    status_result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        **options,
    )

    commit = commit_result.stdout.strip()
    working_tree_dirty = bool(
        status_result.stdout.strip()
    )

    return {
        "commit": commit,
        "working_tree_dirty": working_tree_dirty,
    }


def collect_package_versions(
    package_names: tuple[str, ...] = RUNTIME_PACKAGES,
) -> dict[str, str]:
    """Return installed versions for named distributions."""
    versions = {}

    for package_name in package_names:
        versions[package_name] = package_version(
            package_name
        )

    return versions


def create_run_bundle(
    *,
    repository_path: Path,
    created_at: datetime,
    config: ExperimentConfig,
    snapshot_metadata: Mapping[str, Any],
    selected_ewma_name: str,
    selected_ewma_lambda: float,
    validation_forecasts: pd.DataFrame,
    validation_losses: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_forecasts: pd.DataFrame,
    test_losses: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> Path:
    """Create and persist one versioned experiment result bundle."""
    git_state = collect_git_state(
        repository_path
    )
    package_versions = (
        collect_package_versions()
    )

    run_id = build_run_id(
        created_at=created_at,
        commit_hash=str(
            git_state["commit"]
        ),
    )

    metadata = build_run_metadata(
        run_id=run_id,
        created_at=created_at,
        code_commit=str(
            git_state["commit"]
        ),
        working_tree_dirty=bool(
            git_state["working_tree_dirty"]
        ),
        config=config,
        snapshot_metadata=snapshot_metadata,
        selected_ewma_name=selected_ewma_name,
        selected_ewma_lambda=(
            selected_ewma_lambda
        ),
        package_versions=package_versions,
    )

    # Build provenance and metadata before creating the directory
    # so early failures do not leave an empty run directory behind.
    run_directory = create_run_directory(
        config.results_root,
        run_id,
    )

    save_run_artifacts(
        run_directory,
        metadata=metadata,
        validation_forecasts=(
            validation_forecasts
        ),
        validation_losses=validation_losses,
        validation_metrics=validation_metrics,
        test_forecasts=test_forecasts,
        test_losses=test_losses,
        test_metrics=test_metrics,
    )

    return run_directory
