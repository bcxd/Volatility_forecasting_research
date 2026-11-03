from pathlib import Path

from src.config import ExperimentConfig


def test_experiment_config_contains_locked_protocol(tmp_path):
    raw_path = tmp_path / "snapshot.json"

    config = ExperimentConfig(raw_path=raw_path)

    assert config.raw_path == raw_path

    assert config.symbol == "BTCUSDT"
    assert config.interval == "1d"
    assert config.sample_start == "2019-01-01"
    assert config.sample_end_exclusive == "2026-01-01"

    assert config.train_end == "2024-01-01"
    assert config.validation_end == "2025-01-01"
    assert config.test_end == "2026-01-01"

    assert config.variance_floor == 1e-12
    assert config.familywise_alpha == 0.05
    assert config.ewma_lambdas == (
        0.90,
        0.94,
        0.97,
        0.99,
    )

    assert config.garch_method == "SLSQP"
    assert config.garch_bounds == (
        (1e-12, None),
        (0.0, 1.0),
        (0.0, 1.0),
    )
    assert config.garch_persistence_margin == 1e-8
    assert config.garch_refit_every == 7
    assert config.results_root == Path("results/runs")
    assert config.protocol_status == "exploratory"
    assert config.amendments_path == Path(
        "docs/amendments.md"
    )
