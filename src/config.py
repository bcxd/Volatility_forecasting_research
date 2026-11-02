from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    raw_path: Path

    results_root: Path = Path("results/runs")
    protocol_status: str = "exploratory"
    amendments_path: Path = Path(
        "docs/amendments.md"
    )

    symbol: str = "BTCUSDT"
    interval: str = "1d"
    sample_start: str = "2019-01-01"
    sample_end_exclusive: str = "2026-01-01"

    train_end: str = "2024-01-01"
    validation_end: str = "2025-01-01"
    test_end: str = "2026-01-01"

    variance_floor: float = 1e-12
    ewma_lambdas: tuple[float, ...] = (
        0.90,
        0.94,
        0.97,
        0.99,
    )

    garch_method: str = "SLSQP"
    garch_bounds: tuple[
        tuple[float | None, float | None],
        ...,
    ] = (
        (1e-12, None),
        (0.0, 1.0),
        (0.0, 1.0),
    )
    garch_persistence_margin: float = 1e-8
    garch_refit_every: int = 7