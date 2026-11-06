# Project Architecture

## Purpose

The project separates data acquisition, experiment execution, and reporting.
A saved market-data snapshot can be analyzed repeatedly without contacting
Binance, and tables and figures can be rebuilt from a saved run without
refitting models.

## Workflow

```text
Binance API
    |
    v
fetch_data.py  -->  immutable raw JSON + snapshot metadata
    |
    v
pipeline.py    -->  validated data + versioned experiment run
    |
    v
generate_report.py  -->  tables and figures from the saved run
```

The three actions are explicit:

```powershell
python -m src.fetch_data --symbol BTCUSDT --interval 1d --start 2019-01-01 --end 2026-01-01
python -m src.pipeline --raw-path data/raw/btcusdt_1d_2019-01-01_2026-01-01.json
python -m src.generate_report --run-directory results/runs/<run_id>
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Fixed experiment settings and paths |
| `data_fetch.py` | Binance HTTP retrieval |
| `data_io.py` | Snapshot and processed-data input/output |
| `data_validation.py` | Schema, timestamps, continuity, and value checks |
| `features.py` | Log returns and squared-return variance proxy |
| `splits.py` | Chronological train, validation, and test split |
| `benchmarks.py` | Persistence and historical-variance forecasts |
| `models.py` | EWMA and GARCH estimation and forecasting |
| `evaluation.py` | Daily losses and aggregate metrics |
| `statistical_tests.py` | DM/HAC tests and Holm adjustment |
| `artifacts.py` | Versioned run output and metadata |
| `pipeline.py` | Experiment orchestration |
| `reporting/loaders.py` | Load and validate saved artifacts |
| `reporting/tables.py` | Build reporting tables |
| `reporting/plots.py` | Build diagnostic figures |
| `generate_report.py` | Reporting command and output orchestration |

`pipeline.py` controls the sequence and passes data between stages. Model
equations, loss formulas, statistical formulas, serialization details, and
plotting logic remain in their focused modules.

## Data and result artifacts

A fetch writes an immutable pair:

```text
data/raw/<snapshot>.json
data/raw/<snapshot>.metadata.json
```

Snapshot metadata records the endpoint, request, retrieval time, returned date
range, row count, filename, and SHA-256 checksum. Existing snapshots are not
silently overwritten.

Each experiment creates a new run directory:

```text
results/runs/<timestamp>_<short_commit>/
|-- metadata.json
|-- validation_targets.csv
|-- validation_forecasts.csv
|-- validation_losses.csv
|-- validation_metrics.csv
|-- test_targets.csv
|-- test_forecasts.csv
|-- test_losses.csv
|-- test_metrics.csv
|-- dm_results.csv
`-- garch_refit_log.csv
```

The run metadata connects the result to its source snapshot, experiment
settings, protocol status, code commit, working-tree state, and package
versions.

Reporting reads one run and creates a separate bundle:

```text
results/reports/<run_id>/
|-- metric_summary.csv
|-- dm_summary.csv
|-- cumulative_qlike.csv
|-- volatility_forecasts.png
|-- cumulative_qlike.png
`-- report_metadata.json
```

## Design invariants

- A normal experiment run never downloads data or changes the raw snapshot.
- Forecasts use only information available before their target date.
- Validation selects EWMA decay; test data do not select settings.
- Run and report directories refuse accidental overwrite.
- Missing, non-finite, or invalid model output raises an explicit error.
- Reporting reads saved artifacts only and never refits a model.
- Public reporting imports are exposed through `src.reporting`.

## Verification

The test suite covers successful paths, invalid inputs, temporal alignment,
failure policies, artifact integrity, statistical calculations, and report
generation. Binance calls are mocked.

```powershell
python -m pytest -q
```

Research choices are defined in [research_plan.md](research_plan.md), protocol
changes in [amendments.md](amendments.md), and findings in
[results.md](results.md).
