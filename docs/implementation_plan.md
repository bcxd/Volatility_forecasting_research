# Implementation Plan

## Objective

Separate data acquisition, experiment execution, and reporting so that a saved market-data snapshot can be analyzed repeatedly without downloading or overwriting data and so that tables and figures can be rebuilt without refitting models.

## Target workflow

The finished project should expose three distinct actions:

1. Acquire a data snapshot when the requested market, interval, or date range changes.
2. Run an experiment from one explicit snapshot and save a versioned result bundle.
3. Generate tables and figures from a saved result bundle without refitting models.

Illustrative commands:

~~~text
python -m src.fetch_data
python -m src.pipeline --raw-path data/raw/btcusdt_1d_2019-01-01_2025-12-31.json
python -m src.reporting --run-dir results/runs/<run_id>
~~~

These are target interfaces. They should be introduced only when the corresponding phase is implemented and tested.

## Responsibility of `pipeline.py`

`pipeline.py` should be an orchestration layer. It may:

- call functions from focused modules;
- control the order of stages;
- pass data and configuration between stages;
- select the EWMA candidate using validation data;
- invoke evaluation and statistical comparison functions; and
- ask the artifact layer to save a complete run.

It should not directly contain:

- EWMA or GARCH equations;
- QLIKE, MSE, MAE, or Diebold–Mariano formulas;
- detailed JSON or CSV serialization logic;
- code for every plot; or
- automatic market-data downloads during a normal experiment run.

## Target module boundaries

~~~text
src/
|-- config.py              fixed experiment settings and paths
|-- data_fetch.py          Binance HTTP retrieval only
|-- data_io.py             snapshot and processed-data read/write
|-- data_validation.py     schema, timestamps, continuity, and value checks
|-- features.py            returns and variance proxy
|-- splits.py              chronological train/validation/test split
|-- benchmarks.py          persistence and historical-variance forecasts
|-- models.py              EWMA and GARCH estimation/forecasting
|-- evaluation.py          loss series and aggregate metrics
|-- statistical_tests.py   DM tests and Holm adjustment
|-- artifacts.py           versioned run output and metadata
|-- reporting.py           tables and figures from saved artifacts
`-- pipeline.py            orchestration only
~~~

Do not create empty placeholder modules. Add a module only when its responsibility is implemented with tests.

## Result bundle

Each experiment writes to a new directory rather than overwriting an earlier run:

~~~text
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
|-- garch_refit_log.csv
`-- figures/
~~~

`metadata.json` should include the source snapshot path and checksum, requested sample, split dates, candidate settings, selected EWMA decay, variance floor, GARCH settings, code commit, package versions, creation time, and protocol/amendment status.

## Delivery phases

### Phase 1 — Freeze and document the workflow

- Record test-period exposure.
- Lock QLIKE effect-size reporting and the exact DM/HAC rule.
- Define the snapshot policy and target architecture.

Acceptance: documentation is internally consistent, committed, and pushed; no numerical behavior changes.

### Phase 2 — Separate data acquisition

- Make downloading an explicit command.
- Generate a descriptive immutable snapshot filename.
- Write snapshot metadata and a SHA-256 checksum.
- Make normal experiment runs require an existing snapshot path.

Acceptance: running the experiment twice does not call Binance or modify the raw snapshot; acquisition and loading have focused tests.

### Phase 3 — Make the pipeline orchestration-only

- Introduce one experiment configuration object.
- Extract preparation, candidate generation, model selection, and final forecast stages where this reduces duplication.
- Keep each extracted function independently testable.

Acceptance: `pipeline.py` reads as a sequence of research stages and contains no model or metric formulas; all existing tests still pass.

### Phase 4 — Persist reproducible run artifacts

- Save dated forecasts, daily losses, aggregate metrics, model-selection output, GARCH refit diagnostics, and metadata.
- Refuse accidental overwrite of an existing run directory.

Acceptance: saved metrics can be recomputed from saved daily loss files; source snapshot and code version are traceable.

### Phase 5 — Add statistical comparison

- Implement the pre-specified DM/HAC calculation.
- Add Holm adjustment in the fixed five-comparison family.
- Test sign convention, bandwidth, degenerate differentials, missing values, and adjustment order.

Acceptance: deterministic unit tests and hand-calculated examples pass; results contain raw and adjusted p-values plus effect sizes.

### Phase 6 — Build reporting from saved results

- Produce summary tables and a small set of diagnostic forecast/loss plots.
- Read saved artifacts only; do not refit models in reporting.

Acceptance: deleting no source data and rerunning reporting reproduces the same tables and figures from the same run directory.

## Planned commit sequence

~~~text
docs: define reproducible experiment workflow
refactor(data): separate snapshot acquisition from experiments
refactor(pipeline): separate experiment stages
feat(results): persist reproducible run artifacts
feat(statistics): add DM tests and Holm adjustment
feat(reporting): generate tables and figures from saved runs
~~~
