# Research Protocol Amendments

This file records changes and disclosures that affect interpretation of the pre-specified research protocol. Entries are append-only: a later correction should add a new amendment rather than erase the earlier record.

## Amendment 001 — Test-period exposure

- Date: 2026-10-25
- Status when recorded: the 2025 test-period metrics had already been inspected.
- Affected material: interpretation of all results calculated on the 2025 test period.

During software development, aggregate 2025 test metrics were viewed before the statistical-comparison and reproducible-artifact layers were completed. The 2025 period therefore no longer qualifies as an untouched confirmatory holdout.

Current 2025 findings must be labeled exploratory/development results. A future confirmatory claim requires a new unseen holdout period collected after the protocol and implementation are frozen, or another design that clearly separates model development from final evaluation.

## Amendment 002 — Statistical reporting details

- Date: 2026-10-25
- Status when recorded: fixed before implementation or inspection of Diebold–Mariano p-values.
- Affected material: Sections 8 and 9 of the research plan.

The following choices are fixed for the current implementation:

- QLIKE effect sizes are reported as mean loss differences, not relative percentage improvements.
- Relative percentage improvements may be reported for MSE and MAE when the benchmark mean loss is strictly positive.
- Diebold–Mariano comparisons are two-sided and asymptotic.
- For a comparison A versus B, the differential is QLIKE(A) minus QLIKE(B); a negative mean favors A.
- The long-run variance uses a Bartlett-kernel HAC estimator with bandwidth `floor(4 * (T / 100) ** (2 / 9))`.
- No Harvey–Leybourne–Newbold small-sample correction is applied.
- Holm adjustment is applied across the five pre-specified comparisons at family-wise alpha 0.05.

This amendment clarifies the method but does not restore the 2025 period to untouched-holdout status.
