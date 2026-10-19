Research Plan: One-Day-Ahead BTCUSDT Variance Forecasting

1. Motivation

Observation from prior research

Financial return volatility is commonly described as persistent: periods of high volatility tend to be followed by periods of high volatility. This established feature motivates conditional-variance forecasting. It is background from prior research, not a finding of this project.

A descriptive analysis may examine the autocorrelation of returns and squared returns in the study sample. These diagnostics will be reported as descriptions of the sample and will not be used to change the confirmatory model set after inspecting the test period.

Puzzle

If volatility is persistent, does a GARCH model provide better one-day-ahead variance forecasts than a simpler EWMA model and basic historical benchmarks?

EWMA and GARCH both use past information to update variance forecasts, but they impose different structures. Theory alone does not determine which will forecast better in this dataset.

Mechanism

EWMA updates its variance estimate by applying an exponentially declining weight to past squared returns. GARCH(1,1) updates conditional variance using both the latest squared return shock and the previous conditional variance.

These different update rules may produce different forecasts after large return shocks. The direction and size of any difference are empirical questions; this plan does not assume that either method responds faster or forecasts better.

2. Research Question and Hypotheses

Primary research question

For one-day-ahead forecasts of daily BTCUSDT variance, does Gaussian GARCH(1,1) achieve lower out-of-sample QLIKE loss than EWMA with its decay parameter selected on the validation period?

Secondary research question

Do EWMA and GARCH(1,1) improve out-of-sample forecast performance relative to persistence and expanding historical-variance benchmarks?

Statistical hypotheses

For each pre-specified pairwise comparison, the null hypothesis is equal expected forecast loss. The primary comparison is GARCH(1,1) versus validation-selected EWMA. The test is two-sided because the prior theory does not establish which model should perform better.

“Predictive value” in this study means lower out-of-sample forecast loss. The study does not test causal effects or prove that a forecast improves investment performance.

3. Data and Forecast Target

Data source and sample

The study uses Binance Spot BTCUSDT daily candles, sampled in UTC, for candle dates from 2019-01-01 through 2025-12-31.

The source dataset is saved as a raw snapshot. The analysis should record the retrieval date, data filename, code version or commit, and package versions so the reported forecasts can be reproduced. Missing or invalid candles will be reported; they will not be silently interpolated.

Return definition

For daily closing price (C_t), the close-to-close log return is

$$
r_t=\log(C_t/C_{t-1}).
$$

Returns are expressed as decimals, not percentages. Because calculating the first return requires a prior close, the return series begins on 2019-01-02 unless a 2018-12-31 close is added.

Target

The target is the next-day conditional variance,

$$
h_{t+1|t}=\operatorname{Var}(r_{t+1}\mid\mathcal F_t).
$$

The observed proxy is the squared daily return,

$$
y_{t+1}=r_{t+1}^{2}.
$$

The core study assumes a zero conditional mean, so the squared return is used as a noisy proxy for conditional variance. It is not labeled intraday realized variance. Comparing forecasts against squared returns requires a loss suitable for an imperfect volatility proxy; QLIKE is selected as the primary loss. See Patton (2011).

Forecasts are issued after the daily candle for day (t) closes, using information available through that close. The forecast is aligned with the date of the return being predicted.

4. Data Splits

Splits refer to the date of the return being forecast and use an inclusive start date and exclusive end date:

Training: 2019-01-01 to before 2024-01-01.

Validation: 2024-01-01 to before 2025-01-01.

Test: 2025-01-01 to before 2026-01-01.

The return series will start on 2019-01-02 if the dataset contains no preceding 2018 close. The validation period is used to select EWMA decay. The test period is reserved for the final evaluation and is not used to select models or settings.

5. Benchmarks and Candidate Models

All models issue one-day-ahead variance forecasts on the same evaluation dates.

Benchmarks

Persistence

$$
\hat h_{t+1|t}=r_t^2.
$$

Expanding historical variance

$$
\hat h_{t+1|t}
=\frac{1}{N_t}\sum_{s\leq t}r_s^2,
$$

where only returns observed by forecast origin (t) are included.

EWMA

The EWMA forecast is updated as

$$
\hat h_{t+1|t}
=\lambda\hat h_{t|t-1}+(1-\lambda)r_t^2,
\qquad 0\leq\lambda\leq1.
$$

The initial variance is the mean squared return in the available training sample. The pre-specified decay candidates are the fixed values $\lambda\in\{0.90, 0.94, 0.97, 0.99\}$. Candidates are specified directly as decay values, not generated from a half-life grid:

$$
\lambda=0.5^{1/H}.
$$

The candidate set is fixed before validation results are examined. The decay value with the lowest mean validation QLIKE is selected. After selection, the same EWMA state is carried forward from validation into test; it is not reinitialized at the start of test.

GARCH(1,1)

The core GARCH model has a zero conditional mean:

$$
r_t=\epsilon_t,\qquad
\epsilon_t=\sqrt{h_t}z_t,\qquad
z_t\sim N(0,1),
$$

$$
h_t=\omega+\alpha\epsilon_{t-1}^{2}+\beta h_{t-1},
$$

with (\omega>0), (\alpha\geq0), (\beta\geq0), and (\alpha+\beta<1).

The core study evaluates GARCH(1,1) only. Other lag orders are not selected using the test period.

6. GARCH Estimation and Walk-Forward Protocol

GARCH parameters are estimated by minimizing the Gaussian negative log-likelihood on returns available at each refit date. Under the zero-mean specification, the objective is

$$
\frac12\sum_t
\left[
\log(2\pi)+\log(h_t)+\frac{r_t^2}{h_t}
\right].
$$

This normal-based objective is used to estimate variance parameters. It is separate from QLIKE, which evaluates forecasts. The normal innovation assumption is conditional on the past; it is not a claim that the unconditional distribution of BTCUSDT returns is normal. If normality is only an approximation, the estimator is described as Gaussian quasi-maximum likelihood.

The following estimation choices are proposed for the core implementation:

Optimizer: SLSQP.

Starting values: $\omega_0=0.05\bar r^2$, $\alpha_0=0.10$, and $\beta_0=0.85$, where $\bar r^2$ is the training-sample mean squared return.

Bounds: $\omega\geq10^{-12}$, $0\leq\alpha\leq1$, and $0\leq\beta\leq1$.

Persistence constraint: $\alpha+\beta\leq1-10^{-8}$. This matches the constraint used by the implementation.

Initial filtered variance: $h_1=\max(\operatorname{mean}(r_t^2),10^{-12})$.

The numerical constants above use decimal-return variance units and must remain fixed across validation and test.

Forecasts use an expanding estimation window. GARCH parameters are re-estimated every seven forecast origins, beginning with a fit at the first forecast origin of each evaluation period. At each refit, only returns observed before that forecast date are used. Between refits, the variance forecast is updated daily using the latest observed return and the most recently estimated parameters.

At the start of the test period, the available history includes training and validation returns through 2024-12-31. Earlier test returns may enter subsequent forecasts and scheduled refits after they have been observed.

If a scheduled fit fails, the run stops and records the forecast date, optimizer message, and settings. Forecasts will not be silently skipped, and a different optimizer or fallback rule will not be introduced after inspecting test performance.

7. Validation and Model Selection
The EWMA decay candidates are $\lambda\in\{0.90, 0.94, 0.97, 0.99\}$, as specified in Section 5.
The EWMA decay is selected by the lowest mean validation QLIKE across these four pre-specified candidates.

GARCH(1,1) order, distribution, optimizer, constraints, and initialization are fixed in advance. GARCH parameters are re-estimated as specified in the walk-forward protocol; the test period is not used to choose these settings.

After validation selection, the chosen EWMA decay and the core GARCH specification are frozen for test evaluation. Validation results may be used for model selection as defined here; test results may not.

8. Forecast Evaluation

Primary metric

The primary metric is mean QLIKE on the variance scale:

$$
L_{\mathrm{QLIKE}}(y_t,\hat h_t)
=\log(\max(\hat h_t,\epsilon))
+\frac{y_t}{\max(\hat h_t,\epsilon)},
$$

where $\epsilon=10^{-12}$ is a common positive forecast floor in decimal-return variance units. The same floor is applied to all models before scoring.

Secondary metrics

MSE between squared-return proxy and forecast variance.

MAE between squared-return proxy and forecast variance.

Lower loss indicates better performance. Results will include average loss, pairwise loss differences, and relative loss improvement. Volatility forecasts may be reported as $\sqrt{\hat h_t}$, but all primary evaluation is on the variance scale.

9. Statistical Comparison

The primary comparison is GARCH(1,1) versus the validation-selected EWMA. Secondary comparisons are each candidate model versus each benchmark:

GARCH(1,1) versus EWMA.

GARCH(1,1) versus persistence.

GARCH(1,1) versus expanding historical variance.

EWMA versus persistence.

EWMA versus expanding historical variance.

For each pair, the loss differential is the first model’s daily QLIKE minus the second model’s daily QLIKE. A negative average differential favors the first model.

Forecast accuracy is compared using a two-sided Diebold–Mariano test of equal expected loss. The variance of the daily loss differential will use a Bartlett-kernel Newey–West HAC estimator with a pre-specified automatic bandwidth rule. The implementation and package version will be recorded before final test evaluation; bandwidths will not be chosen by inspecting test p-values.

Holm adjustment is applied across the five pre-specified comparisons at a family-wise significance level of 5%. Statistical significance will be reported separately from the size of the loss difference and any practical interpretation. See Diebold and Mariano (1995).

10. Robustness and Future Extensions

The following are outside the confirmatory core study and will be treated as separate analyses.

Alternative EWMA decay and GARCH lag orders

A later study may compare additional half-lives or GARCH orders. The GARCH equation and notation must define explicitly how many lags apply to squared shocks and how many apply to past conditional variances, because order notation varies across sources and software. A candidate set and validation selection rule must be fixed before using the holdout data for that study.

Alternative conditional-mean assumptions

A later study may compare zero mean with a constant conditional mean estimated within each expanding training window. An AR(1) mean may be considered if justified and specified in advance. Under a nonzero mean, variance recursion uses residuals (\epsilon_t=r_t-\mu_t), and the proxy and evaluation target must be defined consistently.

Alternative innovation distributions

A later study may compare Gaussian innovations with standardized Student-t innovations. If Student-t is used, its degrees of freedom and variance normalization must be specified, and any distributional evaluation should use an appropriate density score in addition to variance forecast metrics.

Intraday realized variance

A separate extension may construct daily realized variance from intraday returns:

$$
RV_t=\sum_{j=1}^{M}r_{t,j}^{2}.
$$

This changes the evaluation target and requires intraday data. The sampling interval, UTC day boundary, data source, missing-bar policy, and treatment of market microstructure noise must be fixed before evaluation. Using intraday realized variance only as an evaluation proxy is distinct from using intraday data as model input. See Andersen et al. (2001).

Regime analysis and economic value

Regime comparisons require a regime definition that does not use future information. Volatility targeting or portfolio analysis is a separate extension and must define the position rule, volatility target, leverage limit, rebalancing schedule, transaction costs, and performance outcomes before backtesting. Forecast accuracy alone does not establish economic value.

11. Amendments, Reproducibility, and Data-Snooping Control

The core protocol will be saved and versioned before test results are examined. Any amendment will record its date, reason, affected hypotheses, and whether test results had already been inspected.

If a change is made after examining test results, the resulting analysis will be labeled exploratory and will not be presented as a confirmatory test of the original protocol. Confirmatory claims for a materially expanded model set or a changed target require a new untouched holdout period or another design that separates model selection from final evaluation.

The project will retain the raw data snapshot, forecast output by date and model, selected validation settings, optimizer status and messages, metric calculations, statistical test output, code commit, and package versions. Forecasts with missing or non-positive values will be reported according to the pre-specified failure policy rather than silently removed.

12. References

Andersen, T. G., Bollerslev, T., Diebold, F. X., & Ebens, H. (2001). “The Distribution of Realized Stock Return Volatility.” Journal of Financial Economics, 61(1), 43–76. https://doi.org/10.1016/S0304-405X(01)00055-1

Bollerslev, T., & Wooldridge, J. M. (1992). “Quasi-Maximum Likelihood Estimation and Inference in Dynamic Models with Time-Varying Covariances.” Econometric Reviews, 11(2), 143–172. https://doi.org/10.1080/07474939208800229

Diebold, F. X., & Mariano, R. S. (1995). “Comparing Predictive Accuracy.” Journal of Business & Economic Statistics, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599

Patton, A. J. (2011). “Volatility Forecast Comparison Using Imperfect Volatility Proxies.” Journal of Econometrics, 160(1), 246–256. https://doi.org/10.1016/j.jeconom.2010.03.034

White, H. (2000). “A Reality Check for Data Snooping.” Econometrica, 68(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152