# Research Protocol: One-Day-Ahead BTCUSDT Variance Forecasting

## Protocol status

The 2025 test-period metrics were inspected during software development.
Results from this period are therefore **exploratory**, not untouched
confirmatory evidence. The disclosure and requirements for a future
confirmatory study are recorded in [amendments.md](amendments.md).

## 1. Research objective

Volatility is persistent, but persistence alone does not establish whether a
more complex conditional-variance model forecasts better than a simple
adaptive rule.

**Primary question:** Does Gaussian GARCH(1,1) achieve lower one-day-ahead
QLIKE loss than EWMA with its decay selected on the validation period?

**Secondary question:** Do GARCH(1,1) and validation-selected EWMA improve on
persistence and expanding historical-variance benchmarks?

Predictive value means lower out-of-sample forecast loss. This study does not
make a causal claim or test portfolio performance.

## 2. Design at a glance

| Item | Definition |
|---|---|
| Market | Binance Spot BTCUSDT |
| Frequency | Daily candles in UTC |
| Data sample | 2019-01-01 through 2025-12-31 |
| Return | Close-to-close log return, expressed as a decimal |
| Forecast target | Next-day conditional variance |
| Observed proxy | Next-day squared log return |
| Horizon | One day ahead |
| Window | Expanding |
| Train | Returns before 2024-01-01 |
| Validation | 2024-01-01 through 2024-12-31 |
| Test | 2025-01-01 through 2025-12-31 |

For closing price $C_t$,

$$
r_t = \log(C_t/C_{t-1}), \qquad y_t=r_t^2.
$$

The target is $h_{t+1\mid t}=\operatorname{Var}(r_{t+1}\mid\mathcal F_t)$,
and $y_{t+1}$ is its noisy daily proxy. It is not labeled intraday realized
variance. A forecast dated $t+1$ uses information available only through the
close of day $t$.

## 3. Models and selection

All models forecast variance on the same dates.

| Model | Specification or selection rule |
|---|---|
| Persistence | $\hat h_{t+1\mid t}=r_t^2$ |
| Historical variance | Expanding mean of squared returns available at $t$ |
| EWMA | Candidates $\lambda\in\{0.90,0.94,0.97,0.99\}$; select the lowest validation QLIKE |
| GARCH | Fixed zero-mean Gaussian GARCH(1,1) |

EWMA updates as

$$
\hat h_{t+1\mid t}
=\lambda\hat h_{t\mid t-1}+(1-\lambda)r_t^2.
$$

Its initial variance is the training-sample mean squared return. After
validation selection, the chosen EWMA state is carried into test rather than
reinitialized.

GARCH(1,1) is

$$
r_t=\epsilon_t, \qquad
h_t=\omega+\alpha\epsilon_{t-1}^2+\beta h_{t-1},
$$

with $\omega>0$, $\alpha\geq0$, $\beta\geq0$, and
$\alpha+\beta<1$. Parameters are estimated by Gaussian quasi-maximum
likelihood. The Gaussian assumption defines the estimation objective; it is
not a claim that unconditional Bitcoin returns are normally distributed.

### Locked numerical settings

| Setting | Value |
|---|---|
| Optimizer | SLSQP |
| Starting values | $\omega_0=0.05\overline{r^2}$, $\alpha_0=0.10$, $\beta_0=0.85$ |
| Bounds | $\omega\geq10^{-12}$, $0\leq\alpha\leq1$, $0\leq\beta\leq1$ |
| Persistence constraint | $\alpha+\beta\leq1-10^{-8}$ |
| Initial filtered variance | $\max(\operatorname{mean}(r^2),10^{-12})$ |
| Refit schedule | Every 7 forecast origins |
| Common variance floor | $10^{-12}$ |

GARCH is fit at the first forecast origin of validation and test and then on
the fixed seven-day schedule. Each fit uses only returns observed before its
forecast date. Between refits, variance is updated daily with the latest
observed return and the current parameters.

If a scheduled fit fails, the run stops and records the date, optimizer
message, and settings. Forecasts are not silently skipped and no fallback
optimizer is introduced after inspecting test results.

## 4. Evaluation and statistical inference

The primary loss is mean QLIKE on the variance scale:

$$
L_{\mathrm{QLIKE}}(y_t,\hat h_t)
=\log(\max(\hat h_t,\epsilon))
+\frac{y_t}{\max(\hat h_t,\epsilon)},
\qquad \epsilon=10^{-12}.
$$

MSE and MAE are secondary losses. Lower values are better. QLIKE effect sizes
are reported as mean loss differences, not percentage improvements. For
comparison $A$ versus $B$,

$$
d_t=L_{A,t}-L_{B,t},
$$

so a negative mean differential favors model $A$.

The five pre-specified comparisons are:

1. GARCH versus selected EWMA;
2. GARCH versus persistence;
3. GARCH versus historical variance;
4. selected EWMA versus persistence; and
5. selected EWMA versus historical variance.

Each comparison uses a two-sided asymptotic Diebold-Mariano test. The
long-run variance uses a Bartlett-kernel Newey-West HAC estimator with

$$
L=\left\lfloor4(T/100)^{2/9}\right\rfloor.
$$

No Harvey-Leybourne-Newbold small-sample correction is applied. Holm
adjustment controls the family-wise error rate across the five comparisons at
$\alpha=0.05$. Results report raw and adjusted p-values, the mean loss
difference, and the comparison direction. Missing or non-finite paired losses
stop the run rather than being silently removed.

## 5. Information and reproducibility rules

- Validation may select only the EWMA decay from the fixed candidate set.
- GARCH order, distribution, optimizer, constraints, and refit schedule are
  fixed before test evaluation.
- Test observations may enter later expanding-window forecasts only after
  they have been observed.
- Raw data are stored as an immutable snapshot with request metadata and a
  SHA-256 checksum.
- Every run stores dated forecasts, daily losses, aggregate metrics, EWMA
  selection, GARCH refit diagnostics, statistical results, code commit, and
  package versions.
- Changes made after viewing test results are labeled exploratory and recorded
  in [amendments.md](amendments.md).

## 6. Interpretation and limitations

The daily squared return is a noisy proxy for latent variance. The study
covers one asset, one forecast horizon, and one test year. Statistical
significance does not establish economic value, and failure to reject equal
loss does not prove model equivalence.

Additional GARCH orders, conditional-mean models, Student-t innovations,
intraday realized variance, regime analysis, and volatility targeting are
separate extensions. Each requires a written protocol and an unseen holdout
or another design that separates development from final evaluation.

## References

- Bollerslev, T., & Wooldridge, J. M. (1992). Quasi-maximum likelihood
  estimation and inference in dynamic models with time-varying covariances.
  *Econometric Reviews, 11*(2), 143-172.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
  *Journal of Business & Economic Statistics, 13*(3), 253-263.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies. *Journal of Econometrics, 160*(1), 246-256.
- White, H. (2000). A reality check for data snooping. *Econometrica, 68*(5),
  1097-1126.
