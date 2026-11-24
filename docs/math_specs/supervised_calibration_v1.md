# Supervised Probabilistic Calibration & Proper Scoring Specification (`spec:supervised_calibration_v1`)

- **Spec ID**: `spec:supervised_calibration_v1`
- **Traceability Module Keys**: `supervised_calibration`, `calibration`

## Governing Equations

Given predicted probabilities $\hat{p}_i \in [0, 1]$ and binary outcomes $y_i \in \{0, 1\}$ over $N \ge 1$ samples, the Brier score and its Murphy decomposition ($\mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC}$) with base rate $\bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$ and reliability bins $\{B_m\}_{m=1}^M$ satisfy:

$$
\mathrm{BS} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2 = \underbrace{\frac{1}{N}\sum_{m=1}^M |B_m|\bigl(\bar{p}_m - \bar{y}_m\bigr)^2}_{\mathrm{REL}} - \underbrace{\frac{1}{N}\sum_{m=1}^M |B_m|\bigl(\bar{y}_m - \bar{y}\bigr)^2}_{\mathrm{RES}} + \underbrace{\bar{y}(1 - \bar{y})}_{\mathrm{UNC}}
$$

Sample-weighted Expected Calibration Error (`ECE`) and single-parameter post-hoc temperature scaling over logits $\mathbf{z}_i \in \mathbb{R}^K$ ($T > 0$) are given by:

$$
\mathrm{ECE} = \sum_{m=1}^M \frac{|B_m|}{N}\,\bigl|\bar{y}_m - \bar{p}_m\bigr|, \qquad \hat{\mathbf{q}}_i(T) = \mathrm{softmax}\!\left(\frac{\mathbf{z}_i}{T}\right)
$$

## Numerical Stability & Error Bounds

- Symmetrically clipped cross-entropy (`binary_log_loss`, `multiclass_log_loss`) clamps probabilities to $[\varepsilon, 1 - \varepsilon]$ ($\varepsilon \in (0, 0.5)$) to prevent $-\ln(0) = +\infty$ divergence, and temperature scaling evaluates log-sum-exp shifted by $\max_k (z_{i,k} / T)$ to eliminate exponential overflow.
- Pool Adjacent Violators (`PAVA`) isotonic regression merges adjacent monotonicity violations in $\mathcal{O}(N)$ single-pass block reductions without heap re-allocations once sorted.

## Domain of Validity

- Non-empty finite sample arrays ($N \ge 1$) of equal length, forecast probabilities $\hat{p}_i \in [0, 1]$ (`UnitInterval`), bin count $M \ge 1$, clipping tolerance $\varepsilon \in (0, 0.5)$, and strictly positive temperature $T > 0$.

## References

- `@brier1950verification` — Brier (1950), *Verification of forecasts expressed in terms of probability*, `DOI:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2`.
- `@guo2017calibration` — Guo, Pleiss, Sun & Weinberger (2017), *On Calibration of Modern Neural Networks*, `arXiv:1706.04599`.
