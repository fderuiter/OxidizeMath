Adaptive Moment Estimation (Adam) first-order optimizer:

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t, \quad
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^{\odot 2}, \quad
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$

Reference: Kingma & Ba (2015), arXiv:1412.6980; [`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Initialization bias correction $\hat{\mathbf{m}}_t = \mathbf{m}_t/(1 - \beta_1^t)$, $\hat{\mathbf{v}}_t = \mathbf{v}_t/(1 - \beta_2^t)$ prevents early-step scaling drift while $\epsilon > 0$ guards division by zero on flat coordinates.

# Domain of Validity
Requires $\eta > 0$, exponential decay factors $0 \le \beta_1, \beta_2 < 1$, and numerical floor $\epsilon > 0$.
