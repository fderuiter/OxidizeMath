Zero-allocation projected-gradient solver for long-only multi-asset Kelly allocation with optional gross-leverage cap $L_{\max}$,
$$\max_{\mathbf{w} \ge \mathbf{0},\, \mathbf{1}^\top \mathbf{w} \le L_{\max}} \Bigl(\mathbf{w}^\top(\boldsymbol{\mu} - r\mathbf{1}) - \frac{1}{2}\mathbf{w}^\top \Sigma \mathbf{w}\Bigr),$$
iterating $\mathbf{w}_{k+1} = \Pi_{\mathcal{C}}\bigl(\mathbf{w}_k + \eta(\boldsymbol{\mu} - r\mathbf{1} - \Sigma \mathbf{w}_k)\bigr)$ inside `PortfolioKellyWorkspace` with zero heap allocations.

Reference: Merton (1969), *Rev. Econ. Stat.* 51(3), DOI: 10.2307/1926560;
Duchi et al. (2008), *ICML*, DOI: 10.1145/1390156.1390191;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Uses Lipschitz step bound $\eta = 0.95 / \max(\|\Sigma\|_\infty, 10^{-12}) < 2 / \lambda_{\max}(\Sigma)$ for guaranteed monotone ascent on the strictly concave quadratic objective, paired with exact $O(N \log N)$ Euclidean capped-simplex projection and $\|\mathbf{w}_{k+1} - \mathbf{w}_k\|_\infty \le 10^{-10}$ stopping tolerance.

# Domain of Validity
Requires $N \ge 1$ assets, finite expected returns and risk-free rate $r$, symmetric positive-definite $N \times N$ `covariance` (`KellyError::NotPositiveDefinite`), finite non-negative `max_gross_leverage` ($L_{\max} \ge 0$) when `Some`, and `PortfolioKellyWorkspace` capacity $\ge N$.
