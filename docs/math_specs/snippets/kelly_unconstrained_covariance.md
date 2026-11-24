Zero-allocation unconstrained multi-asset covariance Kelly solver computing optimal continuous-time weights
$$\mathbf{w}^* = \Sigma^{-1}(\boldsymbol{\mu} - r\mathbf{1}), \qquad g_{\text{excess}}(\mathbf{w}^*) = \frac{1}{2}(\boldsymbol{\mu} - r\mathbf{1})^\top \Sigma^{-1}(\boldsymbol{\mu} - r\mathbf{1})$$
via symmetric positive-definite Cholesky factorization $\Sigma = L L^\top$ inside caller-supplied `PortfolioKellyWorkspace` buffers.

Reference: Kelly (1956), *Bell Syst. Tech. J.* 35(4), DOI: 10.1002/j.1538-7305.1956.tb03809.x;
Merton (1969), *Rev. Econ. Stat.* 51(3), DOI: 10.2307/1926560;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Symmetrizes off-diagonal covariance entries within standard tolerance and enforces Cholesky diagonal pivots $L_{ii} > 10^{-12}$ (`KellyError::NotPositiveDefinite`), avoiding explicit $\Sigma^{-1}$ matrix inversion.

# Domain of Validity
Requires $N \ge 1$ assets with matching `expected_returns`, `out_weights`, and $N \times N$ symmetric positive-definite `covariance` dimensions (`KellyError::DimensionMismatch`), finite returns and risk-free rate $r$, and `PortfolioKellyWorkspace` capacity $\ge N$.
