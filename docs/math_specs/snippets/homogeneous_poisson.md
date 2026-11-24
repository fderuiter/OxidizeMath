Homogeneous Poisson counting process $\{N(t) : t \ge 0\}$ with constant intensity $\lambda > 0$,
evolving independent exponential interarrival increments $\Delta \tau_k \sim \operatorname{Exp}(\lambda)$ and exact count distribution
$$\mathbb{P}(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}, \qquad k \in \mathbb{N}_0.$$

Reference: Kingman (1993), *Poisson Processes*, DOI: 10.1093/oso/9780198536932.001.0001;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Evaluates log-space Stirling/Lanczos PMF terms $\ln \mathbb{P}(N(t) = k) = k \ln(\lambda t) - \lambda t - \ln(k!)$ to prevent factorial overflow for large $k$ or $\lambda t$.

# Domain of Validity
Requires strictly positive finite rate $\lambda > 0$ (`PoissonError::InvalidRate`) and finite non-negative observation horizon $t \ge 0$ (`PoissonError::InvalidHorizon`).
