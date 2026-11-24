Non-stationary Poisson point process $\{N(t) : t \ge 0\}$ with time-varying intensity $\lambda(t) \ge 0$ and integrated mean
$\Lambda(t) = \int_0^t \lambda(s)\,ds$, simulated via exact Lewis-Shedler thinning against a dominating majorant $\bar{\lambda} \ge \sup_{0 \le s \le T} \lambda(s)$:
$$\mathbb{P}(N(t) = k) = \frac{\Lambda(t)^k e^{-\Lambda(t)}}{k!}, \qquad p_{\text{accept}}(t_i) = \frac{\lambda(t_i)}{\bar{\lambda}} \in [0, 1].$$

Reference: Lewis & Shedler (1979), *Simulation of nonhomogeneous Poisson processes by thinning*,
DOI: 10.1002/nav.3800260304; [`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Thinning proposals step strictly forward via exponential interarrivals with rate $\bar{\lambda} > 0$, terminating in finite expected proposals $\bar{\lambda} T$ without time-grid discretization bias.

# Domain of Validity
Requires a validated [`IntensityProfile`] satisfying $\lambda(t) \ge 0$ for all $t \ge 0$ (`PoissonError::InvalidIntensityProfile`) and finite non-negative horizon $T \ge 0$ (`PoissonError::InvalidHorizon`).
