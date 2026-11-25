# Canonical Mathematical Specification: `pure_math`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/pure_math`](../../crates/pure_math/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace manuscripts.

## 1. Arbitrary-Precision Modular Arithmetic & Number Theory (`math_types`, `number_theory`)

### Governing Equations

Binary modular exponentiation (`Integer::pow_mod`) and extended Euclidean inversion (`Integer::invert`) evaluate residue classes in $\mathbb{Z}/m\mathbb{Z}$:

$$
a^e \bmod m = \begin{cases} \bigl(a \bmod m\bigr)^e \bmod m & e \ge 0, \\ \bigl(a^{-1} \bmod m\bigr)^{|e|} \bmod m & e < 0,\ \gcd(a, m) = 1. \end{cases}
$$

Partition $q$-series, Hurwitz-Kronecker class numbers, and AMBS/UALBF quasiperfect modularity checks (`solve_quasiperfect_modularity`) enforce divisor-sum congruences $\sigma(n) = 2n + 1$.

### Citations

- BibTeX keys: [`pushpa2022eisenstein`](../../papers/references.bib#pushpa2022eisenstein) (DOI: `10.1007/s40065-022-00360-6`), [`nath2025arithmetic`](../../papers/references.bib#nath2025arithmetic) (DOI: `10.1016/j.jsc.2025.102555`), [`CohenHagis1982`](../../papers/references.bib#CohenHagis1982), [`cohen1993course`](../../papers/references.bib#cohen1993course).
- Manuscripts: [`papers/partitions_implementation.tex`](../../papers/partitions_implementation.tex), [`papers/ualbf_quasiperfect.tex`](../../papers/ualbf_quasiperfect.tex), [`papers/quasiperfect_alcf.tex`](../../papers/quasiperfect_alcf.tex), [`papers/hurwitz_kronecker.tex`](../../papers/hurwitz_kronecker.tex).

### Stability

Arbitrary-precision `dashu::IBig` arithmetic eliminates 64-bit integer overflow across all intermediate squarings in $O(\log |e|)$ steps; negative exponents fail closed with `NumberTheoryError::NoModularInverse` whenever $\gcd(a, m) \ne 1$.

### Domain of Validity

Requires non-zero modulus $m \ne 0$ (`NumberTheoryError::ZeroModulus`), radix $b \in [2, 36]$ for string parsing, and coprime $(a, m)$ when $e < 0$ or calling `Integer::invert`.

```rust
use math_explorer::math_types::Integer;

let base = Integer::from(3);
let exp = Integer::from(20);
let modulus = Integer::from(100);
assert_eq!(base.pow_mod(&exp, &modulus).unwrap(), Integer::from(1));
```

## 2. Optimization, Linear Algebra, ODE/PDE & Stochastic Solvers (`analysis`, `differential_geometry`, `statistics`)

### Governing Equations

First-order parameter optimizers (`SGD`, `Adam`) and $L_1$-regularized least squares (`L1RegularizedLeastSquares`) update iterate vectors $\theta_t$ under objective gradients $g_t = \nabla_\theta \mathcal{L}(\theta_t)$:

$$
\theta_{t+1}^{\text{SGD}} = \theta_t - \eta g_t, \qquad \hat{m}_t = \frac{\beta_1 m_{t-1} + (1-\beta_1)g_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{\beta_2 v_{t-1} + (1-\beta_2)g_t^2}{1 - \beta_2^t}, \quad \theta_{t+1}^{\text{Adam}} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Initial/boundary value, Riemannian/Lorentzian manifold, and combinatorial/stochastic/financial solvers evolve $\dot{y} = f(t, y)$ (`OdeModel`, `Solver`, symplectic `ImplicitMidpoint`, `GaussLegendre4`, `StormerVerlet`, `Yoshida4`), intrinsic cotangent Laplace-Beltrami meshes $(\Delta_S u)_i = A_i^{-1}\sum_{j \in \mathcal{N}(i)} w_{ij}(u_j - u_i)$ (`CotangentLaplacian`), orientation-preserving $\mathrm{SU}(1,1)$ Möbius disk isometries $z \mapsto (a z + b)/(\bar{b} z + \bar{a})$ across Poincaré/Beltrami-Klein models and $\{p, q\}$ geodesic billiards (`HyperbolicModel`, `Su11Matrix`, `HyperbolicBilliardSystem`), closed-form 4D Schwarzschild connections $\Gamma^\mu_{\alpha\beta}$ (`SchwarzschildConnection4D`), $\partial_t u = \alpha \Delta u$ (`HeatEquationSolver`), $\partial_{tt} u = c^2 \Delta u - \gamma \partial_t u$ (`Wave2DSolver`), $-\Delta u = f$ (`GreenFunctionSolver1D`, `RectangularLaplaceSolver`), $O(N)$ complex Thomas tridiagonal systems $a_{i-1} x_{i-1} + b_i x_i + c_i x_{i+1} = d_i$ (`solve_tridiagonal_factored`, `solve_tridiagonal_in_place`), Kruskal minimum spanning forests $\min_{F \subseteq E} \sum_{e \in F} w_e$ (`solve_minimum_spanning_tree_in_place`), Edmonds-Karp `s-t` max-flow/min-cut duality $\max_f |f| = \min_{(S,T)} c(S,T)$ (`solve_max_flow_min_cut_in_place`), exact Ornstein-Uhlenbeck Gaussian transitions (`ExactOuSolver`, `EulerMaruyama`), homogeneous/Lewis-Shedler thinning Poisson processes (`HomogeneousPoisson`, `InhomogeneousPoisson`), Erlang birth-death queues (`Mm1Queue`, `MmcQueue`), exact jump Markov trajectories (`GillespieSolver`), discrete/hidden Markov chains (`HiddenMarkovModel`, `solve_parrondo_ratchet`), Lundberg ruin adjustment equations (`solve_lundberg_ruin`), non-parametric `PAVA` isotonic regression and damped-Newton logistic/softmax recalibrators (`IsotonicCalibrator`, `PlattScaler`, `TemperatureScaler`), Genz separation-of-variables Gaussian copulas $C_\Sigma(u_1,\dots,u_d) = \Phi_\Sigma(\Phi^{-1}(u_1),\dots,\Phi^{-1}(u_d))$ (`GaussianCopula`), multi-asset Cholesky covariance and Duchi-projected long-only Kelly allocations $\mathbf{w}^* = \Sigma^{-1}(\boldsymbol{\mu} - r\mathbf{1})$ paired with log-normal Value-at-Risk / Conditional Value-at-Risk (`solve_unconstrained_kelly_into`, `solve_long_only_kelly_into`, `parametric_lognormal_tail_risk`), univariate Acerbi-Tasche empirical, Gaussian, and 4th-order Cornish-Fisher `VaR`/`CVaR`, underwater drawdowns, and `Sharpe`/`Sortino`/`Calmar` ratios (`historical_cvar`, `cornish_fisher_cvar`, `summarize_return_risk`, DOI: `10.1016/S0378-4266(02)00283-2`, DOI: `10.2307/2332008`), and closed-form Black-Scholes-Merton European option valuation and analytical Greeks (`black_scholes_merton`).

### Citations

- BibTeX keys: [`kingma2014adam`](../../papers/references.bib#kingma2014adam) (`arXiv:1412.6980`), [`nalgebra`](../../papers/references.bib#nalgebra), [`rabiner1989tutorial`](../../papers/references.bib#rabiner1989tutorial) (DOI: `10.1109/5.18626`), [`norris1997markov`](../../papers/references.bib#norris1997markov), [`bafna1999`](../../papers/references.bib#bafna1999).
- Manuscripts: [`papers/advanced_linear_algebra.tex`](../../papers/advanced_linear_algebra.tex), [`papers/tensors.tex`](../../papers/tensors.tex), [`papers/graph_parameters_rust.tex`](../../papers/graph_parameters_rust.tex), [`papers/algorithmic_information_rust.tex`](../../papers/algorithmic_information_rust.tex).

### Stability

`SGD` converges on $L$-smooth convex objectives for $0 < \eta < 2/L$; `Adam` guards denominator division via $\epsilon > 0$ and checked step-counter conversions; explicit 2D wave/heat stencils enforce Courant-Friedrichs-Lewy bounds ($c\,\Delta t \sqrt{\Delta x^{-2} + \Delta y^{-2}} \le 1$); LU/QR linear solvers reject singular or rank-deficient systems via typed `LinearAlgebraError`.

### Domain of Validity

Requires finite matching matrix/vector dimensions, positive step sizes $\Delta t > 0$, $\beta_1, \beta_2 \in [0, 1)$ and $\epsilon > 0$ for `Adam`, row-stochastic transition matrices ($\sum_j P_{ij} = 1$, $P_{ij} \ge 0$) for Markov models, and positive net edge for Lundberg ruin solvers.

```rust
use math_explorer::nalgebra::DVector;
use math_explorer::pure_math::analysis::optimization::{Optimizer, SGD};

let mut sgd = SGD::new(0.1_f64);
let mut theta = DVector::from_vec(vec![2.0, -4.0]);
let grad = DVector::from_vec(vec![4.0, -8.0]);
sgd.update_vector(0, &mut theta, &grad).unwrap();
assert!((theta[0] - 1.96).abs() < 1e-12);
```
