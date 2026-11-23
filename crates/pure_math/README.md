# Pure Math

Rigorous abstract algebra, number theory, differential geometry, and statistical algorithms.

## 1. Overview & Architecture

The `pure_math` crate provides rigorous algorithmic implementations of foundational mathematical structures. Unlike domain crates that focus on applied physical phenomena, `pure_math` develops the core mathematical machinery from abstract algebra, analysis, and topology.

Implementations prioritize mathematical correctness, transparent algorithmic structure, and educational clarity, serving as canonical reference implementations across the OxidizeMath ecosystem.

### Module Structure

- **`algebra`**: Abstract algebraic structures and type hierarchies:
  - Semigroups, monoids, groups (`Group`, `AbelianGroup`, `Zn<N>`, `is_subgroup`, `generate_left_coset`), rings, and finite fields (`Fp`).
  - Polynomial rings and polynomial arithmetic over arbitrary fields.
- **`number_theory`**: Prime number generation (`primes_up_to`), primality testing, integer partition theory, and Q-series.
- **`analysis`**: Differential equation solvers (`OdeSystem`, explicit Runge-Kutta and zero-allocation symplectic `StormerVerlet`, `Yoshida4`, and `GaussLegendre4` steppers), zero-allocation bracketed and derivative root-finding (`roots::Bisection`, `roots::NewtonRaphson`, `roots::Brent`), 1D and zero-allocation 2D partial differential equation solvers (`pde::tridiagonal` for `core`-slice complex Thomas sweeps, `pde::wave_2d::Wave2DSolver` for hyperbolic wave tanks, Kirchhoff-Love biharmonic Chladni plates, Chladni sand drift, Marching-Squares nodal contours, and pre-planned 2D FFT spectra), numerical integration, and calculus algorithms.
- **`differential_geometry`**: Manifolds, metric tensors, and Riemannian curvature.
- **`hyperbolic`**: Poincaré disk and upper half-plane models, `PSU(1,1)` / `PSL(2,R)` Möbius isometries, `{p,q}` / `{infinity,q}` tilings, and chaotic geodesic billiards.
- **`elliptic_curves`**: Zero-allocation short-Weierstrass curve group arithmetic ($P + Q$, $\[k\]P$, cyclic orbits) over $\mathbb{R}$ and prime fields $\mathbb{F}_p$, Frobenius traces $a_p$, Hasse-Weil verification, Breuer modular polynomial valuation bounds, and bridges to binary quadratic forms (`BQF`), Hurwitz-Kronecker class numbers, and modular $q$-series (`QSeries<i64>`).
- **`graph_theory`**: Network graph structures, centrality metrics, structural parameters (`degeneracy`, 2-approx `vertex_cover`, and exact bitmask branch-and-bound elimination `treewidth` for `|V| <= 24` with preallocated min-fill fallback), traversal and combinatorial optimization solvers (`dijkstra`, `bfs`, `dfs`, zero-allocation Edmonds-Karp `max_flow_min_cut`, and Kruskal `minimum_spanning_tree`), and deterministic `.dot`, `.graphml`, and `.csv` graph interchange codecs.
- **`statistics`**: Verified univariate continuous and discrete probability distributions (`distributions`: Normal, Beta, Gamma, Student-$t$, Chi-Squared, Exponential, Uniform, Log-Normal, Weibull, Bernoulli, Binomial, Poisson, Geometric, Negative Binomial) with safeguarded quantiles and MLE/MoM fitters; conjugate Bayesian inference and zero-allocation Metropolis-Hastings MCMC (`bayesian`); Cholesky-validated multivariate Gaussian copulas with Genz Gauss-Legendre quadrature, Richtmyer Quasi-Monte Carlo, and Same Game Parlay pricing (`copula`); Expectation-Maximization Zero-Inflated Poisson covariate regression (`zip_regression`); topological data analysis (`tda`); stochastic processes (Markov chains, Ornstein-Uhlenbeck); rating systems (Glicko-2, venue-adjusted sports Elo); calibrated 2D match score grids (Maher Poisson, Karlis-Ntzoufras bivariate Poisson, Dixon-Coles MLE); Kelly criterion, edge validation, and risk-of-ruin models; and zero-allocation bookmaker market pricing (`market_pricing` for Multiplicative, Additive, Power, and Shin devigging, `+EV`, arbitrage, Dutching, middles, and hedging).
- **`special_functions`**: Classical mathematical functions (Bessel functions, Legendre polynomials).

---

## 2. Developer Instructions

### Prime Number Sieve and Finite Field Arithmetic

Generate prime sequences and compute polynomial arithmetic over finite fields:

```rust
# use pure_math::__doctest_facade as math_explorer;
use math_explorer::pure_math::algebra::fields::Fp;
use math_explorer::pure_math::algebra::polynomial::Polynomial;
use math_explorer::pure_math::number_theory::primes::primes_up_to;

// Generate prime numbers up to a specified limit
let primes = primes_up_to(20);
assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);

// Perform arithmetic in the finite field F_7
let a = Fp::<7>::new(3);
let b = Fp::<7>::new(5);
assert_eq!(a + b, Fp::<7>::new(1)); // (3 + 5) % 7 = 1
assert_eq!(a * b, Fp::<7>::new(1)); // (3 * 5) % 7 = 15 % 7 = 1

// Construct and sum polynomials over F_7
let p1 = Polynomial::new(vec![b, Fp::<7>::new(0), a]);
let q1 = Polynomial::new(vec![Fp::<7>::new(1), Fp::<7>::new(2)]);
let sum = p1 + q1;

assert_eq!(sum.coeffs[0], Fp::<7>::new(6));
assert_eq!(sum.coeffs[1], Fp::<7>::new(2));
assert_eq!(sum.coeffs[2], Fp::<7>::new(3));
```

### Parametric Distributions and Conjugate Bayesian Updating

Evaluate exact quantiles, fit parameters, and compute conjugate Highest Posterior Density (HPD) intervals:

```rust
# use pure_math::__doctest_facade as math_explorer;
use math_explorer::pure_math::statistics::bayesian::BetaBinomial;
use math_explorer::pure_math::statistics::distributions::{
    ContinuousDistribution, GammaDist,
};

# fn main() -> math_explorer::error::Result<()> {
let gamma = GammaDist::new(3.0, 1.5)?;
let median = gamma.quantile(0.5)?;
assert!((gamma.cdf(median) - 0.5).abs() < 1e-10);

let posterior = BetaBinomial::uniform().update(16, 20)?;
let hpd95 = posterior.hpd_interval(0.95)?;
assert!(hpd95.contains(posterior.posterior_mean()));
# Ok(())
# }
```

---

## 3. Ecosystem Integration

The abstract structures in `pure_math` supply algorithms to domain engines (e.g., differential equation integrators in `domain_physics` and statistics in `domain_applied`) and interactive educational visualizers in `math_explorer_gui`.
