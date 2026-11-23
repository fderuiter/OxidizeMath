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
- **`analysis`**: Differential equation solvers (`OdeSystem`, explicit Runge-Kutta and zero-allocation symplectic `StormerVerlet`, `Yoshida4`, and `GaussLegendre4` steppers), zero-allocation bracketed and derivative root-finding (`roots::Bisection`, `roots::NewtonRaphson`, `roots::Brent`), 1D and zero-allocation 2D partial differential equation solvers (`pde::wave_2d::Wave2DSolver` for hyperbolic wave tanks, Kirchhoff-Love biharmonic Chladni plates, Chladni sand drift, Marching-Squares nodal contours, and pre-planned 2D FFT spectra), numerical integration, and calculus algorithms.
- **`differential_geometry`**: Manifolds, metric tensors, and Riemannian curvature.
- **`elliptic_curves`**: Zero-allocation short-Weierstrass curve group arithmetic ($P + Q$, $\[k\]P$, cyclic orbits) over $\mathbb{R}$ and prime fields $\mathbb{F}_p$, Frobenius traces $a_p$, Hasse-Weil verification, Breuer modular polynomial valuation bounds, and bridges to binary quadratic forms (`BQF`), Hurwitz-Kronecker class numbers, and modular $q$-series (`QSeries<i64>`).
- **`graph_theory`**: Network graph structures, centrality metrics, and traversal algorithms.
- **`statistics`**: `#![no_std]`-backed probability and graphical-model inference delegated to Tier 2 (`engine_prob` for Lanczos `ln_gamma`, regularized incomplete beta/gamma, `erf`/`erfc`/`erf_inv`, parametric continuous/discrete quantiles & sampling, and conjugate Bayesian posteriors; `engine_graph` for zero-allocation chain forward-backward and factor-graph belief propagation), Zero-Inflated Poisson count modeling (`zip_regression::ZipDistribution` with `ln_pmf`, `quantile`, deterministic `sample`, and `ZipRegression::fit` EM + IRLS + damped Newton solver without $\rho$ clamping), stochastic processes (Markov chains, Ornstein-Uhlenbeck), rating systems (Glicko-2, venue-adjusted sports Elo), calibrated 2D match score grids (Maher Poisson, Karlis-Ntzoufras bivariate Poisson, Dixon-Coles MLE), Kelly criterion decision models, and zero-allocation bookmaker market pricing (`market_pricing` for Multiplicative, Additive, Power, and Shin devigging, `+EV`, arbitrage, Dutching, middles, and hedging).
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

### Zero-Inflated Poisson Quantiles and Conjugate Inference

Evaluate exact Zero-Inflated Poisson log-probabilities and quantiles alongside `#![no_std]` conjugate Beta-Binomial updates:

```rust
# use pure_math::__doctest_facade as math_explorer;
use math_explorer::engine_prob::BetaBinomial;
use math_explorer::pure_math::statistics::zip_regression::{Count, ZipDistribution};
use math_explorer::rng::OxidizeRng;

let zip = ZipDistribution::from_values(0.35, 3.2).expect("valid ZIP parameters");
let p0 = zip.ln_pmf(Count::new(0)).exp();
assert!((p0 - zip.pmf(Count::new(0))).abs() < 1.0e-12);
assert_eq!(zip.quantile(0.20).expect("valid quantile"), Count::new(0));

let mut rng = OxidizeRng::new(7);
let draw = zip.sample(&mut rng);
assert!(zip.cdf(draw) > 0.0);

let post = BetaBinomial::uniform_prior().update(18, 6);
let (lo, hi) = post.credible_interval(0.95).expect("credible interval");
assert!(lo < post.posterior_mean() && post.posterior_mean() < hi);
```

---

## 3. Ecosystem Integration

The abstract structures in `pure_math` build atop `#![no_std]` Tier-2 kernels (`engine_prob` and `engine_graph`) and supply algorithms to domain engines (e.g., differential equation integrators in `domain_physics` and statistical inference in `domain_applied`) and interactive educational visualizers in `math_explorer_gui`.
