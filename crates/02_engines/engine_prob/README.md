# `engine_prob`: `#![no_std]` Probability, Bayesian, Calibration & Optimal Transport Engine

## 1. Overview & Architecture

`engine_prob` is OxidizeMath's Tier 2 `#![no_std]` probabilistic and statistical computing engine (`crates/02_engines/engine_prob`), designed for deterministic execution across bare-metal targets, native hosts, and `wasm32-unknown-unknown` with optional `alloc` support.

Core subsystems:
- **`special`**: Lanczos `ln_gamma` and `gamma`, `beta` and `ln_beta`, regularized lower/upper incomplete gamma ($P(a, x)$, $Q(a, x)$), Lentz continued-fraction regularized incomplete beta ($I_x(a, b)$), error functions (`erf`, `erfc`, `erf_inv`, `erfc_inv`), and Halley-refined standard normal CDF/quantile primitives.
- **`distributions`**: Continuous (`Normal`, `Uniform`, `Exponential`, `Gamma`, `Beta`, `StudentsT`, `ChiSquared`) and discrete (`Bernoulli`, `Binomial`, `Poisson`, `Categorical`) distributions with densities, log-densities, CDFs, quantiles, moments, zero-allocation Bayes normalization, and deterministic `rand_core::RngCore` sampling.
- **`conjugate`**: Zero-allocation closed-form conjugate exponential families (`BetaBinomial`, `DirichletMultinomial`, `NormalNormal`, `NormalInverseGamma`, `GammaPoisson`) with exact credible intervals, posterior predictive distributions, marginal likelihoods, and Bayes factors.
- **`calibration`**: Finite-sample split conformal quantiles ($\lceil (n + 1)(1 - \alpha) \rceil / n$), prediction-set masks, conformal p-values, and anytime-valid `EValue` calibrators.
- **`transport`**: Zero-allocation 1D Wasserstein ($W_1$ / Earth Mover's Distance) and log-domain stabilized Sinkhorn-Knopp optimal transport over caller-supplied scratch slices.

```rust
# use engine_prob::__doctest_facade as math_explorer;
use math_explorer::engine_prob::{BetaBinomial, ContinuousDist, Normal};

let std_norm = Normal::standard();
let q975 = std_norm.quantile(0.975).unwrap();
assert!((q975 - 1.959963984540054).abs() < 1e-12);

let model = BetaBinomial::new(2.0, 2.0).unwrap().update(8, 2);
let (lo, hi) = model.credible_interval(0.95).unwrap();
assert!(lo < model.posterior_mean() && model.posterior_mean() < hi);
```

## 2. Developer Instructions

Verify `#![no_std]` compilation (with and without `alloc`), unit tests, zero-allocation invariants, and `wasm32-unknown-unknown` compatibility:

```bash
cargo check -p engine_prob --no-default-features
cargo check -p engine_prob --no-default-features --features alloc
cargo check -p engine_prob --target wasm32-unknown-unknown
cargo test -p engine_prob
```
