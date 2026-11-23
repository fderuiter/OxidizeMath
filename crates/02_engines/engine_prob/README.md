# `engine_prob`: Tier 2 `#![no_std]` Stochastic, Cubature & Bootstrap Engine

## 1. Overview & Architecture

`engine_prob` is the Tier 2 `#![no_std]` numerical probability and Monte Carlo engine for the `OxidizeMath` workspace. Every estimator executes over caller-supplied borrowed [`SolverWorkspace`] slices (`&mut [f64]` and `&mut [usize]`) and deterministic [`oxidize_core::rng::OxidizeRng`] streams with zero OS entropy and zero heap allocations inside hot loops:

- **`workspace`**: Borrowed [`SolverWorkspace`] slice container (`&mut [f64]` and `&mut [usize]`) with fallible capacity partitioning, plus optional `alloc`-gated `OwnedSolverWorkspace` convenience buffers.
- **`cubature`**: Multidimensional Quasi-Monte Carlo integration over $[0, 1]^d$ and $\prod_{j=1}^d [a_j, b_j]$ using Richtmyer (with Baker's tent transform), Halton, and Joe-Kuo Sobol sequences with Cranley-Patterson shift error bounds, alongside Genz separation-of-variables Gaussian copula / MVN cubature.
- **`variance_reduction`**: Paired antithetic variates (Gaussian, uniform, and correlated Cholesky Bernoulli indicators), optimal linear control variates ($\hat{\beta}^* = \widehat{\mathrm{Cov}}(Y, C) / \widehat{\mathrm{Var}}(C)$), and self-normalized importance sampling with Kish effective sample size (ESS).
- **`bootstrap`**: Zero-allocation nonparametric percentile and Efron BCa (bias-corrected and accelerated) bootstrap intervals with leave-one-out jackknife acceleration $\hat{a}$ over arbitrary indexable samples and scalar slices.

## 2. Developer Instructions

Verify `#![no_std]` compilation, optional `alloc` helpers, WebAssembly targets, and unit tests from the workspace root:

```bash
cargo check -p engine_prob --no-default-features
cargo check -p engine_prob --no-default-features --target wasm32-unknown-unknown
cargo test -p engine_prob --no-default-features
cargo test -p engine_prob --features alloc
```

### Quick Start

```rust
use engine_prob::bootstrap::bca_bootstrap_mean;
use engine_prob::cubature::{CubatureConfig, CubatureRule, integrate_unit_cube};
use engine_prob::workspace::SolverWorkspace;
use oxidize_core::rng::{BASE_SEED, OxidizeRng};

let mut floats = [0.0_f64; 512];
let mut indices = [0_usize; 32];
let mut ws = SolverWorkspace::new(&mut floats, &mut indices);

// 1. 3D Sobol cubature with Cranley-Patterson shift error bounds:
let cfg = CubatureConfig::new(CubatureRule::Sobol, 256);
let est = integrate_unit_cube(3, &cfg, &mut ws, |u| u[0] * u[1] + u[2]).unwrap();
assert!((est.value - 0.75).abs() < 1e-2);
assert!(est.std_error >= 0.0);

// 2. Zero-allocation BCa bootstrap with jackknife acceleration:
let sample = [1.2, 1.5, 1.7, 2.1, 2.8, 3.9, 5.4, 8.2];
let mut rng = OxidizeRng::from_seed(BASE_SEED);
let ci = bca_bootstrap_mean(&sample, 200, 0.95, &mut rng, &mut ws).unwrap();
assert!(ci.lower < ci.point_estimate && ci.point_estimate < ci.upper);
```
