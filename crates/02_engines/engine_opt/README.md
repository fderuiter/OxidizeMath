# `engine_opt`

![Maturity: Certified](https://img.shields.io/badge/maturity-Certified-2ea043.svg)

> **Maturity Tier:** `Certified` — `#![no_std]` stack-allocated polynomial and damped Levenberg-Marquardt curve-fitting engine with `VerifiedAllocator` zero-heap-allocation proofs.

## 1. Overview & Architecture

`engine_opt` (`crates/02_engines/engine_opt`) is the `#![no_std]` Tier 2 optimization and curve-fitting engine for the OxidizeMath workspace. It executes centered polynomial ordinary least squares, damped Levenberg-Marquardt nonlinear regression, and closed-form parameter seeding on fixed stack buffers over `math_scalars` and `engine_linalg` without upward dependencies from `math_commons`.

### Key Capabilities

- **Centered Polynomial OLS (`1..=4`)**: Streaming Givens QR regression via `engine_linalg::StreamingQr` over affine $[-1, 1]$ scaled coordinates with optional $\log_{10}$ axis transforms.
- **Damped Levenberg-Marquardt (`1..=6` Parameters)**: Analytic and central finite-difference Jacobian solvers backed by `engine_linalg::solve_damped_normal` on fixed stack buffers with zero heap allocations.
- **Closed-Form Seeds**: Stack-allocated `WeightedOlsAccum` and deterministic `libm`-backed parameter seed estimators (`Exponential`, `PowerLaw`, `Logarithmic`, `Logistic`, and `Sinusoidal`).
- **Typed Diagnostics**: `#![no_std]` `Copy` [`FitError`](crate::error::FitError) preserving causal [`LinalgError`](crate::error::LinalgError) sources.

### Example

```rust
use engine_opt::fit::{FitModel, fit_model};

let pts = [
    [0.0, 2.0],
    [1.0, 2.0 * (-0.5_f64).exp()],
    [2.0, 2.0 * (-1.0_f64).exp()],
];
let curve = fit_model(&pts, FitModel::Exponential, false).unwrap();
assert!((curve.coefficients()[0] - 2.0).abs() < 1e-6);
```

## 2. Developer Instructions

Verify `#![no_std]` WebAssembly compilation and run the unit, `serde`, and `VerifiedAllocator` zero-allocation test suite from the workspace root:

```bash
cargo check -p engine_opt --target wasm32-unknown-unknown --no-default-features
cargo test -p engine_opt --all-features
```
