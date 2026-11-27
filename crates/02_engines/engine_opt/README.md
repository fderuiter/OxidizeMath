# `engine_opt`

![Maturity: Certified](https://img.shields.io/badge/maturity-Certified-2ea043.svg)

> **Maturity Tier:** `Certified` — `#![no_std]` zero-allocation curve-fitting, single-shooting `iLQR` / projected `MPC`, Howard `DP`, Mann `MFG`, and Nash equilibrium engine with `VerifiedAllocator` zero-heap-allocation proofs.

## 1. Overview & Architecture

`engine_opt` (`crates/02_engines/engine_opt`) is the `#![no_std]` Tier 2 optimization, optimal control, dynamic programming, and game-theoretic equilibrium engine for the OxidizeMath workspace. Every iterative solver operates over caller-supplied `math_scalars::workspace::SolverWorkspace` slices (with deterministic `*_workspace_len` sizing helpers) and optional `alloc` support while preserving downward Tier 2 dependencies (`engine_linalg`, `engine_pde`, `math_scalars`, `math_diagnostics`).

### Key Capabilities

- **Curve Fitting (`fit`)**: Streaming Givens QR polynomial regression (`1..=4`) and damped Levenberg-Marquardt nonlinear least squares (`1..=6` parameters) with closed-form `libm` parameter seeds.
- **Single-Shooting `iLQR` & Projected `MPC` (`ilqr`, `mpc`)**: Box-projected iterative Linear Quadratic Regulator and receding-horizon Model Predictive Control using `engine_linalg::cholesky_solve` Riccati sweeps, tightened state barriers, and Armijo backtracking.
- **Dynamic Programming (`dp`)**: Finite-horizon backward induction and infinite-horizon Howard policy iteration backed by `engine_linalg::lu_solve` exact policy evaluation.
- **Mean-Field Games (`mfg`)**: 1D Achdou-Capuzzo-Dolcetta coupled Hamilton-Jacobi-Bellman / Fokker-Planck solver with monotone Engquist-Osher Hamiltonians, adaptive CFL substeps down to `viscosity = 0.0`, quadrature normalization, and Mann damping.
- **Nash Equilibria (`nash`)**: Lexicographic Lemke-Howson complementary pivoting for bimatrix games and `CFR+` regret matching for $N$-player polymatrix games.
- **Typed Diagnostics (`error`)**: `#![no_std]` `Copy` `FitError` and `OptError` implementing `math_diagnostics::Diagnostic` with causal `LinalgError` retention.

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

Verify `#![no_std]` WebAssembly compilation and run the unit, `alloc`, `serde`, and `VerifiedAllocator` zero-allocation test suites from the workspace root:

```bash
cargo check -p engine_opt --target wasm32-unknown-unknown --no-default-features
cargo test -p engine_opt --all-features
```
