# `engine_symbolic`

Tier-2 `#![no_std]` symbolic computer algebra (e-graph equality saturation) and forward/reverse automatic differentiation engine for `OxidizeMath`.

## 1. Overview & Architecture

`engine_symbolic` provides deterministic, bounded-memory differentiation and symbolic simplification primitives without relying on external CAS crates or the Rust standard library:

- **Forward-Mode `Dual` Numbers (`dual`)**: First-order forward-mode tangents over real (`Dual<f64>`) and complex (`Dual<Complex<f64>>`) scalars with exact Clarke subgradients on piecewise primitives (`abs`, `min`, `max`, `clamp`, `step`, `rect`).
- **Second-Order `Jet2` Scalars (`jet2`)**: Stack-allocated $N$-variate second-order Taylor jets (`Jet2<2>`, `Jet2<3>`) propagating value, gradient, and Hessian entries in a single pass with zero heap allocations.
- **Reverse-Mode `SolverWorkspace` (`workspace`)**: Fixed-capacity inline tape workspace enabling single-pass reverse-mode gradient accumulation under `VerifiedAllocator`.
- **Bounded E-Graph Equality Saturation (`sym`, `egraph`, `alloc` feature)**: Iterative union-find, canonical e-node hashconsing, algebraic/trigonometric rewrite saturation capped at `2048` nodes and `16` passes, and iterative minimum-cost AST extraction with zero recursion.

```rust
use engine_symbolic::{Dual, Jet2};

// Exact first derivative of clamp(2 * x, -1, 1) inside and at kinks
let x_in = Dual::variable(0.25_f64);
let y_in = (x_in * 2.0).clamp(Dual::constant(-1.0), Dual::constant(1.0));
assert!((y_in.re - 0.5).abs() < 1e-15);
assert!((y_in.eps - 2.0).abs() < 1e-15);

// Single-pass bivariate Hessian on shallow saddle z = 1e-4 * (u^2 - v^2)
let u = Jet2::<2>::var(0.0, 0);
let v = Jet2::<2>::var(0.0, 1);
let z = (u * u - v * v) * 1.0e-4;
assert!((z.duu() * z.dvv() - (-4.0e-8)).abs() < 1e-22);
```

## 2. Developer Instructions

- Keep the crate strictly `#![no_std]`; gate heap-backed e-graph structures behind `#[cfg(feature = "alloc")]`.
- Maintain zero heap allocations inside `Dual`, `Jet2`, and `SolverWorkspace` hot paths verified via `VerifiedEngine::run_verified`.
- Enforce iterative worklists across all e-graph congruence, saturation, and extraction passes (`MAX_EGRAPH_NODES = 2048`, `MAX_REWRITE_PASSES = 16`).

```bash
cargo test -p engine_symbolic
cargo check -p engine_symbolic --no-default-features
cargo test -p engine_symbolic --target wasm32-unknown-unknown --lib
```
