# `math_scalars`

![Maturity: Validated](https://img.shields.io/badge/Maturity-Validated-brightgreen)

## 1. Overview & Architecture

`math_scalars` is a Tier-0 `#![no_std]` foundation crate with optional `alloc` support and zero upward workspace dependencies. It unifies real floating-point numbers, stack-allocated forward-mode automatic differentiation jets, and outward-rounded IEEE-754 verified enclosures behind a deterministic `libm`-backed trait hierarchy:

- **`Scalar` & `RealScalar`**: `Scalar` captures algebraic field operations and deterministic `libm` elementary functions across reals, dual numbers, and set enclosures, while `RealScalar: Scalar + PartialOrd` isolates total-order comparisons and IEEE-754 bit-level ULP stepping (`next_up`, `next_down`) to `f32` and `f64`.
- **Forward Automatic Differentiation**:
  - `Dual<T>`: Zero-allocation first-order scalar forward AD $(f(x), f'(x))$.
  - `DualVec<T, const N: usize>`: Single-pass multi-tangent forward AD for exact gradients and Levenberg-Marquardt Jacobians.
  - `HyperDual<T>`: Exact second-order forward AD $(f, f_u, f_v, f_{uv})$ eliminating subtractive cancellation in Gaussian curvature, Laplace-Beltrami, Christoffel, and Hessian operators.
- **Portable Certified Enclosures**:
  - `Interval<T>`: Outward-rounded `[lo, hi]` interval arithmetic using portable IEEE-754 `next_down` / `next_up` bit-stepping and certified multi-ULP `libm` transcendental bounds without mutating FPU rounding state (`fesetround`).
  - `Ball<T>`: Outward-rounded midpoint-radius `(mid, rad)` ball arithmetic and Krawczyk / interval-Newton root certification across native (`x86_64`) and `wasm32-unknown-unknown` targets.

```rust
use math_scalars::{Ball, Dual, HyperDual, Interval, RealScalar, Scalar};

// Exact first derivative at f32 scale x = 64.0 without step underflow:
let x = Dual::variable(64.0_f32);
let y = (x * 0.25_f32).sin();
assert!((y.eps - 0.25_f32 * 16.0_f32.cos()).abs() <= 2.0 * f32::EPSILON);

// Exact mixed second partial derivative via HyperDual:
let u = HyperDual::bivariate_u(1.2_f64);
let v = HyperDual::bivariate_v(0.7_f64);
let f = (u * v).exp();
let expected_uv = (1.2_f64 * 0.7_f64).exp() * (1.0 + 1.2 * 0.7);
assert!((f.eps12 - expected_uv).abs() < 1e-14);

// Certified IntervalNewton / Krawczyk root enclosures within <= 4 ULPs:
let cert = math_scalars::certify_reference_roots();
assert!(cert.all_contained);
assert!(cert.max_interval_ulp_width <= 4);
assert!(cert.max_ball_ulp_width <= 4);
let _ = Ball::from_interval(Interval::point(1.0_f64.next_up()));
```

## 2. Developer Instructions

Verify `#![no_std]`, `alloc`, and `wasm32` compilation plus unit and zero-allocation tests:

```bash
cargo check -p math_scalars --no-default-features
cargo check -p math_scalars --features alloc
cargo check -p math_scalars --target wasm32-unknown-unknown --no-default-features
cargo test -p math_scalars --all-features
```
