# `math_scalars`

![Maturity: Validated](https://img.shields.io/badge/maturity-Validated-2ea043.svg)

> **Maturity Tier:** `Validated` — `#![no_std]` Tier 0 forward-mode automatic differentiation and IEEE-754 outward-rounded interval/ball foundation crate qualified in `traceability.toml`.

## 1. Overview & Architecture

`math_scalars` is the Tier 0 `#![no_std]` (`alloc`-optional and `serde`-optional) scalar foundation crate for OxidizeMath (`crates/00_foundation/math_scalars`), providing:

- **`Scalar` and `RealScalar` trait hierarchy**: Separates general algebraic + deterministic `libm` elementary operations (`Scalar`, implemented for `f64`, `f32`, `Dual<T>`, `DualVec<T, N>`, `HyperDual<T>`, `Interval<T>`, and `Ball<T>`) from ordered IEEE-754 floating-point operations (`RealScalar`, implemented for `f64` and `f32`).
- **Zero-allocation forward-mode automatic differentiation**:
  - `Dual<T>`: First-order scalar tangent propagation.
  - `DualVec<T, const N: usize>`: Stack-allocated multi-tangent forward-mode gradients and Jacobians.
  - `HyperDual<T>`: Exact second-order directional and mixed partial derivatives free of subtractive cancellation.
- **Portable IEEE-754 outward-rounded enclosures**:
  - `Interval<T>` (`\[lo, hi\]`) and midpoint-radius `Ball<T>` (`mid`, `rad`) using pure IEEE-754 bit-level `next_up` / `next_down` ULP bounds and conservative `libm` transcendental padding without hardware FPU rounding-mode state mutation (`fesetround` is never invoked), guaranteeing bit-deterministic parity on `x86_64` and `wasm32-unknown-unknown`.

### Quick Example

```rust
use math_scalars::{Ball, Dual, HyperDual, Interval, Scalar};

fn energy<S: Scalar>(x: S) -> S {
    x.sin() + x.sqr()
}

let d1 = energy(Dual::variable(0.5_f64));
assert!((d1.eps - (0.5_f64.cos() + 1.0)).abs() < 1e-15);

let d2 = energy(HyperDual::second_order(0.5_f64));
assert!((d2.eps12 - (-0.5_f64.sin() + 2.0)).abs() < 1e-15);

let encl = energy(Interval::point(0.5_f64));
assert!(encl.contains(0.5_f64.sin() + 0.25));

let ball = Ball::from_interval(encl);
assert!(ball.contains(0.5_f64.sin() + 0.25));
```

## 2. Developer Instructions

Verify `#![no_std]` native and WebAssembly compilation (`--no-default-features` and `--features alloc`) and execute the unit and integration test suites from the workspace root:

```bash
cargo check -p math_scalars --no-default-features
cargo check -p math_scalars --features alloc
cargo check -p math_scalars --target wasm32-unknown-unknown --no-default-features
cargo check -p math_scalars --target wasm32-unknown-unknown --features alloc
cargo test -p math_scalars --all-features
```
