# `math_scalars`

![Maturity: Validated](https://img.shields.io/badge/maturity-Validated-3fb950.svg)

> **Maturity Tier:** `Validated` — `#![no_std]` Tier 0 foundational scalar hierarchy, `0`-ULP `libm`-backed IEEE-754 kernels, and caller-owned scratch workspaces.

## 1. Overview & Architecture

`math_scalars` (`crates/00_foundation/math_scalars`) is the `#![no_std]` Tier 0 scalar foundation for the OxidizeMath workspace with zero upward workspace dependencies. It unifies borrowed-reference algebraic traits, `0`-ULP IEEE-754 floating-point kernels, forward-mode dual numbers, zero-copy 2D coordinate series views, and caller-owned scratch workspaces across all engines and domains.

### Key Capabilities

- **Unified Scalar Hierarchy**: [`RefOps`], [`Scalar`], [`Ring`], [`Field`], [`ComplexField`], and [`RealField`] support borrowed reference arithmetic (`ref_add`, `ref_sub`, `ref_mul`, `ref_neg`, `ref_div`) without requiring `Copy` or `PartialOrd` below [`RealField`].
- **`0`-ULP IEEE-754 Kernels**: Deterministic `libm`-backed [`sqrt_f64`], [`sqrt_f32`], Dekker-Borges 106-bit [`hypot_f64`], widened [`hypot_f32`], and Dekker `diff_of_products` across native hosts and `wasm32-unknown-unknown`.
- **Complex & Dual Numbers**: Full hierarchy implementations for `num_complex::Complex<T>` and exact first-order forward-mode [`Dual<T>`].
- **Workspace & Series Primitives**: Shared [`ScalarWorkspace`], [`SolverWorkspace`], [`ByteArena`], [`StepSize`], [`GridIndex`], [`Dimension`], [`Sample2D`], [`PointSeries`], and [`ScalarError`] without conversion adapters.

### Example

```rust
use math_scalars::{ComplexField, Dual, ScalarWorkspace, sqrt_f64};

assert_eq!(sqrt_f64(3.999_999_999_999_999_6), 1.999_999_999_999_999_8);

let x = Dual::variable(4.0_f64);
let y = (x * x + Dual::constant(9.0_f64)).sqrt();
assert!((y.re - 5.0).abs() < 1e-15);
assert!((y.eps - 0.8).abs() < 1e-15);

let mut buf = [0.0_f32; 8];
let mut ws = ScalarWorkspace::new(&mut buf);
let slice = ws.try_take_slice(4).unwrap();
assert_eq!(slice.len(), 4);
```

## 2. Developer Instructions

Verify `#![no_std]` WebAssembly compilation and run the unit, `0`-ULP IEEE-754 parity, and zero-allocation test suite from the workspace root:

```bash
cargo check -p math_scalars --target wasm32-unknown-unknown --no-default-features
cargo test -p math_scalars --all-features
```
