# `math_scalars` — Context-Capable `#![no_std]` Algebraic & Floating-Point Scalar Foundation

## 1. Overview

`math_scalars` is OxidizeMath's Tier 0 `#![no_std]` algebraic and numerical scalar foundation crate. It unifies two complementary mathematical models without `std` thread-local or global state:

- **Zero-Overhead Stateless Scalar Hierarchy (`Scalar` -> `Ring` -> `Field` -> `ComplexField` -> `RealField`)**:
  - Supports `f32`, `f64`, [`Complex<T>`](num_complex::Complex), forward-mode automatic differentiation [`Dual<T>`], signed integers, compile-time prime fields, and (under `alloc`) `dashu::integer::IBig` and `dashu::rational::RBig`.
  - Delivers deterministic `0`-ULP IEEE-754 square-root (`libm::sqrt` / `libm::sqrtf`) and overflow-safe hypotenuse (`libm::hypot` / `libm::hypotf`) across host and `wasm32-unknown-unknown` targets.
- **Explicit Runtime-Configured Algebraic Domains (`DomainRing` & `DomainField`)**:
  - [`StatelessContext<T>`]: Zero-sized (`size_of == 0`) inline adapter bridging any stateless `Ring` / `Field` into context-polymorphic algorithms with zero runtime dispatch overhead.
  - [`ModPrimeContext`]: Bare `#![no_std]` runtime prime field $\mathrm{GF}(p)$ supporting residue arithmetic, overflow-free `u128`-widened multiplication, extended-Euclidean inversion, and Horner polynomial evaluation.
  - [`DashuFloatContext`] & [`DashuModContext`] (`alloc` feature): Dynamic-precision `dashu::float::FBig` and arbitrary-modulus `dashu::integer::IBig` contexts parameterized explicitly without thread-local state.

```rust
use math_scalars::{DomainField, DomainRing, Dual, ModPrimeContext, RealField, StatelessContext};

let x = 3.9999999999999996_f64;
assert_eq!(<f64 as RealField>::sqrt(x).to_bits(), libm::sqrt(x).to_bits());

let d = Dual::variable(2.0_f64);
let y = (d * d + Dual::constant(1.0)).sqrt();
assert!((y.eps - 2.0 / 5.0_f64.sqrt()).abs() < 1e-15);

let gf = ModPrimeContext::try_new(1_000_000_007).unwrap();
let inv3 = gf.try_inv(&3).unwrap();
assert_eq!(gf.mul(&3, &inv3), 1);

let real_ctx = StatelessContext::<f64>::new();
assert_eq!(core::mem::size_of_val(&real_ctx), 0);
assert_eq!(real_ctx.eval_horner(&[1.0, 2.0, 3.0], &2.0), 17.0);
```

## 2. Developer Instructions

Verify `math_scalars` across bare `#![no_std]`, `alloc`, `std`, and WebAssembly targets:

```bash
cargo check -p math_scalars --no-default-features
cargo check -p math_scalars --no-default-features --features alloc,serde
cargo check -p math_scalars --target wasm32-unknown-unknown --no-default-features --features alloc,serde
cargo test -p math_scalars --no-default-features
cargo test -p math_scalars --all-features
```
