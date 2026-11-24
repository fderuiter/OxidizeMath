# `math_scalars`

Canonical Tier 0 `#![no_std]` mathematical scalar and algebraic substrate for the OxidizeMath workspace.

## 1. Overview

`math_scalars` provides zero-allocation stack scalars alongside optional `alloc`-gated arbitrary-precision numeric types so numerical engines, domain crates, and WebAssembly targets share one mathematically rigorous type foundation:

- **Scalar Contracts (`traits`)**: Splits unordered [`FieldScalar`] (`f32`, `f64`, [`Complex<f32>`], [`Complex<f64>`]) from ordered [`RealScalar`] (`f32`, `f64`).
- **Fixed 256-Bit Stack Integers (`uint256`, `int256`)**: [`U256`] and two's-complement [`I256`] with checked and Euclidean division and zero heap allocations when `alloc` is disabled.
- **Arbitrary-Precision Types (`big`, `feature = "alloc"`)**: `dashu`-backed [`BigInt`], [`BigRational`] (preserving exact integer numerators and denominators), and [`BigFloat`] (`try_with_prec` retaining exact caller bit precision while rejecting `NaN` and `Infinity`).
- **Cayley-Dickson Algebras (`complex`, `quaternion`, `unit_quaternion`, `octonion`)**: `#[repr(C)]` [`Complex<T>`], [`Quaternion<T>`], [`UnitQuaternion<T>`], and [`Octonion<T>`] with conjugation, Euclidean norms, and fallible inverses.
- **Montgomery Prime Fields & Galois Extensions (`fp`, `poly`, `galois`)**: Compile-time primality-validated [`Fp<const P: u64>`] using carry-safe 128-bit Montgomery reduction up to $P = 18{,}446{,}744{,}073{,}709{,}551{,}557$ (plus an explicit $P = 2$ path), stack polynomials [`StackPoly`], and Galois quotient fields [`FpExt`] ($\mathbb{F}_{p^n}$).
- **Truncated $p$-Adic Rings (`padic`)**: Stack-allocated [`Qp<const P: u64, const N: usize>`] supporting ultrametric valuations, unit inversion, and zero-allocation Newton-Hensel polynomial root lifting.

```rust
use math_scalars::{Complex, FieldScalar, Fp, Qp, U256};

let z = Complex::new(3.0_f64, 4.0_f64);
assert!((z.abs() - 5.0).abs() < 1e-12);
assert!((z * z.try_inv().unwrap() - Complex::ONE).abs() < 1e-12);

// Large 64-bit prime above 2^63 using carry-safe Montgomery multiplication
const P_MAX: u64 = 18_446_744_073_709_551_557;
let a = Fp::<P_MAX>::new(P_MAX - 2);
let b = Fp::<P_MAX>::new(P_MAX - 3);
assert_eq!((a * b).value(), 6);

// Hensel-lift sqrt(2) in Q_7 to 7^6 precision with zero heap allocations
let root = Qp::<7, 6>::hensel_lift_root(&[-2, 0, 1], 3).unwrap();
assert_eq!((root * root - Qp::<7, 6>::from_i64(2)).valuation(), None);

let wide = U256::from_u128(u128::MAX).checked_mul(U256::from_u64(2)).unwrap();
assert!(wide > U256::from_u128(u128::MAX));
```

## 2. Developer Instructions

Build and test `math_scalars` across pure `#![no_std]`, `alloc`, release overflow-checked, and WebAssembly targets:

```bash
cargo test -p math_scalars --no-default-features
cargo test -p math_scalars --features alloc
cargo test -p math_scalars --all-features --release
cargo check -p math_scalars --target wasm32-unknown-unknown --no-default-features
cargo check -p math_scalars --target wasm32-unknown-unknown --features alloc
```
