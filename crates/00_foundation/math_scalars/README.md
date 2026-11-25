# `math_scalars`

![Maturity: Validated](https://img.shields.io/badge/maturity-Validated-1f6feb.svg)

> **Maturity Tier:** `Validated` — Tier 0 `#![no_std]` scalar, fixed/arbitrary-precision numeric, Cayley-Dickson hypercomplex, 64-bit Montgomery prime field, Galois extension, and stack $p$-adic foundation crate verified with algebraic law suites, `wasm32` compilation, and `VerifiedAllocator` zero-allocation proofs.

Zero-allocation Tier 0 `#![no_std]` scalar and algebraic substrate (`crates/00_foundation/math_scalars`) centralizing unordered field contracts, ordered real scalar contracts, fixed-width 256-bit integers, optional `alloc`-backed arbitrary-precision numbers, Cayley-Dickson division/composition algebras, carry-safe 64-bit Montgomery prime fields, stack polynomial rings, Galois extension fields $\mathbb{F}_{p^n}$, and fixed-precision $p$-adic numbers $\mathbb{Q}_p$ with Newton-Hensel root lifting.

## 1. Overview & Architecture

`math_scalars` sits at Tier 0 (`crates/00_foundation/math_scalars`) depending only on sibling Tier 0 [`math_diagnostics`](../math_diagnostics/README.md) (`#![no_std]` `DiagnosticVisitor` contract) and `libm`, providing deterministic behavior across native (`x86_64`, `aarch64`) and WebAssembly (`wasm32-unknown-unknown`) targets.

Key components:
- **Scalar Contracts (`traits`)**:
  - `FieldScalar`: Unordered field arithmetic, conjugation, squared/Euclidean norms, and fallible inversion (`f32`, `f64`, `Complex<T>`).
  - `RealScalar`: Ordered `PartialOrd` IEEE-754 / `libm` transcendental contract (`f32`, `f64`).
- **Fixed & Arbitrary Precision (`fixed256`, `fixed256_signed`, `bigint`)**:
  - `U256` and `I256`: Stack-allocated 256-bit unsigned and two's-complement signed integers with checked and Euclidean arithmetic (`--no-default-features`).
  - `BigInt`, `BigRational`, and `BigFloat` (`--features alloc`): `std`-free `dashu`-backed exact integers, exact rationals with non-narrowing `numerator()` / `denominator()` accessors, and arbitrary-precision binary floats preserving requested bit precision while rejecting `NaN` and `Infinity`.
- **Cayley-Dickson Algebras (`complex`, `quaternion`, `octonion`)**:
  - `#[repr(C)]` `Complex<T>`, `Quaternion<T>`, `UnitQuaternion<T>`, and `Octonion<T>` with flat coordinate memory layouts, norm multiplicativity, Moufang identities, and fallible `try_inverse`.
- **Finite Fields, Polynomials & $p$-adic Rings (`modular`, `fp`, `poly`, `fp_ext`, `padic`)**:
  - `Fp<const P: u64>`: Compile-time Miller-Rabin validated prime fields up to $18{,}446{,}744{,}073{,}709{,}551{,}557 = 2^{64} - 59$ via carry-safe 128-bit Montgomery reduction plus an explicit $P = 2$ path.
  - `StackPoly<T, const CAP: usize>` and `FpExt<const P: u64, const DEG: usize, M>`: Zero-allocation Euclidean polynomial division, extended GCD, and Galois quotient fields $\mathbb{F}_{p^n} \cong \mathbb{F}_p\[x\]/(m(x))$.
  - `Qp<const P: u64, const N: usize>`: Stack-allocated $p$-adic numbers with ultrametric valuations and zero-allocation Newton-Hensel simple root lifting.

| Feature | `no_std` | Heap (`alloc`) | Included Capabilities |
| :--- | :---: | :---: | :--- |
| `--no-default-features` | Yes | No | `FieldScalar`, `RealScalar`, `U256`, `I256`, `Complex`, `Quaternion`, `UnitQuaternion`, `Octonion`, `Fp`, `StackPoly`, `FpExt`, `Qp` |
| `--features serde` | Yes | No | Adds `#![no_std]` `serde` `Serialize` and invariant-validating `Deserialize` across all enabled types |
| `--features alloc` | Yes | Yes | Adds `BigInt`, `BigRational`, `BigFloat` (`dashu` without `std`) |
| `--features std` | Optional | Yes | Enables `dashu/std` and `math_diagnostics/std` integration for hosted callers |

## 2. Developer Instructions

Verify strict `#![no_std]` compilation on host and WebAssembly alongside the full feature test suite:

```bash
cargo check -p math_scalars --no-default-features
cargo check -p math_scalars --target wasm32-unknown-unknown --no-default-features
cargo test -p math_scalars --all-features
```

### Usage Example

```rust
use math_scalars::{Complex, Fp, Qp, U256, hensel_lift_root};

let z = Complex::new(3.0_f64, 4.0_f64);
assert!((z.norm() - 5.0).abs() < 1e-12);

// Largest 64-bit prime P = 2^64 - 59 without overflow
const P_MAX: u64 = 18_446_744_073_709_551_557;
let a = Fp::<P_MAX>::from_u64(P_MAX - 2);
let inv = a.try_inverse().unwrap();
assert_eq!(a * inv, Fp::<P_MAX>::ONE);

// Lift x^2 - 2 ≡ 0 (mod 7) from r_0 = 3 to Q_7 with 6 digits of 7-adic precision
let root: Qp<7, 6> = hensel_lift_root::<7, 6, 4>(&[-2, 0, 1], 3).unwrap();
assert_eq!((root * root).valuation(), Qp::<7, 6>::from_i64(2).valuation());

let wide = U256::from_u128(u128::MAX).checked_mul(U256::from_u64(2)).unwrap();
assert!(!wide.is_zero());
```
