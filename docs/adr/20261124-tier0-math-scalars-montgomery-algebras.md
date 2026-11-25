# ADR 20261124: Tier 0 `math_scalars` Substrate, Carry-Safe 64-Bit Montgomery Fields, Cayley-Dickson Algebras, and Stack $p$-Adic Hensel Lifting

## Status
Accepted (2026-11-24; aligned with [ADR 20261124](20261124-no-std-math-diagnostics-and-causal-error-wiring.md), [ADR 20261124](20261124-optional-no-std-serde-substrate-and-facade.md), and [ADR 20261124](20261124-stack-core-errors-and-fallible-companion-apis.md))

## Context

Prior to this decision, fundamental scalar traits, arbitrary-precision wrappers, hypercomplex rotations, modular arithmetic, polynomial rings, and $p$-adic lifting across OxidizeMath suffered from five architectural bottlenecks:

1. **Conflated ordered vs. unordered scalar contracts:** `engine_linalg::Scalar` and `math_commons::Scalar` required `PartialOrd` alongside field arithmetic, blocking complex-valued linear solvers (`Complex<f64>`, `Complex<f32>`) without violating the mathematical fact that $\mathbb{C}$ is an unordered field.
2. **Duplicated and `std`-coupled scalar wrappers:** `dashu` pulled default `std` features in the workspace root; `pure_math::Float::with_val` ignored caller-requested bit precision while accepting `NaN` / `±Infinity`; `pure_math::Rational` forced algorithmic-information callers (`fractran`, `geometry`) to access raw `.0` fields; and 3D unit quaternions were duplicated across `domain_ai` Gaussian splatting and `oxidize_core` PLY exporters without a shared `#[repr(C)]` Cayley-Dickson substrate.
3. **Signed `i64` modulus ceiling and overflow in $\mathbb{F}_p$:** `Fp<const P: i64>` capped prime fields at `i64::MAX`, lacked compile-time primality enforcement, and duplicated `i128` modular helpers in `elliptic_curves::modular_fp`, while `FpWeierstrassCurve::new` overflowed `u64` trial division near $2^{64}$.
4. **Heap-bound polynomials and missing extension fields:** Univariate polynomial division required `Vec` allocations and lacked stack-backed Galois quotient fields $\mathbb{F}_{p^n} \cong \mathbb{F}_p[x]/(m(x))$.
5. **Missing stack $p$-adic numbers and Hensel lifting:** Number-theoretic AMBS valuations and elliptic-curve point lifting lacked a fixed-precision stack $\mathbb{Q}_p$ type with zero-allocation Newton-Hensel simple root lifting.

## Decision

1. **Create Tier 0 `#![no_std]` crate `crates/00_foundation/math_scalars`:**
   - Configure `[workspace.dependencies].dashu` with `default-features = false, features = ["num-order", "num-traits_v02"]` so `math_scalars` compiles with `--no-default-features` on host and `wasm32-unknown-unknown`, with optional `alloc`, `std`, and `#![no_std]` `serde` features (`serde_impl`) plus zero-allocation `math_diagnostics::Diagnostic` codes (`SCALAR-001`..`SCALAR-012`) on `ScalarError`.
   - Provide stack-allocated 256-bit unsigned (`U256`) and two's-complement signed (`I256`) integers with checked and Euclidean (`checked_div_euclid`, `checked_rem_euclid`) arithmetic under zero heap allocations.
   - Provide `alloc`-gated `BigInt`, `BigRational` (exposing exact `numerator()`, `denominator()`, `from_parts()`, and `try_from_integers()`), and `BigFloat` (`try_with_prec` preserving caller-requested bit precision across arithmetic and `serde` roundtrips while rejecting `prec == 0` with `ScalarError::ZeroPrecision` and non-finite inputs with `ScalarError::NonFiniteFloat`), integrated into `pure_math::math_types` and `algorithmic_information`.

2. **Split `FieldScalar` and `RealScalar`, and Unify Cayley-Dickson Algebras & `Complex<T>` Linear Solvers:**
   - Define unordered `FieldScalar` (`conj`, `norm_sqr`, `abs`, `checked_inv`, `try_inverse`, `from_f64`) and ordered `RealScalar: FieldScalar + PartialOrd` (`sqrt`, `hypot`, `sin`, `cos`, `exp`, `ln`, `powf`, `atan2`, etc.) in `math_scalars`, re-exported through `engine_linalg` and `math_commons` (`pub use math_scalars::RealScalar as Scalar`).
   - Implement flat-coordinate `#[repr(C)]` `Complex<T>`, `Quaternion<T>`, `UnitQuaternion<T>`, and `Octonion<T>` with Cayley-Dickson multiplication, norms, and fallible `try_inverse`.
   - Generalize `engine_linalg` LU factorization, restarted GMRES, vector/matrix norms, `DenseView`, `CsrMatrixView`, and `math_commons::SolverWorkspace` to `T: FieldScalar` (`T::Real: RealScalar`), and unify `domain_ai::Gaussian3D` (`UnitQuaternion<f64>`) and `oxidize_core::export_splat_ply` (`UnitQuaternion<f32>`).

3. **Carry-Safe 64-Bit Montgomery `Fp<const P: u64>`, `StackPoly`/`FpExt`, and Stack `Qp<const P: u64, const N: usize>`:**
   - Implement compile-time deterministic Miller-Rabin validated `Fp<const P: u64>` using 128-bit Montgomery REDC (`N' = -P^{-1} mod 2^{64}`, `R^2 mod P`) for odd primes up to `MAX_PRIME_U64 = 18_446_744_073_709_551_557` ($2^{64} - 59$) plus an explicit $P = 2$ path and fallible `try_inverse` / `checked_inverse`, replacing `Fp<const P: i64>` in `pure_math::algebra::fields` and retiring duplicate `i128` helpers in `elliptic_curves::modular_fp`.
   - Implement stack-backed `StackPoly<T, const CAP: usize>` with Euclidean `checked_div_rem` and extended GCD plus `FpExt<const P: u64, const DEG: usize, M>` Galois fields $\mathbb{F}_{p^n}$, bridged into `pure_math::algebra::polynomial`.
   - Implement fixed-precision stack $p$-adic numbers `Qp<const P: u64, const N: usize>` with ultrametric valuations and zero-allocation Newton-Hensel simple root lifting (`hensel_lift_root`), bridged into dedicated `pure_math::number_theory::ambs` and `pure_math::elliptic_curves` submodules.

## Consequences

- **Positive:** Complex linear solvers (`lu_solve`, `gmres_solve`), 3D `UnitQuaternion` splat rotations, 256-bit fixed integers, 64-bit Montgomery finite fields up to $2^{64} - 59$, $\mathbb{F}_{p^n}$ extension fields, and $\mathbb{Q}_p$ Hensel root lifting execute with zero dynamic heap allocations under `VerifiedAllocator`.
- **Positive:** Arbitrary-precision `BigFloat`/`Float` strictly honor caller bit precision (including across `serde` roundtrips) and reject `NaN`/`±Infinity` at construction boundaries, while `Rational` callers eliminate raw tuple-field coupling.
- **Neutral:** Workspace member count advances from `26` to `27` crates while preserving `math_commons::Scalar` compatibility for ordered real solvers.
