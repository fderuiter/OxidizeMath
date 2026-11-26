# ADR 20261125: Tier 0 Context-Capable `math_scalars`, `0`-ULP `libm` Parity, and Inline `N <= 8` Curve-Fitting Micro-Solvers

## Status

Accepted

## Context

OxidizeMath previously split scalar contracts across `math_commons::math_kernel::types::Scalar`, `engine_linalg::scalar::Scalar`, `pure_math::pure_math::algebra::traits::{Ring, Field}`, and a private `EvalScalar` trait inside `math_commons::ast::eval`. Those traits required `Copy`, `PartialOrd`, or `std`, excluding `Complex<T>`, forward-mode automatic differentiation (`Dual<T>`), runtime prime fields $\mathrm{GF}(p)$, and dynamic-precision `dashu` arithmetic without global state.

In addition, two structural and numerical defects remained:

1. `engine_linalg::scalar::sqrt_f64` used an unrounded Heron refinement that drifted by `1` ULP on `25.12%` of normal `f64` inputs (for example rounding `3.9999999999999996_f64.sqrt()` to `2.0` instead of `1.9999999999999998`).
2. Tier 0 `math_commons` depended upward on Tier 2 `engine_linalg` solely for three bounded `N <= 8` curve-fitting solve sites in `math_commons::fit::{poly, lm}`, while `engine_pde` (`44 / 46` scalar functions) and `math_commons` numeric workflows hardcoded concrete `f64`.

## Decision

1. **Create `#![no_std]` Tier 0 `crates/00_foundation/math_scalars`**:
   - Define the stateless algebraic hierarchy `Scalar -> Ring -> Field -> ComplexField -> RealField` alongside `#![no_std]` `Complex<T>` and forward-mode `Dual<T>` scalars.
   - Pair stateless traits with explicit `DomainRing` and `DomainField` contexts (`StatelessContext<T>`, `#![no_std]` `ModPrimeContext` over Montgomery/128-bit reduced $\mathrm{GF}(p)$, and `alloc`-gated `DashuFloatContext` / `DashuModContext`) so runtime-modulus and arbitrary-precision polynomial evaluation requires zero `std` thread-local state.
   - Delegate `<f32 as RealField>::{sqrt, hypot}` and `<f64 as RealField>::{sqrt, hypot}` (plus `engine_linalg::scalar::{sqrt_f64, hypot_f64}`) directly to `libm`, achieving `0`-ULP IEEE-754 parity across `200,000` deterministic normal `f64` samples and `4^k` predecessors.
   - Promote `LinalgError` into Tier 0 `crates/00_foundation/math_diagnostics` and re-export it from `engine_linalg` and `math_commons::error`.

2. **Inline `N <= 8` Stack Micro-Solvers in `math_commons::fit` and Drop `engine_linalg`**:
   - Implement zero-allocation 2D stack-array (`[[T; 8]; 8]`) coupled Givens QR (`StreamingGivensQr`, `solve_givens_dense_small`) and square-root-free symmetric $LDL^\top$ (`solve_marquardt_ldlt`) micro-solvers in `crates/math_commons/src/fit/micro_solvers.rs`.
   - Remove `engine_linalg` from `crates/math_commons/Cargo.toml`, keeping general-purpose `N > 8` dense and sparse factorizations exclusively in Tier 2 `crates/02_engines/engine_linalg`.

3. **Unify `pure_math`, `math_commons`, `engine_diffeq`, and `engine_pde`**:
   - Unify `pure_math::pure_math::algebra` (`Fp<P>`, `Polynomial<T>`, `LaurentPolynomial<T>`, `eval_in_domain`, and `Integer`/`Rational`/`Float`) and `math_commons::ast::{eval, compiled}` onto `math_scalars`, deleting private `EvalScalar`.
   - Generalize `engine_pde`, `engine_diffeq`, and `math_commons` (`fit`, `poi`, `stochastic`, `series`, `primitives`, `wigner`, and `math_kernel`) over `T: RealField` and `T: ComplexField` with `= f64` default type parameters on public workspace structs.

```text
Tier 0 Foundation: math_diagnostics, math_rng, math_scalars, math_commons
Tier 2 Engines:    engine_linalg, engine_diffeq, engine_pde -> depend downward on Tier 0
```

## Consequences

- Workspace crate count increases from `26` to `27` synchronized across [Cargo.toml](../../Cargo.toml), [system_overview.md](../architecture/system_overview.md), `xtask`, `unified_verification`, `traceability.toml`, and `verification_whitelist.toml`.
- `math_commons` is now a self-contained Tier 0 crate with zero upward Tier 2 engine dependencies, while preserving zero heap allocations under `VerifiedAllocator` and identical `R^2` / RMSE tolerances across curve-fitting suites.
- Continuum stencils, radix-2 FFTs, stochastic integrators, curve fitting, and AST evaluators execute generically over `f32`, `f64`, `Complex<T>`, and `Dual<T>`.
- `oxidize_core` removes its unused direct `rand` dependency in favor of `math_rng` re-exports, `egui_plot` plateau compaction scans `as_chunks::<8>()` slices without per-chunk copies in debug builds while clamping GIF frame sliders on interactive edits only, and `scripts/verify_wasm_budget.sh` enforces a `5 MB` compressed `.wasm` budget across the expanded multi-domain workspace.
