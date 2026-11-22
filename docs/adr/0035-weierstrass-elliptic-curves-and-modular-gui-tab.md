# ADR 0035: Zero-Allocation Short-Weierstrass Elliptic Curve Arithmetic and Auto-Discovered `EllipticCurvesTab`

- **Status:** Accepted
- **Date:** 2026-11-21

## Context

1. **Gap in `pure_math::elliptic_curves`:** Previously, [`crates/pure_math/src/pure_math/elliptic_curves/mod.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/mod.rs) implemented only Breuer's modular polynomial valuation bounds (`prime_factors`, `psi`, `theorem_1_1_bounds`, `theorem_1_2_bounds`), leaving out the short-Weierstrass group arithmetic advertised in [`crates/pure_math/README.md`](../../crates/pure_math/README.md) and leaving [`traceability.toml`](../../traceability.toml) research artifacts (`hurwitz_kronecker.tex` and `modular_polynomials_review.tex`) disconnected from finite-field curve point counts and modular $q$-series.
2. **Missing Interactive Visualization in `math_explorer_gui`:** [`math_explorer_gui/src/tabs/number_theory/mod.rs`](../../math_explorer_gui/src/tabs/number_theory/mod.rs) exposed four number-theory tools without surfacing elliptic curves, Frobenius traces $a_p$, imaginary quadratic class numbers (`class_number::BQF`, `class_number`), Hurwitz-Kronecker trace sums (`hurwitz_kronecker`), or Breuer modular valuation bounds.

## Decision

1. **Zero-Allocation Short-Weierstrass Engine (`pure_math::elliptic_curves::weierstrass`):**
   - Introduce stack-allocated `CurvePoint<T>` (`Infinity`, `Affine { x, y }`), `WeierstrassCurve` over `f64`, const-generic `FpWeierstrassCurve<const P: i64>` reusing [`Fp<P>`](../../crates/pure_math/src/pure_math/algebra/fields.rs), and runtime-modulus `PrimeWeierstrassCurve` in [`crates/pure_math/src/pure_math/elliptic_curves/weierstrass.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/weierstrass.rs) and [`crates/pure_math/src/pure_math/elliptic_curves/weierstrass/fp_curve.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/weierstrass/fp_curve.rs).
   - Reject singular curves ($4a^3 + 27b^2 = 0$) with `EllipticCurveError::SingularCurve`, reject small or composite moduli ($p \le 3$ or non-prime) with `EllipticCurveError::InvalidModulus`, and cap lattice enumeration at prime $p \le 997$ (`MAX_LATTICE_PRIME`) via `EllipticCurveError::ModulusExceedsLatticeCap`.
   - Instrument group-law methods (`add`, `negate`, `double`, `scalar_mul`), caller-provided slice writers (`write_orbit`, `write_affine_points`), and Euler-criterion point counting (`point_count`, `frobenius_trace`, `satisfies_hasse_weil`) with `#[verified_engine::verified]` so they execute with zero dynamic heap allocations inside `VerifiedEngine::run_verified`.

2. **Modular Invariants & Number-Theory Bridge (`pure_math::elliptic_curves::modular_bridge`):**
   - Add [`crates/pure_math/src/pure_math/elliptic_curves/modular_bridge.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/modular_bridge.rs) computing real and finite-field $j$-invariants, pairing them with `psi`, `theorem_1_1_bounds`, and `theorem_1_2_bounds`, linking Frobenius discriminants $D = a_p^2 - 4p$ to reduced principal `BQF`, `class_number`, `weighted_class_number`, `hurwitz_class_number`, and `verify_summation_formula`, and generating truncated `QSeries<i64>` expansions (`ramanujan_delta_q_series`, `eisenstein_e4_q_series`, and overflow-guarded `j_invariant_q_scaled`).
   - Re-export `weierstrass` and `modular_bridge` from [`crates/pure_math/src/pure_math/elliptic_curves/mod.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/mod.rs) while preserving all pre-existing public function signatures and aligning citations with `modular_polynomials_review` and `hurwitz_kronecker`.

3. **Auto-Discovered `EllipticCurvesTab` (`math_explorer_gui/src/tabs/elliptic_curves/`):**
   - Create [`math_explorer_gui/src/tabs/elliptic_curves/mod.rs`](../../math_explorer_gui/src/tabs/elliptic_curves/mod.rs) annotated with `// @explorer_feature = "pure_math"` and overriding `ExplorerTab::academic_domain` (`"Pure Math"`) and `ExplorerTab::url_slug` (`"elliptic_curves"`) directly so [`math_explorer_gui/src/tabs/mod.rs`](../../math_explorer_gui/src/tabs/mod.rs) and [`math_explorer_gui/src/tabs/number_theory/mod.rs`](../../math_explorer_gui/src/tabs/number_theory/mod.rs) remain untouched.
   - Register two `SimulationFramework` tools (`WeierstrassPlaygroundTool` and `ModularInvariantsTool`) and two `StarterPresetMetadata` entries validated by `#[serde(deny_unknown_fields)]` and `ToolStateBounds`, with `AccessKit` `GraphicsDocument` canvas keyboard navigation and `#tab=elliptic_curves` URL sharing.

## Consequences

- **Positive:** Cryptographic subgroup geometry, toroidal finite-field secants, Hasse-Weil Frobenius trace bands, Hurwitz-Kronecker trace sums, and Breuer $p$-adic modular valuations are unified across desktop and `wasm32` with zero-allocation core arithmetic and zero changes to existing `number_theory` or shell routing signatures.
- **Trade-offs:** Lattice point enumeration in `PrimeWeierstrassCurve` and GUI presets is bounded at $p \le 997$ and integer $q \cdot j(q)$ expansion is capped at 12 terms to guarantee deterministic frame times and `i64` overflow safety.

## Verification Commands

```bash
# Regenerate API baseline and run workspace tests
cargo run -p xtask -- regenerate-baseline
cargo test --workspace --lib --tests --all-features
cargo test --doc --workspace --all-features

# Verify feature isolation, WASM compilation, and repository lint gates
cargo run -p xtask -- test-features pure_math
cargo run -p xtask -- test-features core-only
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features
cargo run -p xtask -- lint-all
```
