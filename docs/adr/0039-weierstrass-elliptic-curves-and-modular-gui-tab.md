# ADR 0039: Zero-Allocation Short-Weierstrass Elliptic Curve Arithmetic and Auto-Discovered `EllipticCurvesTab`

## Status

Accepted (2026-11-22)

## Context

1. **Complementary Real, Const-Generic, and Modular Bridges in `pure_math::elliptic_curves`:** Alongside runtime $\mathbb{F}_p$ torus curves in [`crates/pure_math/src/pure_math/elliptic_curves/weierstrass.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/weierstrass.rs), `pure_math::elliptic_curves` needed continuous real short-Weierstrass curves over $\mathbb{R}$, compile-time `Fp<P>` curve arithmetic reusing [`algebra::fields::Fp`](../../crates/pure_math/src/pure_math/algebra/fields.rs), and direct modular bridges connecting [`traceability.toml`](../../traceability.toml) research artifacts (`hurwitz_kronecker.tex` and `modular_polynomials_review.tex`) to imaginary quadratic class numbers (`class_number::BQF`, `class_number`), Hurwitz-Kronecker trace sums, and modular $q$-series (`QSeries<i64>`).
2. **Dedicated Elliptic Curves Domain Tab in `math_explorer_gui`:** Exposing a standalone **Elliptic Curves** domain tab (`#tab=elliptic_curves`) under **Pure Math** lets users explore continuous $\mathbb{R}$ and toroidal $\mathbb{F}_p \times \mathbb{F}_p$ chord-and-tangent addition alongside multi-lens modular invariants (Hasse-Weil bands, Hurwitz-Kronecker trace identities, Breuer modular polynomial valuations, and Ramanujan/Eisenstein $q$-series) without modifying [`math_explorer_gui/src/tabs/mod.rs`](../../math_explorer_gui/src/tabs/mod.rs) or [`math_explorer_gui/src/tabs/number_theory/mod.rs`](../../math_explorer_gui/src/tabs/number_theory/mod.rs).

## Decision

1. **Zero-Allocation Real, Const-Generic, and Lattice-Bounded Weierstrass Submodules:**
   - Introduce stack-allocated `CurvePoint<T>` (`Infinity`, `Affine { x, y }`) and `WeierstrassCurve` over `f64` in [`crates/pure_math/src/pure_math/elliptic_curves/weierstrass/real_curve.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/weierstrass/real_curve.rs), plus const-generic `FpWeierstrassCurve<const P: i64>` and lattice-bounded `PrimeWeierstrassCurve` in [`crates/pure_math/src/pure_math/elliptic_curves/weierstrass/fp_curve.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/weierstrass/fp_curve.rs), reusing modular helpers from [`crates/pure_math/src/pure_math/elliptic_curves/modular_fp.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/modular_fp.rs).
   - Reject singular real curves with `EllipticCurveError::SingularRealCurve`, reject invalid or singular modular curves with `EllipticCurveError::InvalidModulus` and `EllipticCurveError::SingularCurve`, and cap `PrimeWeierstrassCurve` lattice enumeration at prime $p \le 997$ (`MAX_LATTICE_PRIME`) via `EllipticCurveError::ModulusExceedsLatticeCap`.
   - Instrument all group-law kernels (`add`, `negate`, `double`, `scalar_mul`), caller-provided slice writers (`write_orbit`, `write_affine_points`), and Euler-criterion point counting (`point_count`, `frobenius_trace`, `satisfies_hasse_weil`) with `#[verified_engine::verified]` for zero dynamic heap allocations inside `VerifiedEngine::run_verified`.

2. **Modular Invariants & Number-Theory Bridge (`pure_math::elliptic_curves::modular_bridge`):**
   - Add [`crates/pure_math/src/pure_math/elliptic_curves/modular_bridge.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/modular_bridge.rs) computing real and finite-field $j$-invariants, pairing them with `psi`, `theorem_1_1_bounds`, and `theorem_1_2_bounds`, linking Frobenius discriminants $D = a_p^2 - 4p$ to reduced principal `BQF`, `class_number`, `weighted_class_number`, `hurwitz_class_number`, and `verify_summation_formula`, and generating truncated `QSeries<i64>` expansions (`ramanujan_delta_q_series`, `eisenstein_e4_q_series`, and overflow-guarded `j_invariant_q_scaled`).

3. **Auto-Discovered `EllipticCurvesTab` (`math_explorer_gui/src/tabs/elliptic_curves/`):**
   - Create [`math_explorer_gui/src/tabs/elliptic_curves/mod.rs`](../../math_explorer_gui/src/tabs/elliptic_curves/mod.rs) annotated with `// @explorer_feature = "pure_math"` and overriding `ExplorerTab::academic_domain` (`"Pure Math"`) and `ExplorerTab::url_slug` (`"elliptic_curves"`).
   - Register two `SimulationFramework` tools (`WeierstrassPlaygroundTool` and `ModularInvariantsTool`) and two `StarterPresetMetadata` entries validated by `#[serde(deny_unknown_fields)]` and `ToolStateBounds`, with `AccessKit` `GraphicsDocument` canvas keyboard navigation and `#tab=elliptic_curves` URL sharing.

## Consequences

- **Positive:** Real and finite-field short-Weierstrass geometry, Hasse-Weil Frobenius trace bands, Hurwitz-Kronecker trace sums, and Breuer $p$-adic modular valuations coexist cleanly across desktop and `wasm32` with zero-allocation core arithmetic and zero duplicate modular primitives.
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
