# ADR 0035: Runtime-Prime Short Weierstrass Elliptic Curves and Torus Hasse-Weil Explorer

## Status
Accepted (extends [ADR 0004](0004-unified-verification-dependency-audit.md) and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/pure_math/src/pure_math/elliptic_curves/mod.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/mod.rs) previously exposed classical modular polynomial degree and $p$-adic valuation bounds (`psi`, `theorem_1_1_bounds`, `theorem_1_2_bounds`), while [`crates/pure_math/src/pure_math/algebra/fields.rs`](../../crates/pure_math/src/pure_math/algebra/fields.rs) models compile-time `const P: i64` prime fields (`FiniteField<P>`). Interactive number-theory exploration across user-selected primes $5 \le p \le 997$ and Hasse-Weil Frobenius trace sweeps requires runtime-prime short Weierstrass curve arithmetic without modifying `FiniteField<P>` or incurring hot-loop heap allocations.

## Decision
1. **Stack-Only Runtime-Prime Short Weierstrass Arithmetic in `pure_math::elliptic_curves`**:
   - Add [`crates/pure_math/src/pure_math/elliptic_curves/weierstrass.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/weierstrass.rs) and [`crates/pure_math/src/pure_math/elliptic_curves/modular_fp.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/modular_fp.rs), re-exporting `WeierstrassCurveFp`, `AffinePointFp`, `ChordStepFp`, `HasseSweepEntry`, `mod_pow_fp`, `mod_inv_fp`, `tonelli_shanks_fp`, and `sweep_hasse_bounds` from [`crates/pure_math/src/pure_math/elliptic_curves/mod.rs`](../../crates/pure_math/src/pure_math/elliptic_curves/mod.rs) without altering [`crates/pure_math/src/pure_math/algebra/fields.rs`](../../crates/pure_math/src/pure_math/algebra/fields.rs).
   - Validate prime moduli via `pure_math::number_theory::primes::is_prime`, reject singular cubic discriminants $4a^3 + 27b^2 \equiv 0 \pmod p$, and write point enumerations (`enumerate_points`), cyclic torsion orbits (`torsion_orbit`), and Hasse sweeps (`sweep_hasse_bounds`) into caller-supplied slices with zero dynamic heap allocations under `VerifiedAllocator`.
2. **Interactive Torus Lattice and Hasse-Weil Explorer in `math_explorer_gui`**:
   - Add [`math_explorer_gui/src/tabs/number_theory/elliptic_curve_playground.rs`](../../math_explorer_gui/src/tabs/number_theory/elliptic_curve_playground.rs), [`math_explorer_gui/src/tabs/number_theory/elliptic_curve_canvas.rs`](../../math_explorer_gui/src/tabs/number_theory/elliptic_curve_canvas.rs), and [`math_explorer_gui/src/tabs/number_theory/elliptic_curve_hasse.rs`](../../math_explorer_gui/src/tabs/number_theory/elliptic_curve_hasse.rs) registered via `ToolMetadata` and `StarterPresetMetadata` inside [`math_explorer_gui/src/tabs/number_theory/mod.rs`](../../math_explorer_gui/src/tabs/number_theory/mod.rs).
   - Pair an AccessKit-accessible $(\mathbb{Z}/p\mathbb{Z})^2$ torus lattice canvas (animating Gauss-reduced wrap-around secant/tangent geodesics, vertical reflection to $P + Q$, `[k]P` orbits, and a singularity guardrail when $4a^3 + 27b^2 \equiv 0 \pmod p$) with an `egui_plot` Hasse-Weil arch chart ($a_q = q + 1 - \#E(\mathbb{F}_q)$ inside $[-2\sqrt{q}, 2\sqrt{q}]$).

## Consequences
- **Positive**: Downstream users of [`math_explorer`](../../math_explorer/src/lib.rs) and the `NumberTheoryTab` GUI gain zero-allocation runtime-prime elliptic curve arithmetic, torus chord-and-tangent visualization, and Hasse-Weil trace sweeps across `5 <= p <= 997` on native and `wasm32-unknown-unknown`.
- **Neutral**: Constant-time cryptographic signing, Weil/Tate pairings, and isogeny volcanoes remain intentionally outside scope.
