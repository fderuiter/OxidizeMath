# Elliptic Curve Discriminant Atlas, Singular Cubic Rendering, and Hasse-Weil Challenges

## Status

Accepted (2026-11-23)

## Context

Three interactive tools span OxidizeMath's elliptic-curve workflows across two domain tabs:

- `WeierstrassPlaygroundTool` and `ModularInvariantsTool` in `EllipticCurvesTab` (`elliptic_curves`).
- `EllipticCurvePlayground` in `NumberTheoryTab` (`number_theory`).

Previously:

1. Users adjusted Weierstrass parameters `(a, b)` via isolated 1D sliders with no 2D moduli-space cartography of the discriminant locus $\Delta = -16(4a^3 + 27b^2) = 0$, complex-multiplication rays $J = 0$ ($a = 0$) and $J = 1728$ ($b = 0$), or finite-field group orders $\#E(\mathbb{F}_p)$.
2. Hitting singular parameter pairs (roughly $20.08\%$ of integer slider pairs at $p = 5$, plus real pinch points $(0, 0)$, $(-3, 2)$, and $(-3, -2)$) blanked curve rendering, emptied dataset exports, or rejected state loads because `WeierstrassCurve::new`, `PrimeWeierstrassCurve::new`, and `WeierstrassCurveFp::new` enforce smooth non-zero discriminants by algebraic contract.
3. None of the three tools exposed cross-tab `(p, a, b, P, Q, k)` handoffs, framework `observables()`, guided `PresetStoryMetadata` tours, `GoalChallengeMetadata` AST predicates, or locked-prime `ChallengeMetadata` inverse challenges.

## Decision

1. **Zero-Allocation Singular Cubic Companion (`crates/pure_math/src/pure_math/elliptic_curves/singular.rs`)**:
   - Preserve `WeierstrassCurve::new`, `PrimeWeierstrassCurve::new`, and `WeierstrassCurveFp::new` smooth contracts unchanged (`api_baseline.json` compatibility).
   - Introduce `SingularityKind` (`Cusp`, `Crunode`, `Acnode`), `RealSingularCubic`, and `FpSingularCubic` with `#[verified_engine::verified]` zero-allocation classification, smooth-locus point counting ($p$, $p - 1$, $p + 1$), preallocated $\mathbb{F}_p$ affine/projective lattice enumeration capped at `MAX_LATTICE_PRIME = 997`, and smooth-locus addition that suppresses only undefined secant/tangent slopes at $(x_s, 0)$.

2. **Verified 2D Moduli-Space Cartographer (`WeierstrassRegimeCartographer`) & `ParameterAtlasController`**:
   - Implement `RegimeCartographer` in `math_explorer_gui/src/tabs/elliptic_curves/atlas.rs` with `#[verified_engine::verified]` on `evaluate_cell` (zero heap allocations under `VerifiedAllocator`).
   - Map real `(a, b) ∈ [-4, 4]^2` regimes ($\Delta > 0$ oval+branch, $\Delta < 0$ single-branch, $J = 0$ and $J = 1728$ CM rays, and the $\Delta = 0$ semicubical cusp contour $27b^2 = -4a^3$) and $\mathbb{F}_p$ integer `(a, b) ∈ [-15, 15]^2` regimes ($\#E(\mathbb{F}_p)$, supersingular $a_p = 0$, Hasse-Weil extremal boundary $|a_p| = \lfloor 2\sqrt{p}\rfloor$, and singular cubics).
   - Embed `ParameterAtlasController` in `WeierstrassPlaygroundTool`, `ModularInvariantsTool`, and `EllipticCurvePlayground` with hover previews, arrow-key navigation, and click-to-teleport parameter updates.

3. **Singular Geometry Rendering, Cross-Tab Handoffs, Observables, and Challenges**:
   - Render real crunodes (crossing tangents $\pm\sqrt{3x_s}$ at $(x_s, 0)$), isolated acnodes at $(x_s, 0)$ plus the $x \ge -2x_s$ smooth branch, origin cusps $y^2 = x^3$, and $\mathbb{F}_p$ singular affine sets highlighting $(x_s, 0)$ across all canvases with non-empty dataset exports.
   - Route one-click `(p, a, b, P, Q, k)` handoffs across `EllipticCurvesTab` and `NumberTheoryTab` via `domain_bar.rs:poll_cross_tab_tool_handoffs` while preserving URL bookmark export/import.
   - Expose 9 finite scalar `observables()` (`discriminant`, `real_components`, `j_invariant`, `group_order`, `frobenius_trace`, `hasse_gap`, `point_order`, `cofactor`, `is_singular`), implement `evaluate_challenge()`, and register guided `PresetStoryMetadata` tours, `GoalChallengeMetadata` AST goals, and the locked-prime `hasse_weil_boundary_p23` `ChallengeMetadata` in `starter_gallery/elliptic_packs.rs`.

## Consequences

- Smooth elliptic-curve group-law APIs remain strictly non-singular and backwards-compatible, while singular degenerations render geometrically and export valid plot datasets across $\mathbb{R}$ and $\mathbb{F}_p$.
- `WeierstrassRegimeCartographer::evaluate_cell` executes with zero heap allocations inside background progressive-refinement tiles on both native and `wasm32` targets.
- Lattice enumeration remains strictly bounded by $p \le 997$; pairings, isogeny volcanoes, and Schoof point counting remain out of scope.
