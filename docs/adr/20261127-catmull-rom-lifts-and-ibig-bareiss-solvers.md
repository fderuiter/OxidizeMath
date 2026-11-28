# ADR 20261127: Interpolating Catmull-Rom Diagram Lifts, Euler Planarity, Dehornoy Braid Equality, and `IBig` Bareiss Solvers

## Status

Accepted

## Context

`pure_math::knot_theory` and `pure_math::algebra` previously exhibited several geometric and algebraic fidelity gaps:

1. **Non-interpolating B-spline 3D lifts and chordal braid segments**: `PlanarDiagram::from_braid` omitted waypoints on interior braid segments and routed closure arcs onto outer radii $0.84 + 0.04 \cdot s$, while `PlanarDiagram::lift_to_3d_link` smoothed control points with a uniform cubic B-spline that cut corners across crossing bridges and placed detached `free_loops` inside the lifted XY radius.
2. **Vertex-crossing sensitivity in `PlanarDiagram::from_3d_link`**: At composite sample counts such as `curve_samples = 72`, braid-closure crossings landed directly on polyline vertices (`s = 0` or `s = 1`), causing missed or duplicated crossings.
3. **Unchecked planarity, braid group relations, and fixed-width solver overflow**:
   - `PlanarDiagram::validate` verified local 4-valent slot reciprocity without checking connected-component Euler characteristic $V - E + F = 2$ or running during `serde::Deserialize`.
   - `BraidWord` derived structural `Vec<i32>` equality/hashing instead of Artin braid group equivalence, matched only homogeneous Yang-Baxter triples, and lacked `MAX_BRAID_WORD_LEN` guards on `Semigroup::operate` and `Deserialize`.
   - `BurauWorkspace::eval_alexander_in_place` hit a $0 / 0$ removable singularity at $t = -1.0$ on even strand counts where $1 + t + \dots + t^{n-1} = 0$.
   - `bareiss_laurent_det` over `i128` overflowed on high-crossing pseudo-Anosov braids, `kauffman_bracket` used unchecked `i64` loop powers above `62` loops, and `LaurentPolynomial` exposed public `min_exp` / `coeffs` fields without checked arithmetic companions or validated `Deserialize`.

## Decision

1. **Concentric Annular Braid Routing and Periodic Centripetal Catmull-Rom 3D Lifts (`diagram_braid.rs`, `diagram_lift/`)**:
   - Route both interior and closure braid arcs along disjoint concentric strand radii with angular collars so every non-crossing strand remains locked to its own radial track.
   - Replace the cubic B-spline in `lift_to_3d_link` with periodic centripetal Catmull-Rom interpolation ($\alpha = 0.5$) through exact `[x, y, +-z_bridge]` crossing bridges and tangent collars, elevate accidental non-crossing 2D Reidemeister overlaps onto disjoint $z$ tiers strictly outside `[-z_bridge, +z_bridge]`, and place detached `free_loops` outside the lifted XY bounding radius.

2. **General-Position $SO(3)$ Micro-Tilts in `PlanarDiagram::from_3d_link` (`project_3d.rs`)**:
   - Detect near-vertex crossing parameters (`s <= 1e-7` or `1.0 - s <= 1e-7`), parallel XY overlaps, and coincident crossings, applying a deterministic golden-angle $SO(3)$ micro-tilt schedule with shared-endpoint deduplication fallback.

3. **Rotation-System Euler Planarity, Reidemeister Fixes, and $5_2$ Lissajous Preset (`diagram_validate.rs`, `reidemeister/`, `curves.rs`)**:
   - Enforce `V - E + F == 2` per connected crossing component via half-edge rotation-system face cycles inside `PlanarDiagram::validate` and `serde::Deserialize`, and reject odd $A$/$q$ exponents in `kauffman_jones.rs` with `KnotError::InvalidDiagram`.
   - Support multi-component unlinks in `apply_reidemeister_i_add`, direct 2-crossing zero-writhe bigons on crossing-free diagrams plus all four planar face orientations in `apply_reidemeister_ii_add`, multi-slot triangle discovery in `match_riii_triangle`, and canonical $(3, 2, 7)$ / $(0.7, 0.2, 0.0)$ parameters for the 7-crossing $5_2$ `LissajousKnot::three_twist`.

4. **Artin Braid Equivalence, Zero-Allocation Dual Burau at $t = -1.0$, `IBig` Bareiss, and Checked `LaurentPolynomial` (`braid/`, `burau_workspace.rs`, `bareiss_ibig.rs`, `laurent_checked.rs`)**:
   - Support all homogeneous and mixed-sign conjugate Yang-Baxter triples (`[s0*i, s1*j, -s0*i] <-> [-s0*j, s1*i, s0*j]`), decide `PartialEq`/`Eq` on `BraidWord` via Dehornoy handle reduction of $a \cdot b^{-1}$ paired with a stack-allocated double-prime modular Burau `Hash`, add `BraidWord::try_operate`, and validate `Deserialize`.
   - Preallocate `Dual<f64>` scratch storage in `BurauWorkspace::new` to evaluate $N'(-1) / (n / 2)$ via zero-allocation dual Gaussian elimination when $|1 + t + \dots + t^{n-1}| < 10^{-12}$, and expose `try_reduced_burau_matrix`.
   - Implement `Ring` and `EuclideanDomain` for `dashu::integer::IBig` to execute arbitrary-precision fraction-free Bareiss determinants and exact polynomial division before checked `i64` conversion, guard `kauffman_bracket` loop powers at `MAX_SAFE_KAUFFMAN_LOOPS = 62`, encapsulate `LaurentPolynomial` behind `min_exp()` / `coeffs()`, and wire `LaurentError` into `PureMathError`, `KnotError`, and `OxidizeMathError`.

## Consequences

- **Positive**: Achieves `341 / 341` lossless 3-strand lift/projection roundtrips and `0 / 341` vertex-crossing duplications at `curve_samples = 72`, preserves zero-allocation `VerifiedEngine` guarantees at $t = -1.0$ across all even strand counts `2..=16`, eliminates integer overflows across polynomial-time Fox/Burau and bounded Kauffman solvers, and keeps `MAX_KAUFFMAN_CROSSINGS = 16`.
- **Negative / Trade-offs**: Exact `PartialEq` on `BraidWord` performs Dehornoy handle reduction rather than $O(n)$ slice comparison, bounded cleanly by `MAX_BRAID_WORD_LEN = 128` and fast-pathed on strand count, writhe, and equal generator slices.
