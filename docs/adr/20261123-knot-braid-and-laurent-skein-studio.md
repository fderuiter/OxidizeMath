# ADR 20261123: Verified Knot, Artin Braid, and Laurent Skein Studio

## Status

Accepted

## Context

`pure_math` and `math_explorer_gui::tabs::geometry_topology` previously modeled 2D parametric surfaces (`ParametricSurface`) and point-cloud persistent homology (`VietorisRipsTool`), omitting 1D spatial knots, multi-component links, infinite Artin braid groups $B_n$, and signed-exponent Laurent polynomial rings $R[t, t^{-1}]$. Educators and researchers lacked a client-side `wasm32` studio uniting smooth 3D Alexander braid closures, interactive planar Reidemeister I/II/III moves, live topological invariants, and 3D-printable watertight tubular mesh exports.

## Decision

1. **Signed-Exponent `LaurentPolynomial<T>` (`pure_math::algebra::polynomial`)**:
   - Store canonicalized contiguous coefficients starting at `min_exp: i32` over any `Ring`, supporting `Add`, `Sub`, `Mul`, `Neg`, monomial `shift`, variable inversion $t \mapsto t^{-1}$, reciprocal checks, and canonical Alexander unit normalization modulo $\pm t^k$.
   - Add `Ring` and `EuclideanDomain` implementations for `i128` to support overflow-free Bareiss fraction-free polynomial determinants.

2. **Infinite Artin Braid Groups (`pure_math::algebra::group::BraidWord`)**:
   - Represent elements of $B_n$ ($1 \le n \le 16$) as bounded signed generator words supporting composition, group inversion, mirror reflection, free inverse cancellation, far commutativity ($|i - j| \ge 2$), Artin Yang-Baxter relations ($\sigma_i^{\pm 1} \sigma_{i+1}^{\pm 1} \sigma_i^{\pm 1}$), strand permutations in $S_n$, and $C^2$-smooth 3D Alexander torus closures.

3. **Zero-Allocation 3D Geometry and Gauss Quadrature (`pure_math::knot_theory`)**:
   - Implement parametric `TorusKnot`, `LissajousKnot`, `figure_eight_knot`, `hopf_link`, and $SO(3)$-symmetric `borromean_rings`.
   - Compute closed-loop holonomy-corrected Bishop parallel-transport frames (`compute_bishop_frames_in_place`) to extrude watertight, outward-oriented `oxidize_core::mesh::Mesh` tubes without Frenet inflection flips.
   - Evaluate continuous 3D Gauss writhe and pairwise integer Gauss linking numbers via zero-allocation 2-point Gauss-Legendre segment quadrature (`gauss_writhe_quadrature`, `gauss_linking_quadrature`, `pairwise_linking_numbers_in_place`), plus zero-allocation numerical Burau stepping (`BurauWorkspace`), verified under `VerifiedEngine::run_verified`.

4. **Oriented Planar Diagrams, Reidemeister Moves, and Skein Cap**:
   - Model 4-valent oriented planar link diagrams (`PlanarDiagram`) with counter-clockwise PD slots, supporting 3D projection (`from_3d_link`), smooth cubic B-spline 3D lifting (`lift_to_3d_link`), and validated Reidemeister I, II, and III moves.
   - Compute polynomial-time Alexander polynomials via both Fox Wirtinger free derivatives and reduced Burau determinants, and cap exact $O(2^c)$ Kauffman bracket / Jones state-sum expansion at `MAX_KAUFFMAN_CROSSINGS = 16` so `wasm32` UI frames never stall.

5. **Interactive `KnotBraidStudioTool` (`math_explorer_gui::tabs::geometry_topology`)**:
   - Synchronize `Plot3DViewport` / `Scene3DBuilder` depth-sorted 3D tubes with an accessible keyboard-navigable 2D planar diagram editor (featuring under-crossing gap cues and WCAG 2.2 AA contrast across Dark, Light, and HighContrast themes), `URL_JSON_BOUNDS` (`4 KiB`) validation, starter presets, and `.stl`/`.ply`/`.obj` solid exports.

## Consequences

- **Positive**: Unifies 3D geometric topology, combinatorial Reidemeister isotopy, and exact algebraic skein invariants in pure Rust and `wasm32` while preserving strict zero-allocation guarantees in hot numerical loops and bounded UI frame times.
- **Negative / Trade-offs**: Exact state-sum Jones polynomials are bounded to $c \le 16$ crossings; higher-crossing diagrams fall back gracefully to polynomial-time writhe, Gauss linking numbers, and Fox/Burau Alexander polynomials with an explicit UI crossing-cap notice.
