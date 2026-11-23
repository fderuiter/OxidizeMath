# ADR 0046: Zero-Allocation Laplace-Beltrami Surface Morphogenesis and Mean-Curvature Buckling

## Status

Accepted (2026-11-23)

## Context

1. **Intrinsic Manifold Diffusion vs. Euclidean Grids:** [`TuringSystem`](../../crates/domain_biology/src/biology/morphogenesis/solvers.rs) in `domain_biology` previously evaluated reaction-diffusion kinetics (`SchnakenbergKinetics` and `GrayScottKinetics`) over flat Euclidean finite-difference grids, while [`heat_equation.rs`](../../crates/pure_math/src/pure_math/differential_geometry/heat_equation.rs) in `pure_math::differential_geometry` clamped parametric boundaries and [`mean_curvature_flow.rs`](../../crates/pure_math/src/pure_math/differential_geometry/mean_curvature_flow.rs) allocated nested `Vec<Vec<Point3<f64>>>` buffers per flow step.
2. **Coupled Morphogen-Driven Buckling & Interactive 3D Studio:** Biological pattern formation on curved 2-manifolds couples metric-weighted Laplace-Beltrami diffusion $\Delta_g u = \frac{1}{\sqrt{|g|}} \partial_i (\sqrt{|g|} g^{ij} \partial_j u)$ across seamless periodic, twisted non-orientable, polar-singular, and open-Neumann topologies to normal mechanical buckling against mean-curvature surface tension. Exposing this cross-domain system inside **Geometry & Topology** (`#tab=geometry_topology&tool=SurfaceMorphogenesisTool`) requires feature gating behind `#[cfg(feature = "biology")]`, zero heap allocations on hot `step` paths under `VerifiedAllocator`, CFL stability clamping with `FiniteStateGuard` rollback, sonification, compact URL state sharing, and watertight vertex-colored `.stl`/`.ply`/`.obj` exports.

## Decision

1. **Topology-Aware Flat-Buffer `BuckledSurfaceFlow` and `HeatEquationSolver` (`pure_math::differential_geometry`):**
   - Introduce `SurfaceTopology` (`PeriodicTorus`, `TwistedKlein`, `PolarSphere`, `OpenNeumann`) and pre-allocated flat-buffer `BuckledSurfaceFlow` in [`crates/pure_math/src/pure_math/differential_geometry/buckled_flow.rs`](../../crates/pure_math/src/pure_math/differential_geometry/buckled_flow.rs) and [`crates/pure_math/src/pure_math/differential_geometry/buckled_flow_geometry.rs`](../../crates/pure_math/src/pure_math/differential_geometry/buckled_flow_geometry.rs), indexed via `flatten_2d_index(col, row, cols)`.
   - Auto-detect both half-turn (`v -> v + pi`) and orientation-reversing (`v -> 2*pi - v`) Klein bottle seam bijections, weld north/south polar cap rings on `Sphere`, and enforce zero-allocation `step_buckled` updates under `VerifiedEngine::run_verified` with finite-state rollback.
   - Extend [`crates/pure_math/src/pure_math/differential_geometry/heat_equation.rs`](../../crates/pure_math/src/pure_math/differential_geometry/heat_equation.rs) with `with_topology(SurfaceTopology)` so manifold heat flow shares identical seam and polar boundary semantics.

2. **Metric-Weighted `LaplaceBeltramiDiffusion2D` (`domain_biology::diffusion`):**
   - Add [`crates/domain_biology/src/biology/diffusion/laplace_beltrami/mod.rs`](../../crates/domain_biology/src/biology/diffusion/laplace_beltrami/mod.rs) and [`crates/domain_biology/src/biology/diffusion/laplace_beltrami/stencil.rs`](../../crates/domain_biology/src/biology/diffusion/laplace_beltrami/stencil.rs) implementing `SpatialDiffusion<N>` for `TuringSystem` using conservative staggered metric fluxes assembled from `first_fundamental_form`, `area_element`, and `metric_tensor_inverse`.
   - Pre-compute per-vertex Gaussian curvature, principal metric anisotropy, and minimum cell spectral radii so `clamp_dt` enforces explicit CFL stability with zero hot-loop allocations.

3. **Feature-Gated `SurfaceMorphogenesisTool`, Sonification, and Watertight 3D Exports (`math_explorer_gui`):**
   - Register `SurfaceMorphogenesisTool` behind `#[cfg(feature = "biology")]` in [`math_explorer_gui/src/tabs/geometry_topology/mod.rs`](../../math_explorer_gui/src/tabs/geometry_topology/mod.rs) and [`math_explorer_gui/src/tabs/geometry_topology/surface_morphogenesis/mod.rs`](../../math_explorer_gui/src/tabs/geometry_topology/surface_morphogenesis/mod.rs), integrating `CustomSurfaceSandbox` AST formulas, frontmost depth-picked 3D pulse brushing, and `MorphogenConcentration` / `GaussianCurvature` / `MetricAnisotropy` overlays.
   - Route morphogen spatial variance and mean-curvature buckling shifts to [`surface_morphogenesis_frame`](../../math_explorer_gui/src/audio/mappings/surface_morphogenesis.rs), enforce zero PDE steps inside `Default::default()` and `try_load_state()` within `URL_JSON_BOUNDS`, and export closed, outward-oriented `.stl`, `.ply`, and `.obj` solids in [`math_explorer_gui/src/tabs/geometry_topology/surface_morphogenesis/export.rs`](../../math_explorer_gui/src/tabs/geometry_topology/surface_morphogenesis/export.rs).

## Consequences

- **Positive:** Turing stripes and Gray-Scott mitosis spots propagate seamlessly across Torus seams, twisted Klein bottle identifications, and Sphere poles, dynamically buckling 3D surfaces into watertight printable solids with zero hot-loop heap allocations.
- **Trade-offs:** Explicit CFL clamping reduces effective `dt` near high-curvature or polar cells to guarantee numerical stability without allocating sparse linear solvers on WASM.

## Verification Commands

```bash
# Format, regenerate API baseline, and run static + AST + ADR gates
cargo fmt --all
cargo run -p xtask -- regenerate-baseline
cargo run -p unified_verification -- batch-static
cargo run -p unified_verification -- batch-ast

# Run domain and GUI verification suites
cargo test -p pure_math --lib buckled_flow
cargo test -p domain_biology --lib
cargo test -p math_explorer_gui --bins surface_morphogenesis
cargo run -p xtask -- test-features pure_math
cargo run -p xtask -- test-features biology
```
