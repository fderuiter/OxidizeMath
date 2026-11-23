# ADR 0047: Poincaré Disk Hyperbolic Geometry Kernel and Geodesic Billiard Laboratory

## Status
Accepted (extends [ADR 0004](0004-unified-verification-dependency-audit.md) and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/pure_math/src/pure_math/differential_geometry/surface.rs`](../../crates/pure_math/src/pure_math/differential_geometry/surface.rs) previously restricted differential-geometric visualizations to extrinsic 3D Euclidean surface embeddings (`Point3<f64>`), leaving constant negative Gaussian curvature ($K = -1$) conformal models unrepresented in both `pure_math` and `GeometryTopologyTab`. Interactive exploration of hyperbolic tilings and Anosov-type geodesic billiards requires:
1. Cancellation-safe arithmetic on the Poincaré unit disk $\mathbb{D}$ and upper half-plane $\mathbb{H}$ paired via the isometric Cayley map $w = i \frac{1 + z}{1 - z}$.
2. Renormalized $PSU(1,1)$ and $PSL(2,\mathbb{R})$ Möbius transformations preventing boundary drift onto $|z| = 1$.
3. Zero-allocation step and drag hot loops under `VerifiedAllocator` in `verified_engine`.

## Decision
1. **Verified `pure_math::hyperbolic` Kernel**:
   - Introduce [`crates/pure_math/src/pure_math/hyperbolic/`](../../crates/pure_math/src/pure_math/hyperbolic/mod.rs) exported through `math_explorer::pure_math::hyperbolic` with `#[verified_engine::verified]` types (`DiskPoint`, `HalfPlanePoint`, `Psu11`, `Psl2R`, `CircleInversion`, `GeodesicDisk`, `GeodesicHalfPlane`, `HorocycleDisk`, `HorocycleHalfPlane`, `HyperbolicTilingBuffer`, and `TwinBilliardSystem`).
   - Enforce bounded memory and loop invariants: Coxeter BFS tilings (`{p, q}` with $(p - 2)(q - 2) > 4$ and Hecke `{infinity, q}` apeirogons) cap reflection depth at `MAX_TILING_DEPTH = 5` and tile count at `MAX_TILING_POLYGONS = 512`; geodesic billiard ring buffers pre-allocate `MAX_BILLIARD_SEGMENTS = 2048` slots seeded deterministically via `OxidizeRng`.
2. **Synchronized Dual-Chart `HyperbolicGeometryTool`**:
   - Register `HyperbolicGeometryTool` and two federated `StarterPresetMetadata` templates inside [`math_explorer_gui/src/tabs/geometry_topology/`](../../math_explorer_gui/src/tabs/geometry_topology/mod.rs).
   - Provide live $PSU(1,1)$ disk and $PSL(2,\mathbb{R})$ half-plane pointer/keyboard Möbius dragging, `.svg`/`.png`/`.gif`/`.csv`/`.json` plot exports via `ExportCapabilities::plot()`, `FiniteStateGuard` rollback hooks, AccessKit focus rings, and `#tab=geometry_topology&tool=HyperbolicGeometryTool&state=...` share links.

## Consequences
- **Positive**: Downstream SDK consumers and `math_explorer_gui` gain exact $K = -1$ conformal geometry, Gauss-Bonnet area verification, Coxeter tilings, and chaotic twin-ray Lyapunov separation $\ln(\delta_t / \delta_0)$ with zero post-initialization heap allocations in simulation and drag hot paths.
- **Neutral**: Expanding the `pure_math` public API updates [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) and increments the registered `InteractiveTool` (`76`) and `StarterPresetMetadata` (`110`) inventory totals.
