# ADR 20261123: Zero-Allocation CSR Spatial Grid and Cortical Thickness Workspace

## Status
Accepted (extends [ADR 0018](0018-uniform-spatial-grid-cortical-thickness.md) and [ADR 20261123](20261123-freesurfer-deformable-mrf-and-cortical-morphometry.md))

## Context
[ADR 0018](0018-uniform-spatial-grid-cortical-thickness.md) accelerated nearest-vertex surface queries in [`crates/domain_applied/src/applied/freesurfer/thickness.rs`](../../crates/03_domains/domain_applied/src/applied/freesurfer/thickness.rs) using a uniform 3D spatial grid (`SpatialGrid`), and [ADR 20261123](20261123-freesurfer-deformable-mrf-and-cortical-morphometry.md) added batch `cortical_thickness_map` to avoid rebuilding `SpatialGrid` per vertex. However:
1. `SpatialGrid::from_surface` still allocated a `std::collections::HashMap<(i64, i64, i64), Vec<[f64; 3]>>` and per-cell `Vec` buckets on the heap, causing thousands of `VerifiedAllocator` runtime violations when invoked under `VerifiedEngine::run_verified` (violating NASA Power of 10 Rule 3 in [Zero-Allocation Runtime Guide](../architecture/zero_allocation_runtime.md)).
2. Standalone `point_to_surface_distance` and `cortical_thickness` rebuilt and dropped two `HashMap` grids on every single-vertex call, incurring heap churn and $O(V)$ setup overhead per query.
3. `CorticalMorphometryTool` in [`math_explorer_gui/src/tabs/neuroscience/cortical_morphometry/phantom.rs`](../../apps/math_explorer_gui/src/tabs/neuroscience/cortical_morphometry/phantom.rs) reallocated fresh spatial grids and output thickness vectors on each surface evaluation.

## Decision
1. **Core-Only CSR `FlatSpatialGrid<'a>` and `ThicknessWorkspace` (`domain_applied`)**:
   - Replace `HashMap`-backed `SpatialGrid` in [`thickness.rs`](../../crates/03_domains/domain_applied/src/applied/freesurfer/thickness.rs) with [`FlatSpatialGrid`](../../crates/03_domains/domain_applied/src/applied/freesurfer/thickness.rs), a `core`-only compressed-sparse-row (CSR) uniform 3D grid over preallocated `cell_offsets: &[usize]` ($M^3 + 1$) and `sorted_vertices: &[[f64; 3]]` ($V$) slices indexed via [`flatten_3d_index`](../../crates/02_engines/math_commons/src/math_kernel/types.rs) with $M = \lceil V^{1/3} \rceil$.
   - Populate `cell_offsets` and `sorted_vertices` in-place via a zero-scratch counting sort (`fill(0)`, cell population count, exclusive prefix sum, vertex scatter, and `copy_within` offset shift), returning `None` if slices lack capacity and falling back to `cell_size = 1.0` for zero-extent or empty bounds.
   - Clamp query cell indices and expanding Chebyshev concentric shell scans to `[0, M - 1]^3` while computing exact IEEE-754 split-plane lower bounds across unsearched grid boundaries so `FlatSpatialGrid::shortest_distance` preserves bit-for-bit Euclidean distance equivalence in expected $O(1)$ time (returning `f64::INFINITY` on empty surfaces and `f64::NAN` on `NaN` queries).
   - Introduce [`ThicknessWorkspace`](../../crates/03_domains/domain_applied/src/applied/freesurfer/thickness.rs) (`new`, `with_max_vertices`, `ensure_capacity`, `build_grids`) preallocating reusable CSR slice pairs outside `VerifiedAllocator`-locked frames.
2. **Zero-Allocation Kernels vs. Allocating Wrapper & Facade Re-Exports**:
   - Switch standalone `point_to_surface_distance` and `cortical_thickness` to a direct zero-allocation scan over `surface.vertices` while retaining exact public signatures.
   - Add `#[verified_engine::verified]` [`cortical_thickness_indexed`](../../crates/03_domains/domain_applied/src/applied/freesurfer/thickness.rs) (single-vertex expected $O(1)$ query over prebuilt `FlatSpatialGrid`s) and [`cortical_thickness_map_into`](../../crates/03_domains/domain_applied/src/applied/freesurfer/thickness.rs) (in-place full-mesh map returning `0` on empty surfaces or insufficient capacity), delegating `cortical_thickness_map` to `ThicknessWorkspace` + `cortical_thickness_map_into`.
   - Re-export `FlatSpatialGrid`, `ThicknessWorkspace`, `cortical_thickness_indexed`, and `cortical_thickness_map_into` from [`freesurfer/mod.rs`](../../crates/03_domains/domain_applied/src/applied/freesurfer/mod.rs) (and `math_explorer::applied::freesurfer`) and regenerate [`crates/unified_verification/api_baseline.json`](../../apps/unified_verification/api_baseline.json).
3. **Interactive Studio Buffer Reuse (`math_explorer_gui`)**:
   - Cache `ThicknessWorkspace` inside `CorticalPhantomCache` in [`phantom.rs`](../../apps/math_explorer_gui/src/tabs/neuroscience/cortical_morphometry/phantom.rs) and evaluate cortical thickness maps in-place via `cortical_thickness_map_into` into `cache.thickness_mm`.

## Consequences
- **Positive**: `cortical_thickness`, `ThicknessWorkspace::build_grids`, `cortical_thickness_indexed`, and `cortical_thickness_map_into` execute with zero dynamic heap allocations inside `VerifiedEngine::run_verified`, verified in [`crates/domain_applied/tests/freesurfer.rs`](../../crates/03_domains/domain_applied/tests/freesurfer.rs).
- **Positive**: Preserves existing `cortical_thickness` and `cortical_thickness_map` signatures and exact Euclidean distances while eliminating per-step grid and vector allocations in `CorticalMorphometryTool`.
