# ADR 20261123: Reusable Flat CSR Spatial Grid for Cortical Thickness

## Status

Accepted (complements [ADR 0018](0018-uniform-spatial-grid-cortical-thickness.md), [ADR 20261123](20261123-freesurfer-deformable-mrf-and-cortical-morphometry.md), and [ADR 20261123](20261123-csr-spatial-grid-thickness-workspace.md))

## Context

[ADR 0018](0018-uniform-spatial-grid-cortical-thickness.md) introduced an internal `SpatialGrid` in `crates/domain_applied/src/applied/freesurfer/thickness.rs` to accelerate point-to-surface Euclidean distance queries using Chebyshev shell expansion, and [ADR 20261123](20261123-csr-spatial-grid-thickness-workspace.md) replaced `HashMap` buckets with slice-borrowed `FlatSpatialGrid<'a>` and `ThicknessWorkspace` for mutable buffer reuse across deformable frames. Static cortical surface callers also benefit from an ergonomic self-contained owned `SpatialGrid` index and slice-clamped batch helpers without managing external CSR slice lifetimes.

## Decision

We unified owned `SpatialGrid` and pre-indexed batch cortical thickness helpers on top of `FlatSpatialGrid<'a>` in `crates/domain_applied/src/applied/freesurfer/thickness.rs` and `crates/domain_applied/src/applied/freesurfer/mod.rs` without duplicating CSR counting-sort or Chebyshev shell traversal:

1. **Owned CSR `SpatialGrid` Delegating to `FlatSpatialGrid<'_>`**:
   - Exposed `pub struct SpatialGrid` owning contiguous `cell_offsets: Vec<usize>` ($M^3 + 1$) and `sorted_vertices: Vec<[f64; 3]>` ($V$) buffers populated once via `FlatSpatialGrid::from_surface`.
   - Added `#[verified_engine::verified] SpatialGrid::as_flat_grid(&self) -> FlatSpatialGrid<'_>` and `#[verified_engine::verified] SpatialGrid::shortest_distance(&self, point: &[f64; 3]) -> f64`, sharing the exact IEEE-754 split-plane Chebyshev shell traversal (plus linear scan fallback on non-finite `±INFINITY` query coordinates) with zero dynamic heap allocations during queries.
2. **Pre-Indexed Single-Vertex and Slice-Clamped Batch APIs**:
   - Added `#[verified_engine::verified] pub fn cortical_thickness_with_grid` delegating to `cortical_thickness_indexed` via `SpatialGrid::as_flat_grid`.
   - Added `#[verified_engine::verified] pub fn cortical_thickness_into` evaluating `white_surface.vertices.len().min(pial_surface.vertices.len()).min(out.len())` vertex pairs into caller-supplied `out: &mut [f64]` and zeroing trailing elements `out[count..]` without panicking on mismatched lengths.
   - Added `pub fn cortical_thickness_mesh` delegating to `cortical_thickness_map` and re-exported `SpatialGrid`, `cortical_thickness_with_grid`, `cortical_thickness_into`, and `cortical_thickness_mesh` alongside `FlatSpatialGrid`, `ThicknessWorkspace`, `cortical_thickness_indexed`, and `cortical_thickness_map_into`.
3. **Serialized Zero-Allocation & IEEE-754 Parity Verification**:
   - Serialized all `VerifiedEngine::run_verified` blocks in `crates/domain_applied/tests/freesurfer.rs` under `VERIFIED_RUN_MUTEX` (`run_verified_locked`) and verified zero post-initialization allocations (including a subprocess `VerifiedAllocator` abort probe) and exact IEEE-754 parity across random meshes, multi-vertex CSR cells, sparse shells, empty surfaces (`f64::INFINITY`), `NaN`/`±INFINITY` query coordinates, zero-extent surfaces, and mismatched slice lengths.

## Consequences

- Callers with evolving deformable meshes reuse `ThicknessWorkspace` + `FlatSpatialGrid`, while callers with static meshes build owned `SpatialGrid` once via `SpatialGrid::from_surface`—both sharing a single CSR counting-sort and Chebyshev shell implementation.
- Both `cortical_thickness` and pre-indexed `SpatialGrid` / `FlatSpatialGrid` queries execute with zero dynamic heap allocations inside `VerifiedEngine::run_verified` under `VerifiedAllocator`.
