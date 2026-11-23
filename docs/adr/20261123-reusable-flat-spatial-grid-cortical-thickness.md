# ADR 20261123: Reusable Flat CSR Spatial Grid for Cortical Thickness

## Status

Accepted

## Context

[ADR 0018](0018-uniform-spatial-grid-cortical-thickness.md) introduced an internal `SpatialGrid` in `crates/domain_applied/src/applied/freesurfer/thickness.rs` to accelerate point-to-surface Euclidean distance queries using Chebyshev shell expansion. However:

1. **Per-Vertex Grid Reconstruction ($O(V^2)$ Scaling)**: `cortical_thickness` called `point_to_surface_distance` twice per vertex, and each call constructed and dropped a new `SpatialGrid` backed by `HashMap<(i64, i64, i64), Vec<[f64; 3]>>`. Evaluating whole-mesh cortical thickness across $V$ vertices rebuilt `HashMap` and per-cell `Vec` buckets $2V$ times, regressing whole-mesh runtime back to $O(V^2)$ with gigabytes of transient heap churn.
2. **Zero-Allocation Runtime Violation**: Allocating `HashMap` and `Vec` buckets inside `#[verified_engine::verified]` `point_to_surface_distance` violated NASA Power of 10 Rule 3 (`docs/architecture/zero_allocation_runtime.md`), causing `VerifiedEngine::run_verified` to abort under `VerifiedAllocator`.

## Decision

We refactored `crates/domain_applied/src/applied/freesurfer/thickness.rs` and `crates/domain_applied/src/applied/freesurfer/mod.rs` while keeping `Surface { pub vertices: Vec<[f64; 3]> }` in `surface.rs` unchanged:

1. **Flat CSR `SpatialGrid` with Deterministic Open Addressing**:
   - Replaced `std::collections::HashMap` and per-cell `Vec<[f64; 3]>` allocations with a single flat Compressed Sparse Row (CSR) `vertices: Vec<[f64; 3]>` buffer paired with a power-of-two open-addressed `Vec<CellBucket>` lookup table (load factor $\le 0.5$) indexed by a deterministic 64-bit SplitMix integer cell mixer (`hash_cell`).
   - Exposed `SpatialGrid`, `SpatialGrid::from_surface`, and `#[verified_engine::verified] SpatialGrid::shortest_distance` publicly in `thickness` and re-exported `SpatialGrid` from `domain_applied::applied::freesurfer`.
   - Clamped Chebyshev shell traversal to `[min_cell, max_cell]` and restricted interior `(kx, ky)` columns to the two shell faces `cz - r` and `cz + r`, visiting only boundary cells in $O(r^2)$ steps with zero dynamic memory allocations during queries.
2. **Zero-Allocation Single-Vertex and Pre-Indexed Batch APIs**:
   - Updated `point_to_surface_distance` (used by `cortical_thickness`) to scan `surface.vertices` directly via `f64::min` without heap allocation, allowing single-vertex `cortical_thickness` calls to pass `VerifiedEngine::run_verified`.
   - Added `#[verified_engine::verified] pub fn cortical_thickness_with_grid` for zero-allocation single-vertex queries against pre-built `SpatialGrid` indices.
   - Added `#[verified_engine::verified] pub fn cortical_thickness_into` to evaluate `white_surface.vertices.len().min(pial_surface.vertices.len()).min(out.len())` vertex pairs into a caller-supplied `&mut [f64]` slice and zero trailing elements `out[count..]` without panicking on mismatched lengths.
   - Added `pub fn cortical_thickness_mesh` to construct one `SpatialGrid` per surface and evaluate whole-mesh cortical thickness in $O(V)$ expected time.
   - Re-exported `SpatialGrid`, `cortical_thickness_with_grid`, `cortical_thickness_into`, and `cortical_thickness_mesh` in `crates/domain_applied/src/applied/freesurfer/mod.rs` and regenerated `crates/unified_verification/api_baseline.json`.
3. **Zero-Allocation & IEEE-754 Parity Verification**:
   - Installed `VerifiedAllocator` in `crates/domain_applied/tests/freesurfer.rs` and verified zero post-initialization allocations inside `VerifiedEngine::run_verified` across `SpatialGrid::shortest_distance`, `cortical_thickness`, `cortical_thickness_with_grid`, and `cortical_thickness_into`, alongside exact IEEE-754 parity tests covering random meshes, sparse shells, empty surfaces (`f64::INFINITY`), `NaN` query coordinates (`f64::NAN`), zero-extent surfaces (`1.0` fallback cell size), and mismatched buffer lengths.

## Consequences

- Whole-mesh cortical thickness queries (`cortical_thickness_into` / `cortical_thickness_mesh`) build at most one flat CSR `SpatialGrid` per surface and run in $O(V)$ expected time instead of $O(V^2)$.
- Both `cortical_thickness` and pre-indexed `SpatialGrid` queries execute with zero dynamic heap allocations inside `VerifiedEngine::run_verified` under `VerifiedAllocator`.
- Deterministic integer cell hashing eliminates `RandomState` entropy while preserving exact IEEE-754 Euclidean distance parity with exhaustive linear scans.
