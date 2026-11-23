# Architecture Decision Record 0018: Uniform Spatial Grid Partitioning for Cortical Thickness

## Status

Accepted

## Context
In cortical thickness estimation for neuroimaging meshes (`crates/domain_applied/src/applied/freesurfer/thickness.rs`), calculating vertex-to-surface shortest distance previously performed an exhaustive linear scan over all surface vertices. For pairwise mesh comparisons across large cortical surfaces ($V$ vertices per mesh), this led to $O(V^2)$ computational complexity. When processing high-resolution cortical reconstructions, exhaustive pairwise distance calculation incurred excessive execution latency and poor computational scaling.

## Decision
We modernized `point_to_surface_distance` in `crates/domain_applied/src/applied/freesurfer/thickness.rs` by introducing uniform 3D spatial voxel grid partitioning (`SpatialGrid`):
1. **Spatial Grid Partitioning**: Introduced internal struct `SpatialGrid` partitioning surface vertices into cubic voxel buckets indexed by integer 3D grid cell tuples `(i64, i64, i64)`. The grid voxel size is computed adaptively from the bounding box extents and vertex count $V$ ($M = \lceil V^{1/3} \rceil$, $\text{cell\_size} = \max(\text{max\_extent} / M, 10^{-6})$).
2. **Accelerated Nearest-Neighbor Query**: Evaluates candidate vertices in expanding Chebyshev concentric shells ($r = 0, 1, 2, \dots$), evaluating adjacent grid cells prior to outer boundary regions. Dynamically expands the search radius for sparse vertex distributions until candidate vertices are discovered or all populated cells are searched.
3. **Exact Mathematical Termination**: Evaluates the minimum Euclidean distance to outer unsearched boundary faces $d_{\text{boundary}}$ and terminates early when $d_{\min} \le d_{\text{boundary}}$, mathematically guaranteeing exact floating-point distance equivalence with the exhaustive scan.
4. **Preserved Public API and Robustness**: Retained the public API signature of `cortical_thickness(v_w: &[f64; 3], v_p: &[f64; 3], white_surface: &Surface, pial_surface: &Surface) -> f64` without breaking changes or API drift. Handled edge cases such as empty surfaces (returning `f64::INFINITY`), degenerate single-vertex or zero-extent surfaces (fallback cell size `1.0`), and NaN query points.
5. **Comprehensive Verification**: Added unit tests in `thickness.rs` and integration tests in `crates/domain_applied/tests/freesurfer.rs` verifying precision, exact floating-point distance equivalence against linear scans across pseudo-random configurations, and sparse vertex expansion.

## Consequences
- **Positive:** Reduces point-to-surface distance search complexity from $O(V)$ down to $O(1)$ expected time per query, bringing pairwise cortical mesh comparisons from $O(V^2)$ to sub-quadratic complexity.
- **Positive:** Maintains identical public API signatures and zero API drift.
- **Positive:** Guarantees exact floating-point distance results matching baseline Euclidean distance formulas.
