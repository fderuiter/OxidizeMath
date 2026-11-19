# ADR 0025: Rolling Two-Slice Edge Cache for Marching Cubes

## Status
Accepted (supersedes the global `HashMap` edge table in [ADR 0013](0013-sparse-edge-index-marching-cubes.md))

## Context
[ADR 0013](0013-sparse-edge-index-marching-cubes.md) replaced a dense full-volume `vec![u32::MAX; total_voxels * 3]` edge table in `crates/domain_applied/src/applied/isosurface/marching_cubes.rs` with `HashMap<(usize, usize, usize, u8), u32>`. While this avoided upfront `O(width * height * depth)` allocations on empty or sparse volumes, `MarchingCubes::extract` visits voxel cubes in deterministic `z -> y -> x` order while retaining every historical surface edge across all previously visited `z` slices. Hashing and rehashing 32-byte `(usize, usize, usize, u8)` keys accounted for roughly 28%–46% of Marching Cubes extraction latency and ~59% of peak heap allocations on `100^3`–`200^3` sphere volumes.

Because traversal proceeds strictly along `z`, then `y`, then `x`, cube `(x, y, z)` can only share intersected edges with:
- the active XY plane `z` (edges `0..=3`, storing X-directed and Y-directed edge slots),
- the next XY plane `z + 1` (edges `4..=7`),
- the active Z-directed row `(y, z)` (edges `8..=9`), and
- the next Z-directed row `(y + 1, z)` (edges `10..=11`).

## Decision
1. **Private Rolling Slice Cache (`SliceEdgeCache`)**: Introduce `crates/domain_applied/src/applied/isosurface/edge_cache.rs` as a private submodule (`mod edge_cache;`) in `crates/domain_applied/src/applied/isosurface/mod.rs`, keeping `marching_cubes.rs` well under the 500-line ceiling and introducing zero public API drift.
2. **`O(1)` Direct Slot Indexing**: Store two XY plane buffers (`curr_xy`, `next_xy`) of length `width * height * 2` and two Z-directed row buffers (`curr_z_row`, `next_z_row`) of length `width`, initialized with `u32::MAX` (`EMPTY_EDGE_VERTEX`). Map cube edges `0..=3`, `4..=7`, `8..=9`, and `10..=11` directly to their canonical buffer slots via `SliceEdgeCache::slot_mut`.
3. **Lazy Allocation for Zero-Cost Empty Grids**: Initialize all four `Vec<u32>` buffers empty (`Vec::new()`), deferring heap allocation until `CUBE_EDGE_FLAGS[index] != 0` accesses a specific buffer. Completely empty volumes perform zero heap allocations.
4. **Dirty-Flag Slice Rotation**: Rotate `(curr_xy, next_xy)` and reset Z-rows on `z > 0` (`advance_z`), and rotate `(curr_z_row, next_z_row)` on `y > 0` (`advance_y`), tracking per-buffer dirty flags so empty `z` slices and `y` rows skip `fill(u32::MAX)` passes once cleared.
5. **Bit-Exact Mesh Equivalence & API Stability**: Preserve `MarchingCubes::new`, `MarchingCubes::extract`, and `extract_isosurface` signatures and edge visitation order (`0..12` across `z -> y -> x`), guaranteeing bit-identical `vertices`, `normals`, and `indices`.

## Consequences
- **Positive**: Eliminates `HashMap` hashing and rehashing in `MarchingCubes::triangulate_cube`, reducing memory footprint from `O(surface_area)` hash buckets to `O(width * height)` rolling slices (e.g., ~160 KB total on a `100^3` grid) while substantially accelerating surface extraction.
- **Positive**: Retains zero heap allocations on empty grids and skips redundant buffer clears across empty exterior slices.
- **Negative / Trade-offs**: Very wide 2D-heavy grids with only a handful of active cubes allocate up to `width * height * 2 * 4` bytes per touched XY plane instead of a tiny hash table, though this remains bounded by two 2D slices rather than the 3D volume.
