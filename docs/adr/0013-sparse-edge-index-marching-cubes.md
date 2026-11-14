# Architecture Decision Record 0013: Sparse Edge Index Table for Marching Cubes

## Context
During Marching Cubes isosurface extraction (`crates/domain_applied/src/applied/isosurface/marching_cubes.rs`), the extractor pre-allocated a dense flat vector `edge_to_vertex` sized proportionally to `total_voxels * 3` (`vec![u32::MAX; max_vertices]`). For a typical 100^3 grid (1M voxels), this dense table consumed over 12 MB of heap space while surface voxels often occupied under 1% of the volume. For empty volumes or large sparse fields, pre-allocating the dense table incurred unnecessary memory overhead and high allocation pressure.

## Decision
We replaced the dense edge cache with a sparse lookup table:
1. Replaced the dense `edge_to_vertex` vector with a `HashMap<(usize, usize, usize, u8), u32>` to store vertex indices only for active edges intersected by the isosurface.
2. Refactored edge key mapping via `get_edge_key(coords, edge_num) -> (usize, usize, usize, u8)` using canonical origin coordinates `(x, y, z)` and axis orientation (`0` for X, `1` for Y, `2` for Z) across adjacent cubes.
3. Updated `CubeData` to carry cube coordinates `(x, y, z)` instead of linear array base indices and strides.
4. Dynamically allocated mesh vertices, normals, and indices rather than pre-allocating maximal voxel bounds, guaranteeing zero allocation overhead for empty volumes.
5. Added mesh equivalence and memory scaling tests in `crates/domain_applied/tests/bench_isosurface.rs` verifying vertex counts match baseline across varying grid sizes.

## Consequences
- **Positive:** Heap memory usage during isosurface extraction now scales proportionally to the surface area rather than total volume.
- **Positive:** Empty volumes return valid empty meshes without massive pre-allocations.
- **Positive:** Public APIs (`extract`, `extract_isosurface`) and generated mesh geometry remain 100% equivalent to baseline.
