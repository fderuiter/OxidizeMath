# ADR 0027: Delete Dead Grid2D and Triangle Types

## Status
Accepted (extends [ADR 0016](0016-delete-unreferenced-boundary-conditions.md), [ADR 0016](0016-purge-dead-code-suppressions.md), and [ADR 0019](0019-purge-dead-code-bypasses.md))

## Context
Earlier migrations replaced `oxidize_core::grid::Grid2D` and `oxidize_core::mesh::Triangle` across downstream solvers and Marching Cubes isosurface extraction, and removed `oxidize_core::double_buffer`. However, `crates/oxidize_core` continued to export two unreferenced types along with stale documentation and verification exemptions:
1. **`Grid2D` (`crates/oxidize_core/src/grid.rs`)**: Defined `pub struct Grid2D<T>` with 8 `#[allow(missing_docs)]` attributes, exported via `#[allow(missing_docs)] pub mod grid;` in `crates/oxidize_core/src/lib.rs`, and exempted by a `"grid.rs"` skip rule in `crates/unified_verification/src/ast_visitor.rs`, despite having zero production or test callers outside a stale `crates/oxidize_core/README.md` doctest.
2. **`Triangle` (`crates/oxidize_core/src/mesh/types.rs`)**: Retained `pub struct Triangle` with 7 `#[allow(missing_docs)]` attributes and downstream re-exports in `crates/domain_applied/src/applied/isosurface/types.rs` and `crates/domain_applied/src/applied/isosurface/mod.rs`, even though `Mesh` exclusively uses indexed `Point3D` vertex and normal buffers.

## Decision
1. **Delete `Grid2D` Module**: Remove `crates/oxidize_core/src/grid.rs` and its `pub mod grid` declaration in `crates/oxidize_core/src/lib.rs`.
2. **Delete `Triangle` Struct and Stale Re-exports**: Remove `pub struct Triangle` from `crates/oxidize_core/src/mesh/types.rs` and drop `Triangle` from `pub use` lists in `crates/domain_applied/src/applied/isosurface/types.rs` and `crates/domain_applied/src/applied/isosurface/mod.rs`, while preserving `Point3D`, `Mesh`, and `export_mesh_to_obj_string`.
3. **Update `oxidize_core` Documentation and Clone Detector**: Remove stale `grid` and `double_buffer` prose from `crates/oxidize_core/README.md`, replace the `Grid2D` doctest with a runnable `OxidizeRng` and indexed `Mesh` example, and remove the obsolete `"grid.rs"` skip condition from `run_clone_detector` in `crates/unified_verification/src/ast_visitor.rs`.
4. **Synchronize Public API Baseline**: Regenerate `crates/unified_verification/api_baseline.json` via `regenerate-baseline` to prune the deleted `Grid2D` and `Triangle` signatures.

## Consequences
- **Positive**:
  - Eliminates dead `Grid2D` and `Triangle` types and stale downstream re-exports across `oxidize_core`, `domain_applied`, and `math_explorer`.
  - Removes 16 `#[allow(missing_docs)]` suppressions in `oxidize_core` and removes the `"grid.rs"` clone-detection exemption in `unified_verification`.
  - Aligns `crates/oxidize_core/README.md` and `crates/unified_verification/api_baseline.json` with active runtime primitives.
- **Negative**:
  - External consumers importing `oxidize_core::grid::Grid2D` or `Triangle` directly must use indexed `Mesh` buffers or domain-specific storage, though zero workspace callers existed.
