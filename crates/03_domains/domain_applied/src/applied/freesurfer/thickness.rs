//! Backward-compatible re-exports of CSR spatial grids (`FlatSpatialGrid`, `SpatialGrid`),
//! `ThicknessWorkspace`, and symmetric cortical thickness kernels from [`oxidize_core::mesh`].
// [cite:freesurfer]

pub use oxidize_core::mesh::{
    FlatSpatialGrid, SpatialGrid, ThicknessWorkspace, cortical_thickness, cortical_thickness_batch,
    cortical_thickness_batch_into, cortical_thickness_indexed, cortical_thickness_into,
    cortical_thickness_map, cortical_thickness_map_into, cortical_thickness_mesh,
    cortical_thickness_with_grid,
};
