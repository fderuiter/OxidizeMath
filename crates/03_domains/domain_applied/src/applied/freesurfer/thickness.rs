//! Re-exports of CSR spatial uniform grids (`FlatSpatialGrid`, `SpatialGrid`),
//! `ThicknessWorkspace`, and cortical thickness kernels from [`pure_math`].
// [cite:freesurfer]

pub use pure_math::pure_math::differential_geometry::thickness::{
    FlatSpatialGrid, SpatialGrid, ThicknessWorkspace, cortical_thickness, cortical_thickness_batch,
    cortical_thickness_batch_into, cortical_thickness_indexed, cortical_thickness_into,
    cortical_thickness_map, cortical_thickness_map_into, cortical_thickness_mesh,
    cortical_thickness_with_grid,
};
