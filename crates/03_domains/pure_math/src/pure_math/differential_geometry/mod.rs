pub mod buckled_flow;
/// Triangulated three-dimensional surface mesh representation (`Surface`).
pub mod cortical_mesh;
pub mod cotangent_mesh;
/// Submodule providing `heat_equation` types, traits, and algorithms.
pub mod heat_equation;
/// Intrinsic hyperbolic geometry on the Poincaré and Beltrami-Klein disks via $\mathrm{SU}(1,1)$.
pub mod hyperbolic;
/// Submodule providing `mean_curvature_flow` types, traits, and algorithms.
pub mod mean_curvature_flow;
/// Submodule providing `operators` types, traits, and algorithms.
pub mod operators;
/// Submodule providing `surface` types, traits, and algorithms.
pub mod surface;
/// Zero-allocation CSR spatial grids and symmetric cortical thickness kernels.
pub mod thickness;

pub use cortical_mesh::{Surface, mean_vertex_height};
pub use cotangent_mesh::CotangentLaplacian;
pub use thickness::{
    FlatSpatialGrid, SpatialGrid, ThicknessWorkspace, cortical_thickness, cortical_thickness_batch,
    cortical_thickness_batch_into, cortical_thickness_indexed, cortical_thickness_into,
    cortical_thickness_map, cortical_thickness_map_into, cortical_thickness_mesh,
    cortical_thickness_with_grid,
};

// [cite:modular_polynomials_review]
