/// Wavefront OBJ mesh exporter and shared geometry validation helpers.
pub mod export;
/// Stanford PLY ASCII/binary mesh exporter.
pub mod export_ply;
/// 3D Gaussian splat PLY exporter.
pub mod export_splat_ply;
/// Binary and ASCII stereolithography (STL) mesh exporters.
pub mod export_stl;
/// Parametric 3D mesh primitive generators.
pub mod primitives;
/// Core 3D point, indexed triangle mesh, and vertex-color types.
pub mod types;

pub use export::*;
pub use export_ply::*;
pub use export_splat_ply::*;
pub use export_stl::*;
pub use primitives::*;
pub use types::*;

#[cfg(test)]
mod mesh_tests;
