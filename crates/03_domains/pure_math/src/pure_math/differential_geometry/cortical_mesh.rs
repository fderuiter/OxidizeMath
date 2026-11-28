//! Triangulated three-dimensional surface mesh representation (`Surface`) and vertex height helper.
// [cite:pure_math]

use nalgebra::Point3;
pub use oxidize_core::mesh::Surface;

/// Computes the arithmetic mean height $\bar{z} = \frac{1}{N}\sum_{i=1}^N z_i$ across 3D mesh vertices.
///
/// Returns `0.0` for an empty vertex slice (`N == 0`).
#[must_use]
#[verified_engine::verified]
pub fn mean_vertex_height(vertices: &[Point3<f64>]) -> f64 {
    if vertices.is_empty() {
        return 0.0;
    }
    let sum_z: f64 = vertices.iter().map(|p| p.z).sum();
    sum_z / vertices.len() as f64
}
