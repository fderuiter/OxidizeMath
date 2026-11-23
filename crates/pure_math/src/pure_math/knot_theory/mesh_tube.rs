//! Watertight orientable 2-manifold tubular [`Mesh`] generation for 3D knots and links.

// [cite:knot_braid_topology]

use super::KnotError;
use super::curves::ClosedLink3D;
use super::frames::compute_bishop_frames;
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::mesh::{Mesh, Point3D};
use std::f64::consts::TAU;

/// Extrudes every closed component of `link` along its holonomy-corrected Bishop frame into
/// a single watertight, outward-oriented [`Mesh`] suitable for `.stl`, `.ply`, and `.obj` 3D printing.
///
/// Every edge on each torus component is shared by exactly two oppositely oriented triangles,
/// guaranteeing Euler characteristic $\chi = 0$ per component and positive enclosed volume.
///
/// # Errors
/// Returns [`KnotError`] if `tube_radius <= 0`, `radial_segments < 3`, or any curve tangent degenerates.
#[verified_engine::verified]
pub fn build_link_tubular_mesh(
    link: &ClosedLink3D,
    tube_radius: f64,
    radial_segments: usize,
) -> Result<Mesh, KnotError> {
    if !tube_radius.is_finite() || tube_radius <= 0.0 {
        return Err(KnotError::InvalidParameters {
            reason: "tube radius must be finite and strictly positive",
        });
    }
    if radial_segments < 3 {
        return Err(KnotError::InsufficientSamples {
            required: 3,
            actual: radial_segments,
        });
    }

    let total_rings = link.total_samples();
    let total_vertices = total_rings * radial_segments;
    let total_indices = 6 * total_vertices;
    let mut mesh = Mesh::with_capacity(total_vertices, total_indices);

    for comp in &link.components {
        append_component_torus(&mut mesh, comp, tube_radius, radial_segments)?;
    }
    Ok(mesh)
}

#[verified_engine::verified]
fn append_component_torus(
    mesh: &mut Mesh,
    points: &[nalgebra::Point3<f64>],
    tube_radius: f64,
    radial_segments: usize,
) -> Result<(), KnotError> {
    let frames = compute_bishop_frames(points)?;
    let n_rings = points.len();
    let base_vertex = mesh.vertices.len();

    for (ring_idx, (pt, frame)) in points.iter().zip(frames.iter()).enumerate() {
        let _ = ring_idx;
        for j in 0..radial_segments {
            let phi = TAU * (j as f64) / (radial_segments as f64);
            let (sp, cp) = phi.sin_cos();
            let normal = frame.normal_u * cp + frame.normal_v * sp;
            let pos = pt + normal * tube_radius;
            mesh.vertices
                .push(Point3D::new(pos.x as f32, pos.y as f32, pos.z as f32));
            mesh.normals.push(Point3D::new(
                normal.x as f32,
                normal.y as f32,
                normal.z as f32,
            ));
        }
    }

    for i in 0..n_rings {
        let i_next = (i + 1) % n_rings;
        for j in 0..radial_segments {
            let j_next = (j + 1) % radial_segments;
            let v00 = base_vertex + flatten_2d_index(j, i, radial_segments);
            let v01 = base_vertex + flatten_2d_index(j_next, i, radial_segments);
            let v10 = base_vertex + flatten_2d_index(j, i_next, radial_segments);
            let v11 = base_vertex + flatten_2d_index(j_next, i_next, radial_segments);
            // Outward right-handed winding: (T x phi_tangent) = T x (-sin U + cos V) = -(cos U + sin V),
            // so [v00, v01, v11] has (v01 - v00) x (v11 - v00) = phi_tangent x T = +outward normal!
            mesh.indices
                .extend_from_slice(&[v00, v01, v11, v00, v11, v10]);
        }
    }
    Ok(())
}
