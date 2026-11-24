#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Front-facing 3D triangle raycast picking and bounded CSR geodesic Gaussian brush kernel.

use super::surface_brush::{BrushPolarity, GeodesicBrushStroke, SurfaceBrushHit};
use crate::camera::Camera3D;
use math_explorer::pure_math::differential_geometry::CotangentLaplacian;

/// Picks the frontmost front-facing vertex hit by a 2D projected plot ray `pointer_xy`.
#[must_use]
pub fn pick_front_facing_vertex(
    camera: &Camera3D,
    positions: &[[f64; 3]],
    triangles: &[[usize; 3]],
    normals: &[[f64; 3]],
    pointer_xy: [f64; 2],
    two_sided: bool,
    fallback_tol: f64,
) -> Option<SurfaceBrushHit> {
    if positions.is_empty() {
        return None;
    }
    let projected: Vec<([f64; 2], f64)> = positions
        .iter()
        .map(|p| camera.project_with_depth(p))
        .collect();

    let mut best_tri_hit: Option<SurfaceBrushHit> = None;
    for &tri in triangles {
        if let Some(hit) = intersect_projected_triangle(&projected, tri, pointer_xy, two_sided) {
            if best_tri_hit.is_none_or(|prev| hit.viewer_z > prev.viewer_z) {
                best_tri_hit = Some(hit);
            }
        }
    }
    best_tri_hit.or_else(|| {
        pick_silhouette_fallback(
            camera,
            &projected,
            normals,
            pointer_xy,
            two_sided,
            fallback_tol,
        )
    })
}

fn intersect_projected_triangle(
    projected: &[([f64; 2], f64)],
    [i0, i1, i2]: [usize; 3],
    pointer_xy: [f64; 2],
    two_sided: bool,
) -> Option<SurfaceBrushHit> {
    let (&(p0, z0), &(p1, z1), &(p2, z2)) =
        (projected.get(i0)?, projected.get(i1)?, projected.get(i2)?);
    let area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
    if (two_sided && area2.abs() <= 1e-10) || (!two_sided && area2 <= 1e-10) {
        return None;
    }
    let inv_area = 1.0 / area2;
    let b0 = ((p1[0] - pointer_xy[0]) * (p2[1] - pointer_xy[1])
        - (p1[1] - pointer_xy[1]) * (p2[0] - pointer_xy[0]))
        * inv_area;
    let b1 = ((p2[0] - pointer_xy[0]) * (p0[1] - pointer_xy[1])
        - (p2[1] - pointer_xy[1]) * (p0[0] - pointer_xy[0]))
        * inv_area;
    let b2 = 1.0 - b0 - b1;
    if b0 < -1e-4 || b1 < -1e-4 || b2 < -1e-4 {
        return None;
    }
    let picked = if b0 >= b1 && b0 >= b2 {
        i0
    } else if b1 >= b2 {
        i1
    } else {
        i2
    };
    let pv = projected[picked].0;
    Some(SurfaceBrushHit {
        vertex: picked,
        distance_2d: ((pv[0] - pointer_xy[0]).powi(2) + (pv[1] - pointer_xy[1]).powi(2)).sqrt(),
        viewer_z: b0 * z0 + b1 * z1 + b2 * z2,
    })
}

fn pick_silhouette_fallback(
    camera: &Camera3D,
    projected: &[([f64; 2], f64)],
    normals: &[[f64; 3]],
    pointer_xy: [f64; 2],
    two_sided: bool,
    fallback_tol: f64,
) -> Option<SurfaceBrushHit> {
    let mut best: Option<SurfaceBrushHit> = None;
    for (idx, &(pv, vz)) in projected.iter().enumerate() {
        if !two_sided {
            if let Some(n) = normals.get(idx) {
                let n_cam = camera.rotate_to_camera_space(n);
                if n_cam[2] < -0.05 {
                    continue;
                }
            }
        }
        let dist = ((pv[0] - pointer_xy[0]).powi(2) + (pv[1] - pointer_xy[1]).powi(2)).sqrt();
        if dist <= fallback_tol {
            let better = match best {
                None => true,
                Some(prev) => {
                    dist < prev.distance_2d - 1e-6
                        || ((dist - prev.distance_2d).abs() <= 1e-6 && vz > prev.viewer_z)
                }
            };
            if better {
                best = Some(SurfaceBrushHit {
                    vertex: idx,
                    distance_2d: dist,
                    viewer_z: vz,
                });
            }
        }
    }
    best
}

/// Applies a bounded 4-hop CSR geodesic Gaussian brush kernel centered at `stroke.center_vertex`.
pub fn apply_geodesic_kernel(
    laplacian: &CotangentLaplacian,
    positions: &[[f64; 3]],
    stroke: GeodesicBrushStroke,
    mut visit_fn: impl FnMut(usize, f64),
) {
    let n = laplacian.num_vertices().min(positions.len());
    if stroke.center_vertex >= n {
        return;
    }
    let center_pos = positions[stroke.center_vertex];
    let sigma = stroke.radius.max(0.05);
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
    let sign = match stroke.polarity {
        BrushPolarity::Excite => 1.0,
        BrushPolarity::Inhibit => -1.0,
    };

    let mut visited = vec![u8::MAX; n];
    let mut frontier = Vec::with_capacity(64);
    visited[stroke.center_vertex] = 0;
    frontier.push(stroke.center_vertex);

    let mut head = 0_usize;
    while head < frontier.len() {
        let v = frontier[head];
        head += 1;
        let hop = visited[v];
        let p = positions[v];
        let d_sq = (p[0] - center_pos[0]).powi(2)
            + (p[1] - center_pos[1]).powi(2)
            + (p[2] - center_pos[2]).powi(2);
        let weight = (-d_sq * inv_two_sigma_sq).exp();
        if weight >= 0.02 {
            visit_fn(v, sign * stroke.strength * weight);
        }
        if hop < 4 && d_sq <= (2.5 * sigma).powi(2) {
            for &nb in laplacian.neighbors(v) {
                if nb < n && visited[nb] == u8::MAX {
                    visited[nb] = hop + 1;
                    frontier.push(nb);
                }
            }
        }
    }
}
