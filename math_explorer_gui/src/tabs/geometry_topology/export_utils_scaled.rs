#![cfg_attr(any(), verified(opt_out = "infrastructure"))]

use math_explorer::pure_math::differential_geometry::surface::ParametricSurface;
use oxidize_core::mesh::{Mesh, Point3D};

/// Builds a seam- and pole-welded triangle [`Mesh`] for `surface` over its natural `(u, v)` domain.
#[cfg(test)]
#[must_use]
pub fn surface_to_mesh_natural(surface: &dyn ParametricSurface, u_res: usize, v_res: usize) -> Mesh {
    surface_to_mesh_scaled(surface, u_res, v_res, 1.0)
}

/// Builds a seam- and pole-welded triangle [`Mesh`] scaled by `scale_mm` physical millimeters.
#[cfg(test)]
#[must_use]
pub fn surface_to_mesh_scaled(
    surface: &dyn ParametricSurface,
    u_res: usize,
    v_res: usize,
    scale_mm: f32,
) -> Mesh {
    surface_to_mesh_with_domain_scaled(
        surface,
        surface.u_domain(),
        surface.v_domain(),
        u_res,
        v_res,
        scale_mm,
    )
}

/// Builds a seam- and pole-welded triangle [`Mesh`] over explicit `(u_range, v_range)` domains,
/// scaled by `scale_mm` physical millimeters.
#[must_use]
pub fn surface_to_mesh_with_domain_scaled(
    surface: &dyn ParametricSurface,
    u_range: (f64, f64),
    v_range: (f64, f64),
    u_res: usize,
    v_res: usize,
    scale_mm: f32,
) -> Mesh {
    let u_steps = u_res.max(3);
    let v_steps = v_res.max(3);
    let natural_u = same_interval(u_range, surface.u_domain());
    let natural_v = same_interval(v_range, surface.v_domain());
    if surface.has_v_poles() && natural_u && natural_v {
        build_polar_surface_mesh(surface, u_range, v_range, u_steps, v_steps, scale_mm)
    } else {
        build_grid_surface_mesh(
            surface,
            u_range,
            v_range,
            (
                natural_u && surface.is_u_periodic(),
                natural_v && surface.is_v_periodic(),
            ),
            u_steps,
            v_steps,
            scale_mm,
        )
    }
}

fn same_interval(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
}

fn push_surface_sample(
    surface: &dyn ParametricSurface,
    u: f64,
    v: f64,
    scale_mm: f32,
    vertices: &mut Vec<Point3D>,
    normals: &mut Vec<Point3D>,
) {
    let p = surface.position(u, v);
    let n = surface.mesh_normal(u, v);
    vertices.push(Point3D::new(
        p.x as f32 * scale_mm,
        p.y as f32 * scale_mm,
        p.z as f32 * scale_mm,
    ));
    normals.push(Point3D::new(n.x as f32, n.y as f32, n.z as f32));
}

fn build_polar_surface_mesh(
    surface: &dyn ParametricSurface,
    u_range: (f64, f64),
    v_range: (f64, f64),
    u_steps: usize,
    v_steps: usize,
    scale_mm: f32,
) -> Mesh {
    let (u_min, u_max) = u_range;
    let (v_min, v_max) = v_range;
    let u_periodic = surface.is_u_periodic();
    let u_cols = if u_periodic { u_steps } else { u_steps + 1 };
    let ring_count = v_steps - 1;
    let total_verts = 2 + ring_count * u_cols;

    let mut vertices = Vec::with_capacity(total_verts);
    let mut normals = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(u_steps * (2 + ring_count.saturating_sub(1) * 2) * 3);

    push_surface_sample(surface, u_min, v_min, scale_mm, &mut vertices, &mut normals);

    for j in 1..v_steps {
        let v = v_min + (v_max - v_min) * (j as f64 / v_steps as f64);
        for i in 0..u_cols {
            let u = u_min + (u_max - u_min) * (i as f64 / u_steps as f64);
            push_surface_sample(surface, u, v, scale_mm, &mut vertices, &mut normals);
        }
    }

    let south_idx = vertices.len();
    push_surface_sample(surface, u_min, v_max, scale_mm, &mut vertices, &mut normals);

    let ring_idx = |ring: usize, col: usize| -> usize { 1 + ring * u_cols + col };

    for i in 0..u_steps {
        let inext = if u_periodic { (i + 1) % u_steps } else { i + 1 };
        indices.extend_from_slice(&[0, ring_idx(0, i), ring_idx(0, inext)]);

        for r in 0..ring_count.saturating_sub(1) {
            let r00 = ring_idx(r, i);
            let r10 = ring_idx(r, inext);
            let r01 = ring_idx(r + 1, i);
            let r11 = ring_idx(r + 1, inext);
            indices.extend_from_slice(&[r00, r01, r11, r00, r11, r10]);
        }

        let last = ring_count - 1;
        indices.extend_from_slice(&[ring_idx(last, i), south_idx, ring_idx(last, inext)]);
    }

    Mesh {
        vertices,
        normals,
        indices,
    }
}

fn build_grid_surface_mesh(
    surface: &dyn ParametricSurface,
    u_range: (f64, f64),
    v_range: (f64, f64),
    periodicity: (bool, bool),
    u_steps: usize,
    v_steps: usize,
    scale_mm: f32,
) -> Mesh {
    let (u_min, u_max) = u_range;
    let (v_min, v_max) = v_range;
    let (u_periodic, v_periodic) = periodicity;
    let u_cols = if u_periodic { u_steps } else { u_steps + 1 };
    let v_rows = if v_periodic { v_steps } else { v_steps + 1 };

    let mut vertices = Vec::with_capacity(u_cols * v_rows);
    let mut normals = Vec::with_capacity(u_cols * v_rows);
    let mut indices = Vec::with_capacity(u_steps * v_steps * 6);

    for i in 0..u_cols {
        let u = u_min + (u_max - u_min) * (i as f64 / u_steps as f64);
        for j in 0..v_rows {
            let v = v_min + (v_max - v_min) * (j as f64 / v_steps as f64);
            push_surface_sample(surface, u, v, scale_mm, &mut vertices, &mut normals);
        }
    }

    for i in 0..u_steps {
        let inext = if u_periodic { (i + 1) % u_steps } else { i + 1 };
        for j in 0..v_steps {
            let jnext = if v_periodic { (j + 1) % v_steps } else { j + 1 };
            let idx00 = i * v_rows + j;
            let idx10 = inext * v_rows + j;
            let idx01 = i * v_rows + jnext;
            let idx11 = inext * v_rows + jnext;
            indices.extend_from_slice(&[idx00, idx10, idx11, idx00, idx11, idx01]);
        }
    }

    Mesh {
        vertices,
        normals,
        indices,
    }
}
