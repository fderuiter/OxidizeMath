#![cfg_attr(any(), verified(opt_out = "infrastructure"))]

use math_explorer::pure_math::differential_geometry::surface::ParametricSurface;
use oxidize_core::mesh::{Mesh, Point3D};

/// Builds a seam- and pole-welded triangle [`Mesh`] for `surface` over its natural `(u, v)` domain.
#[must_use]
pub fn surface_to_mesh(surface: &dyn ParametricSurface, u_res: usize, v_res: usize) -> Mesh {
    surface_to_mesh_scaled(surface, u_res, v_res, 1.0)
}

/// Builds a seam- and pole-welded triangle [`Mesh`] scaled by `scale_mm` physical millimeters.
#[must_use]
pub fn surface_to_mesh_scaled(
    surface: &dyn ParametricSurface,
    u_res: usize,
    v_res: usize,
    scale_mm: f32,
) -> Mesh {
    let u_steps = u_res.max(3);
    let v_steps = v_res.max(3);
    if surface.has_v_poles() {
        build_polar_surface_mesh(surface, u_steps, v_steps, scale_mm)
    } else {
        build_grid_surface_mesh(surface, u_steps, v_steps, scale_mm)
    }
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
    u_steps: usize,
    v_steps: usize,
    scale_mm: f32,
) -> Mesh {
    let (u_min, u_max) = surface.u_domain();
    let (v_min, v_max) = surface.v_domain();
    let u_periodic = surface.is_u_periodic();
    let u_cols = if u_periodic { u_steps } else { u_steps + 1 };
    let ring_count = v_steps - 1;
    let total_verts = 2 + ring_count * u_cols;

    let mut vertices = Vec::with_capacity(total_verts);
    let mut normals = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(u_steps * (2 + ring_count.saturating_sub(1) * 2) * 3);

    // North pole (index 0)
    push_surface_sample(surface, u_min, v_min, scale_mm, &mut vertices, &mut normals);

    // Interior latitude rings j = 1..v_steps
    for j in 1..v_steps {
        let v = v_min + (v_max - v_min) * (j as f64 / v_steps as f64);
        for i in 0..u_cols {
            let u = u_min + (u_max - u_min) * (i as f64 / u_steps as f64);
            push_surface_sample(surface, u, v, scale_mm, &mut vertices, &mut normals);
        }
    }

    // South pole (index south_idx)
    let south_idx = vertices.len();
    push_surface_sample(surface, u_min, v_max, scale_mm, &mut vertices, &mut normals);

    let ring_idx = |ring: usize, col: usize| -> usize { 1 + ring * u_cols + col };

    for i in 0..u_steps {
        let inext = if u_periodic { (i + 1) % u_steps } else { i + 1 };
        // North polar fan (outward CCW)
        indices.extend_from_slice(&[0, ring_idx(0, i), ring_idx(0, inext)]);

        // Interior quad strips
        for r in 0..ring_count.saturating_sub(1) {
            let r00 = ring_idx(r, i);
            let r10 = ring_idx(r, inext);
            let r01 = ring_idx(r + 1, i);
            let r11 = ring_idx(r + 1, inext);
            indices.extend_from_slice(&[r00, r01, r11, r00, r11, r10]);
        }

        // South polar fan (outward CCW)
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
    u_steps: usize,
    v_steps: usize,
    scale_mm: f32,
) -> Mesh {
    let (u_min, u_max) = surface.u_domain();
    let (v_min, v_max) = surface.v_domain();
    let u_periodic = surface.is_u_periodic();
    let v_periodic = surface.is_v_periodic();
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

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use math_commons::ast::{CompiledExpr, parse};
    use math_explorer::pure_math::differential_geometry::surface::{
        CompiledHeightfieldSurface, KleinBottle, Sphere, SurfaceAnalysis, Torus,
    };
    use oxidize_core::mesh::analyze_mesh;
    use std::f64::consts::PI;

    #[test]
    fn test_sphere_30x30_welded_watertight_zero_nan_normals_and_area_within_2_percent() {
        let radius = 2.5_f64;
        let sphere = Sphere { radius };
        // Verify pole normals in SurfaceAnalysis never produce 0.0/0.0 NaNs
        let north_n = sphere.unit_normal(0.0, 0.0);
        let south_n = sphere.unit_normal(1.2, PI);
        assert!(north_n.iter().all(|c| c.is_finite()));
        assert!(south_n.iter().all(|c| c.is_finite()));

        let mesh = surface_to_mesh(&sphere, 30, 30);
        let diag = analyze_mesh(&mesh);
        assert!(diag.is_watertight());
        assert_eq!(diag.open_edges, 0);
        assert_eq!(diag.nan_normals, 0);
        assert!(diag.signed_volume_mm3 > 0.0);

        let exact_area = 4.0 * PI * radius * radius;
        let rel_err = (diag.area_mm2 - exact_area).abs() / exact_area;
        assert!(
            rel_err < 0.02,
            "Expected Sphere 30x30 area within 2% of {exact_area}, got {} (rel_err={rel_err})",
            diag.area_mm2
        );
    }

    #[test]
    fn test_torus_klein_bottle_and_custom_heightfield_exports() {
        let torus = Torus {
            major_radius: 2.0,
            minor_radius: 0.5,
        };
        let torus_mesh = surface_to_mesh_scaled(&torus, 30, 30, 10.0);
        let torus_diag = analyze_mesh(&torus_mesh);
        assert!(torus_diag.is_watertight());
        assert_eq!(torus_diag.open_edges, 0);
        assert_eq!(torus_diag.nan_normals, 0);
        assert!(torus_diag.signed_volume_mm3 > 0.0);

        let klein = KleinBottle { radius: 2.0 };
        let klein_mesh = surface_to_mesh_scaled(&klein, 30, 30, 10.0);
        let klein_diag = analyze_mesh(&klein_mesh);
        assert_eq!(klein_diag.nan_normals, 0);

        let saddle_compiled =
            CompiledExpr::compile_with_vars(&parse("0.25 * (x^2 - y^2)").unwrap(), &["x", "y"])
                .unwrap();
        let saddle = CompiledHeightfieldSurface::new(saddle_compiled, &[]);
        let saddle_mesh = surface_to_mesh_scaled(&saddle, 16, 16, 10.0);
        let saddle_diag = analyze_mesh(&saddle_mesh);
        assert_eq!(saddle_diag.nan_normals, 0);
        assert!((saddle_diag.dimensions_mm.x - 40.0).abs() < 1e-3);
        assert!((saddle_diag.dimensions_mm.y - 40.0).abs() < 1e-3);
    }
}
