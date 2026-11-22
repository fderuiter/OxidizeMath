#![cfg_attr(any(), verified(opt_out = "infrastructure"))]

use math_explorer::pure_math::differential_geometry::surface::ParametricSurface;
use nalgebra::Vector3;
use oxidize_core::mesh::{Mesh, Point3D};

pub fn surface_to_mesh_with_domain(
    surface: &dyn ParametricSurface,
    u_range: (f64, f64),
    v_range: (f64, f64),
    u_res: usize,
    v_res: usize,
) -> Mesh {
    let num_vertices = (u_res + 1) * (v_res + 1);
    let num_triangles = u_res * v_res * 2;
    let mut vertices = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut indices = Vec::with_capacity(num_triangles * 3);

    let (u_min, u_max) = u_range;
    let (v_min, v_max) = v_range;

    for i in 0..=u_res {
        for j in 0..=v_res {
            let u = u_min + (u_max - u_min) * (i as f64 / u_res as f64);
            let v = v_min + (v_max - v_min) * (j as f64 / v_res as f64);
            let p = surface.position(u, v);

            let ru = surface.partial_u(u, v);
            let rv = surface.partial_v(u, v);
            let cross = ru.cross(&rv);
            let n = if cross.norm_squared() > 1e-24 && cross.iter().all(|c| c.is_finite()) {
                cross.normalize()
            } else {
                Vector3::new(0.0, 0.0, 1.0)
            };

            vertices.push(Point3D::new(p.x as f32, p.y as f32, p.z as f32));
            normals.push(Point3D::new(n.x as f32, n.y as f32, n.z as f32));
        }
    }

    for i in 0..u_res {
        for j in 0..v_res {
            let stride = v_res + 1;
            let row0 = i * stride;
            let row1 = (i + 1) * stride;
            let idx00 = row0 + j;
            let idx10 = row1 + j;
            let idx01 = row0 + j + 1;
            let idx11 = row1 + j + 1;

            indices.push(idx00);
            indices.push(idx10);
            indices.push(idx01);

            indices.push(idx10);
            indices.push(idx11);
            indices.push(idx01);
        }
    }

    Mesh {
        vertices,
        normals,
        indices,
    }
}
