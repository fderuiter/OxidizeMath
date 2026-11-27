#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Watertight 3D [`ColoredMesh`] builder for `.ply`, `.obj`, and `.stl` exports from
//! [`super::RiemannianPdeStudio`].

use super::super::custom_surface::CustomSurfaceSandbox;
use super::super::export_utils::{extrude_surface_wall_shell, DEFAULT_WALL_THICKNESS};
use super::mesh_gen::ManifoldGeometry;
use super::state::{ManifoldKind, RiemannianStudioConfig};
use math_commons::math_kernel::colormap::mapped_gradient;
use math_explorer::pure_math::differential_geometry::surface::KleinBottle;
use oxidize_core::mesh::{ColoredMesh, Mesh, Point3D, VertexColor};

/// Maps a finite vertex scalar field onto RGB [`VertexColor`] entries.
#[must_use]
pub fn map_scalar_field_to_colors(values: &[f64]) -> Vec<VertexColor> {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        min_v = 0.0;
        max_v = 1.0;
    }
    let span = max_v - min_v;
    values
        .iter()
        .map(|&v| {
            let rgb = if !v.is_finite() || span.abs() < 1e-10 {
                mapped_gradient(0.5, 0.0, 1.0)
            } else {
                mapped_gradient(v, min_v, max_v)
            };
            VertexColor::new(rgb[0], rgb[1], rgb[2])
        })
        .collect()
}

/// Builds a strictly closed, outward-oriented, watertight [`ColoredMesh`] for any studio manifold.
pub fn build_studio_colored_mesh(
    geom: &ManifoldGeometry,
    cfg: &RiemannianStudioConfig,
    _sandbox: &CustomSurfaceSandbox,
    scalar_field: &[f64],
) -> Result<ColoredMesh, String> {
    if geom.positions.is_empty() || geom.triangles.is_empty() {
        return Err("Manifold geometry is empty".to_string());
    }
    if scalar_field.len() != geom.positions.len() {
        return Err("Scalar field length does not match manifold vertex count".to_string());
    }

    match cfg.manifold {
        ManifoldKind::Sphere | ManifoldKind::FoldedCorticalShell | ManifoldKind::Torus => {
            build_closed_manifold_colored_mesh(geom, scalar_field)
        }
        ManifoldKind::KleinBottle => build_klein_shell_colored_mesh(geom, cfg, scalar_field),
        ManifoldKind::CustomHeightfield | ManifoldKind::CustomParametric => {
            build_extruded_patch_colored_mesh(geom, scalar_field)
        }
    }
}

fn build_closed_manifold_colored_mesh(
    geom: &ManifoldGeometry,
    scalar_field: &[f64],
) -> Result<ColoredMesh, String> {
    let mut vertices = Vec::with_capacity(geom.positions.len());
    let mut normals = Vec::with_capacity(geom.normals.len());
    for (p, n) in geom.positions.iter().zip(geom.normals.iter()) {
        if !p.iter().all(|c| c.is_finite()) || !n.iter().all(|c| c.is_finite()) {
            return Err("Manifold contains non-finite vertex position or normal".to_string());
        }
        vertices.push(Point3D::new(p[0] as f32, p[1] as f32, p[2] as f32));
        normals.push(Point3D::new(n[0] as f32, n[1] as f32, n[2] as f32));
    }
    let mut indices = Vec::with_capacity(geom.triangles.len() * 3);
    for &[i0, i1, i2] in &geom.triangles {
        indices.extend_from_slice(&[i0, i1, i2]);
    }
    let mut mesh = Mesh {
        vertices,
        normals,
        indices,
    };
    ensure_positive_signed_volume(&mut mesh);
    Ok(ColoredMesh::new(
        mesh,
        map_scalar_field_to_colors(scalar_field),
    ))
}

fn build_klein_shell_colored_mesh(
    geom: &ManifoldGeometry,
    cfg: &RiemannianStudioConfig,
    scalar_field: &[f64],
) -> Result<ColoredMesh, String> {
    let nu = cfg.u_res.clamp(8, 62);
    let raw_nv = cfg.v_res.clamp(8, 62);
    let even_nv = if raw_nv.is_multiple_of(2) {
        raw_nv
    } else {
        raw_nv + 1
    };
    let klein = KleinBottle { radius: 1.35 };
    let mut mesh = extrude_surface_wall_shell(&klein, nu, even_nv, DEFAULT_WALL_THICKNESS)?;
    ensure_positive_signed_volume(&mut mesh);

    let (_, cols) = geom.grid_dims;
    let mut sheet_vals = Vec::with_capacity(nu * even_nv);
    for i in 0..nu {
        let src_i = i.min(geom.grid_dims.0.saturating_sub(1));
        for j in 0..even_nv {
            let src_j = (j * cols / even_nv).min(cols.saturating_sub(1));
            let v_idx = (src_i * cols + src_j).min(scalar_field.len() - 1);
            sheet_vals.push(scalar_field[v_idx]);
        }
    }
    let mut full_vals = Vec::with_capacity(2 * sheet_vals.len());
    full_vals.extend_from_slice(&sheet_vals);
    full_vals.extend_from_slice(&sheet_vals);
    Ok(ColoredMesh::new(
        mesh,
        map_scalar_field_to_colors(&full_vals),
    ))
}

fn build_extruded_patch_colored_mesh(
    geom: &ManifoldGeometry,
    scalar_field: &[f64],
) -> Result<ColoredMesh, String> {
    let (rows, cols) = geom.grid_dims;
    if rows < 2 || cols < 2 || rows * cols != geom.positions.len() {
        return Err("Invalid structured grid dimensions for open-patch extrusion".to_string());
    }
    let nu = rows - 1;
    let nv = cols - 1;
    let sheet_len = rows * cols;
    let thickness = DEFAULT_WALL_THICKNESS;

    let mut vertices = vec![Point3D::new(0.0, 0.0, 0.0); 2 * sheet_len];
    let mut normals = vec![Point3D::new(0.0, 0.0, 1.0); 2 * sheet_len];
    for k in 0..sheet_len {
        let p = geom.positions[k];
        let n = geom.normals[k];
        if !p.iter().all(|c| c.is_finite()) || !n.iter().all(|c| c.is_finite()) {
            return Err("Open patch contains non-finite vertex coordinate".to_string());
        }
        vertices[k] = Point3D::new(p[0] as f32, p[1] as f32, p[2] as f32);
        normals[k] = Point3D::new(n[0] as f32, n[1] as f32, n[2] as f32);
        vertices[sheet_len + k] = Point3D::new(
            (p[0] - n[0] * thickness) as f32,
            (p[1] - n[1] * thickness) as f32,
            (p[2] - n[2] * thickness) as f32,
        );
        normals[sheet_len + k] = Point3D::new(-n[0] as f32, -n[1] as f32, -n[2] as f32);
    }

    let indices = build_rim_stitched_slab_indices(nu, nv);
    let mut mesh = Mesh {
        vertices,
        normals,
        indices,
    };
    ensure_positive_signed_volume(&mut mesh);

    let mut full_vals = Vec::with_capacity(2 * sheet_len);
    full_vals.extend_from_slice(scalar_field);
    full_vals.extend_from_slice(scalar_field);
    Ok(ColoredMesh::new(
        mesh,
        map_scalar_field_to_colors(&full_vals),
    ))
}

fn build_rim_stitched_slab_indices(nu: usize, nv: usize) -> Vec<usize> {
    let stride = nv + 1;
    let sheet_len = (nu + 1) * stride;
    let at = |i: usize, j: usize| i * stride + j;
    let mut indices = Vec::with_capacity(12 * nu * nv + 12 * (nu + nv));

    for i in 0..nu {
        for j in 0..nv {
            let (o00, o10, o11, o01) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            indices.extend_from_slice(&[o00, o10, o11, o00, o11, o01]);
            let (w00, w10, w11, w01) = (
                sheet_len + o00,
                sheet_len + o10,
                sheet_len + o11,
                sheet_len + o01,
            );
            indices.extend_from_slice(&[w00, w11, w10, w00, w01, w11]);
        }
    }

    let mut perim = Vec::with_capacity(2 * (nu + nv));
    perim.extend((0..nu).map(|i| at(i, 0)));
    perim.extend((0..nv).map(|j| at(nu, j)));
    perim.extend((1..=nu).rev().map(|i| at(i, nv)));
    perim.extend((1..=nv).rev().map(|j| at(0, j)));

    for k in 0..perim.len() {
        let a = perim[k];
        let b = perim[(k + 1) % perim.len()];
        indices.extend_from_slice(&[b, a, sheet_len + a, b, sheet_len + a, sheet_len + b]);
    }
    indices
}

fn compute_mesh_signed_volume(mesh: &Mesh) -> f64 {
    let mut vol6 = 0.0_f64;
    for tri in mesh.indices.as_chunks::<3>().0 {
        let a = mesh.vertices[tri[0]];
        let b = mesh.vertices[tri[1]];
        let c = mesh.vertices[tri[2]];
        let ax = f64::from(a.x);
        let ay = f64::from(a.y);
        let az = f64::from(a.z);
        let bx = f64::from(b.x);
        let by = f64::from(b.y);
        let bz = f64::from(b.z);
        let cx = f64::from(c.x);
        let cy = f64::from(c.y);
        let cz = f64::from(c.z);
        vol6 += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
    }
    vol6 / 6.0
}

fn ensure_positive_signed_volume(mesh: &mut Mesh) {
    if compute_mesh_signed_volume(mesh) < 0.0 {
        for tri in mesh.indices.as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
        }
        for n in &mut mesh.normals {
            n.x = -n.x;
            n.y = -n.y;
            n.z = -n.z;
        }
    }
}
