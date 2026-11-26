#![cfg_attr(any(), verified(opt_out = "infrastructure"))]

use math_explorer::pure_math::differential_geometry::surface::ParametricSurface;
use nalgebra::Vector3;
use oxidize_core::mesh::{Mesh, Point3D};
use std::f64::consts::{PI, TAU};

/// Minimum normal-offset shell wall thickness for non-orientable or open-patch surface printing.
pub const MIN_WALL_THICKNESS: f64 = 0.05;
/// Maximum normal-offset shell wall thickness for non-orientable or open-patch surface printing.
pub const MAX_WALL_THICKNESS: f64 = 0.50;
/// Default normal-offset shell wall thickness.
pub const DEFAULT_WALL_THICKNESS: f64 = 0.15;

pub(super) fn clamp_grid_res(res: usize) -> usize {
    res.clamp(4, math_commons::registry::MAX_GRID_SIZE)
}

fn safe_unit_vector(v: Vector3<f64>, fallback: Vector3<f64>) -> Vector3<f64> {
    let norm = v.norm();
    if norm.is_finite() && norm > 1e-12 {
        v / norm
    } else {
        fallback
    }
}

fn to_p3(x: f64, y: f64, z: f64) -> Point3D {
    Point3D::new(x as f32, y as f32, z as f32)
}

pub(super) fn is_standard_tau_domain(u_range: (f64, f64), v_range: (f64, f64)) -> bool {
    u_range.0.abs() < 1e-9
        && (u_range.1 - TAU).abs() < 1e-7
        && v_range.0.abs() < 1e-9
        && (v_range.1 - TAU).abs() < 1e-7
}

pub(super) fn has_polar_singularities(surface: &dyn ParametricSurface) -> bool {
    let n0 = surface.position(0.0, 0.0);
    let n1 = surface.position(1.1, 0.0);
    let s0 = surface.position(0.0, PI);
    let s1 = surface.position(1.1, PI);
    (n0 - n1).norm() < 1e-8 && (s0 - s1).norm() < 1e-8 && (n0 - s0).norm() > 1e-6
}

pub(super) fn has_direct_periodic_seams(surface: &dyn ParametricSurface) -> bool {
    let p_u0 = surface.position(0.0, 0.7);
    let p_u1 = surface.position(TAU, 0.7);
    let p_v0 = surface.position(0.7, 0.0);
    let p_v1 = surface.position(0.7, TAU);
    (p_u0 - p_u1).norm() < 1e-7 && (p_v0 - p_v1).norm() < 1e-7
}

pub(super) fn has_klein_seam_topology(surface: &dyn ParametricSurface) -> bool {
    let v_closed_a = (surface.position(0.7, 0.0) - surface.position(0.7, TAU)).norm() < 1e-7;
    let v_closed_b = (surface.position(2.1, 0.0) - surface.position(2.1, TAU)).norm() < 1e-7;
    if !v_closed_a || !v_closed_b {
        return false;
    }
    let (_, min_seam_sq_err) = detect_u_seam_bijection_with_error(surface, 16);
    min_seam_sq_err < 1e-6
}

/// Builds a closed genus-0 manifold mesh for polar surfaces (`v` in `[0, PI]`, welded poles at `(0, 0, ±1)`).
pub fn polar_surface_to_manifold_mesh(
    surface: &dyn ParametricSurface,
    u_res: usize,
    v_res: usize,
) -> Result<Mesh, String> {
    let nu = clamp_grid_res(u_res);
    let nv = clamp_grid_res(v_res);
    let north = surface.position(0.0, 0.0);
    let south = surface.position(0.0, PI);
    if !north.coords.iter().all(|c| c.is_finite()) || !south.coords.iter().all(|c| c.is_finite()) {
        return Err("Surface pole coordinates must be finite".to_string());
    }

    let num_verts = 2 + (nv - 1) * nu;
    let mut vertices = Vec::with_capacity(num_verts);
    let mut normals = Vec::with_capacity(num_verts);
    vertices.push(to_p3(north.x, north.y, north.z));
    normals.push(Point3D::new(0.0, 0.0, 1.0));
    vertices.push(to_p3(south.x, south.y, south.z));
    normals.push(Point3D::new(0.0, 0.0, -1.0));

    for j in 1..nv {
        let v = PI * (j as f64 / nv as f64);
        for i in 0..nu {
            let u = TAU * (i as f64 / nu as f64);
            let p = surface.position(u, v);
            if !p.coords.iter().all(|c| c.is_finite()) {
                return Err("Surface position contains non-finite coordinate".to_string());
            }
            let radial = safe_unit_vector(p.coords, Vector3::new(0.0, 0.0, 1.0));
            let outward = safe_unit_vector(
                surface.partial_v(u, v).cross(&surface.partial_u(u, v)),
                radial,
            );
            vertices.push(to_p3(p.x, p.y, p.z));
            normals.push(to_p3(outward.x, outward.y, outward.z));
        }
    }

    let indices = build_polar_indices(nu, nv);
    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}

fn build_polar_indices(nu: usize, nv: usize) -> Vec<usize> {
    let ring = |j: usize, i: usize| -> usize { 2 + (j - 1) * nu + (i % nu) };
    let mut indices = Vec::with_capacity(6 * (nv - 1) * nu);
    for i in 0..nu {
        indices.extend_from_slice(&[0, ring(1, i), ring(1, i + 1)]);
    }
    for j in 1..(nv - 1) {
        for i in 0..nu {
            let (r00, r10, r11, r01) = (
                ring(j, i),
                ring(j + 1, i),
                ring(j + 1, i + 1),
                ring(j, i + 1),
            );
            indices.extend_from_slice(&[r00, r10, r11, r00, r11, r01]);
        }
    }
    for i in 0..nu {
        indices.extend_from_slice(&[1, ring(nv - 1, i + 1), ring(nv - 1, i)]);
    }
    indices
}

/// Builds a closed genus-1 manifold mesh for doubly periodic surfaces (`u, v` in `[0, 2*PI)`).
pub fn periodic_surface_to_manifold_mesh(
    surface: &dyn ParametricSurface,
    u_res: usize,
    v_res: usize,
) -> Result<Mesh, String> {
    let nu = clamp_grid_res(u_res);
    let nv = clamp_grid_res(v_res);
    let mut vertices = Vec::with_capacity(nu * nv);
    let mut normals = Vec::with_capacity(nu * nv);

    for i in 0..nu {
        let u = TAU * (i as f64 / nu as f64);
        for j in 0..nv {
            let v = TAU * (j as f64 / nv as f64);
            let p = surface.position(u, v);
            if !p.coords.iter().all(|c| c.is_finite()) {
                return Err("Surface position contains non-finite coordinate".to_string());
            }
            let n = safe_unit_vector(
                surface.partial_u(u, v).cross(&surface.partial_v(u, v)),
                Vector3::new(0.0, 0.0, 1.0),
            );
            vertices.push(to_p3(p.x, p.y, p.z));
            normals.push(to_p3(n.x, n.y, n.z));
        }
    }

    let mut indices = Vec::with_capacity(6 * nu * nv);
    let idx = |i: usize, j: usize| -> usize { (i % nu) * nv + (j % nv) };
    for i in 0..nu {
        for j in 0..nv {
            let (v00, v10, v11, v01) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
            indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
        }
    }
    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}

fn detect_u_seam_bijection_with_error(
    surface: &dyn ParametricSurface,
    nv: usize,
) -> (Vec<(usize, bool)>, f64) {
    let p0: Vec<_> = (0..nv)
        .map(|j| surface.position(0.0, TAU * (j as f64 / nv as f64)).coords)
        .collect();
    let p_end: Vec<_> = (0..nv)
        .map(|j| surface.position(TAU, TAU * (j as f64 / nv as f64)).coords)
        .collect();

    let sample_step = (nv / 64).max(1);
    let mut best = (false, 0_usize, f64::INFINITY);
    for reverse_v in [false, true] {
        for shift in 0..nv {
            let mut err = 0.0_f64;
            for j in (0..nv).step_by(sample_step) {
                let m = if reverse_v {
                    (shift + nv - j) % nv
                } else {
                    (j + shift) % nv
                };
                err += (p_end[j] - p0[m]).norm_squared();
            }
            if err < best.2 {
                best = (reverse_v, shift, err);
            }
        }
    }

    let (reverse_v, shift, min_err) = best;
    let same_orient = !reverse_v;
    let map = (0..nv)
        .map(|j| {
            let m = if reverse_v {
                (shift + nv - j) % nv
            } else {
                (j + shift) % nv
            };
            (m, same_orient)
        })
        .collect();
    (map, min_err)
}

/// Extrudes a normal-offset solid shell (`wall_thickness` in `[0.05, 0.50]`) stitching Klein seams into a watertight solid.
pub fn extrude_surface_wall_shell(
    surface: &dyn ParametricSurface,
    u_res: usize,
    v_res: usize,
    wall_thickness: f64,
) -> Result<Mesh, String> {
    if !wall_thickness.is_finite() {
        return Err("Wall thickness must be finite".to_string());
    }
    let half_t = 0.5 * wall_thickness.clamp(MIN_WALL_THICKNESS, MAX_WALL_THICKNESS);
    let nu = clamp_grid_res(u_res);
    let raw_nv = clamp_grid_res(v_res);
    let nv = if raw_nv.is_multiple_of(2) {
        raw_nv
    } else {
        (raw_nv + 1).min(math_commons::registry::MAX_GRID_SIZE)
    };

    let sheet_len = nu * nv;
    let mut vertices = vec![Point3D::new(0.0, 0.0, 0.0); 2 * sheet_len];
    let mut normals = vec![Point3D::new(0.0, 0.0, 1.0); 2 * sheet_len];
    for i in 0..nu {
        let u = TAU * (i as f64 / nu as f64);
        for j in 0..nv {
            let v = TAU * (j as f64 / nv as f64);
            let p = surface.position(u, v);
            if !p.coords.iter().all(|c| c.is_finite()) {
                return Err("Surface position contains non-finite coordinate".to_string());
            }
            let n = safe_unit_vector(
                surface.partial_u(u, v).cross(&surface.partial_v(u, v)),
                Vector3::new(0.0, 0.0, 1.0),
            );
            let p_out = p.coords + n * half_t;
            let p_in = p.coords - n * half_t;
            let k = i * nv + j;
            vertices[k] = to_p3(p_out.x, p_out.y, p_out.z);
            normals[k] = to_p3(n.x, n.y, n.z);
            vertices[sheet_len + k] = to_p3(p_in.x, p_in.y, p_in.z);
            normals[sheet_len + k] = to_p3(-n.x, -n.y, -n.z);
        }
    }

    let (seam_map, _) = detect_u_seam_bijection_with_error(surface, nv);
    let indices = build_wall_shell_indices(nu, nv, &seam_map);
    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}

fn build_wall_shell_indices(nu: usize, nv: usize, seam_map: &[(usize, bool)]) -> Vec<usize> {
    let sheet_len = nu * nv;
    let outer_at = |i: usize, j: usize| -> usize {
        let jj = j % nv;
        if i < nu {
            i * nv + jj
        } else {
            let (m, same_orient) = seam_map[jj];
            if same_orient {
                m
            } else {
                sheet_len + m
            }
        }
    };
    let inner_at = |i: usize, j: usize| -> usize {
        let jj = j % nv;
        if i < nu {
            sheet_len + i * nv + jj
        } else {
            let (m, same_orient) = seam_map[jj];
            if same_orient {
                sheet_len + m
            } else {
                m
            }
        }
    };

    let mut indices = Vec::with_capacity(12 * sheet_len);
    for i in 0..nu {
        for j in 0..nv {
            let (o00, o10, o11, o01) = (
                outer_at(i, j),
                outer_at(i + 1, j),
                outer_at(i + 1, j + 1),
                outer_at(i, j + 1),
            );
            indices.extend_from_slice(&[o00, o10, o11, o00, o11, o01]);

            let (w00, w10, w11, w01) = (
                inner_at(i, j),
                inner_at(i + 1, j),
                inner_at(i + 1, j + 1),
                inner_at(i, j + 1),
            );
            indices.extend_from_slice(&[w00, w11, w10, w00, w01, w11]);
        }
    }
    indices
}

/// Extrudes an open rectangular `(u, v)` surface patch into a closed genus-0 solid slab with stitched rim walls.
pub fn extrude_open_patch_wall_shell(
    surface: &dyn ParametricSurface,
    u_range: (f64, f64),
    v_range: (f64, f64),
    u_res: usize,
    v_res: usize,
    wall_thickness: f64,
) -> Result<Mesh, String> {
    if !wall_thickness.is_finite() {
        return Err("Wall thickness must be finite".to_string());
    }
    let thickness = wall_thickness.clamp(MIN_WALL_THICKNESS, MAX_WALL_THICKNESS);
    let nu = clamp_grid_res(u_res);
    let nv = clamp_grid_res(v_res);
    let (u_min, u_max) = u_range;
    let (v_min, v_max) = v_range;
    let stride = nv + 1;
    let sheet_len = (nu + 1) * stride;
    let mut vertices = vec![Point3D::new(0.0, 0.0, 0.0); 2 * sheet_len];
    let mut normals = vec![Point3D::new(0.0, 0.0, 1.0); 2 * sheet_len];

    for i in 0..=nu {
        let u = u_min + (u_max - u_min) * (i as f64 / nu as f64);
        for j in 0..=nv {
            let v = v_min + (v_max - v_min) * (j as f64 / nv as f64);
            let p = surface.position(u, v);
            if !p.coords.iter().all(|c| c.is_finite()) {
                return Err("Surface position contains non-finite coordinate".to_string());
            }
            let n = safe_unit_vector(
                surface.partial_u(u, v).cross(&surface.partial_v(u, v)),
                Vector3::new(0.0, 0.0, 1.0),
            );
            let p_out = p.coords;
            let p_in = p.coords - n * thickness;
            let k = i * stride + j;
            vertices[k] = to_p3(p_out.x, p_out.y, p_out.z);
            normals[k] = to_p3(n.x, n.y, n.z);
            vertices[sheet_len + k] = to_p3(p_in.x, p_in.y, p_in.z);
            normals[sheet_len + k] = to_p3(-n.x, -n.y, -n.z);
        }
    }

    let indices = build_open_patch_shell_indices(nu, nv);
    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}

fn build_open_patch_shell_indices(nu: usize, nv: usize) -> Vec<usize> {
    let stride = nv + 1;
    let sheet_len = (nu + 1) * stride;
    let at = |i: usize, j: usize| -> usize { i * stride + j };
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

/// Dispatches `surface` over `(u_range, v_range)` to polar welding, periodic welding, Klein shell extrusion, or rim-stitched slab extrusion.
pub fn surface_to_mesh_with_domain_and_thickness(
    surface: &dyn ParametricSurface,
    u_range: (f64, f64),
    v_range: (f64, f64),
    u_res: usize,
    v_res: usize,
    wall_thickness: f64,
) -> Result<Mesh, String> {
    if is_standard_tau_domain(u_range, v_range) {
        if has_polar_singularities(surface) {
            return polar_surface_to_manifold_mesh(surface, u_res, v_res);
        }
        if has_direct_periodic_seams(surface) {
            return periodic_surface_to_manifold_mesh(surface, u_res, v_res);
        }
        if has_klein_seam_topology(surface) {
            return extrude_surface_wall_shell(surface, u_res, v_res, wall_thickness);
        }
    }
    extrude_open_patch_wall_shell(surface, u_range, v_range, u_res, v_res, wall_thickness)
}

#[cfg(test)]
#[path = "export_utils_tests.rs"]
mod tests;

#[cfg(test)]
pub use tests::{count_open_and_nonmanifold_edges, euler_characteristic, signed_volume};
