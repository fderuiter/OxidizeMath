#![cfg_attr(any(), verified(opt_out = "infrastructure"))]

use math_explorer::pure_math::differential_geometry::surface::ParametricSurface;
use nalgebra::Vector3;
use oxidize_core::mesh::{Mesh, Point3D};
#[cfg(test)]
use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

/// Minimum normal-offset shell wall thickness for non-orientable surface printing.
pub const MIN_WALL_THICKNESS: f64 = 0.05;
/// Maximum normal-offset shell wall thickness for non-orientable surface printing.
pub const MAX_WALL_THICKNESS: f64 = 0.50;
/// Default normal-offset shell wall thickness.
pub const DEFAULT_WALL_THICKNESS: f64 = 0.15;

fn clamp_grid_res(res: usize) -> usize {
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

fn has_polar_singularities(surface: &dyn ParametricSurface) -> bool {
    let n0 = surface.position(0.0, 0.0);
    let n1 = surface.position(1.1, 0.0);
    let s0 = surface.position(0.0, PI);
    let s1 = surface.position(1.1, PI);
    (n0 - n1).norm() < 1e-8 && (s0 - s1).norm() < 1e-8 && (n0 - s0).norm() > 1e-6
}

fn has_direct_periodic_seams(surface: &dyn ParametricSurface) -> bool {
    let p_u0 = surface.position(0.0, 0.7);
    let p_u1 = surface.position(TAU, 0.7);
    let p_v0 = surface.position(0.7, 0.0);
    let p_v1 = surface.position(0.7, TAU);
    (p_u0 - p_u1).norm() < 1e-7 && (p_v0 - p_v1).norm() < 1e-7
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

fn detect_u_seam_bijection(surface: &dyn ParametricSurface, nv: usize) -> Vec<(usize, bool)> {
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

    let (reverse_v, shift, _) = best;
    let same_orient = !reverse_v;
    (0..nv)
        .map(|j| {
            let m = if reverse_v {
                (shift + nv - j) % nv
            } else {
                (j + shift) % nv
            };
            (m, same_orient)
        })
        .collect()
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

    let seam_map = detect_u_seam_bijection(surface, nv);
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
            if same_orient { m } else { sheet_len + m }
        }
    };
    let inner_at = |i: usize, j: usize| -> usize {
        let jj = j % nv;
        if i < nu {
            sheet_len + i * nv + jj
        } else {
            let (m, same_orient) = seam_map[jj];
            if same_orient { sheet_len + m } else { m }
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

/// Dispatches `surface` to polar welding, periodic seam welding, or normal-offset shell extrusion.
pub fn surface_to_mesh_with_thickness(
    surface: &dyn ParametricSurface,
    u_res: usize,
    v_res: usize,
    wall_thickness: f64,
) -> Result<Mesh, String> {
    if has_polar_singularities(surface) {
        polar_surface_to_manifold_mesh(surface, u_res, v_res)
    } else if has_direct_periodic_seams(surface) {
        periodic_surface_to_manifold_mesh(surface, u_res, v_res)
    } else {
        extrude_surface_wall_shell(surface, u_res, v_res, wall_thickness)
    }
}

/// Converts any [`ParametricSurface`] into a watertight 2-manifold [`Mesh`] using default shell thickness when needed.
#[cfg(test)]
pub fn surface_to_mesh(surface: &dyn ParametricSurface, u_res: usize, v_res: usize) -> Mesh {
    surface_to_mesh_with_thickness(surface, u_res, v_res, DEFAULT_WALL_THICKNESS).unwrap_or_else(
        |_| Mesh {
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        },
    )
}

/// Counts `(open_boundary_edges, non_manifold_edges)` across all directed triangle half-edges of `mesh`.
#[cfg(test)]
pub fn count_open_and_nonmanifold_edges(mesh: &Mesh) -> (usize, usize) {
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    let mut undirected: HashMap<(usize, usize), usize> = HashMap::new();
    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            *directed.entry((a, b)).or_insert(0) += 1;
            let key = if a <= b { (a, b) } else { (b, a) };
            *undirected.entry(key).or_insert(0) += 1;
        }
    }
    let mut open_edges = 0_usize;
    let mut non_manifold = 0_usize;
    for (&(u, v), &total) in &undirected {
        let fwd = directed.get(&(u, v)).copied().unwrap_or(0);
        let rev = directed.get(&(v, u)).copied().unwrap_or(0);
        if total == 1 {
            open_edges += 1;
        } else if total != 2 || fwd != 1 || rev != 1 {
            non_manifold += 1;
        }
    }
    (open_edges, non_manifold)
}

/// Computes the combinatorial Euler characteristic `V - E + F` of `mesh`.
#[cfg(test)]
pub fn euler_characteristic(mesh: &Mesh) -> isize {
    let mut edges = std::collections::HashSet::new();
    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            edges.insert(if a <= b { (a, b) } else { (b, a) });
        }
    }
    let v = mesh.vertices.len() as isize;
    let e = edges.len() as isize;
    let f = (mesh.indices.len() / 3) as isize;
    v - e + f
}

/// Computes six times the signed volume (`> 0` for outward-wound closed manifolds).
#[cfg(test)]
pub fn signed_volume(mesh: &Mesh) -> f64 {
    let mut sum = 0.0_f64;
    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        let a = mesh.vertices[i0];
        let b = mesh.vertices[i1];
        let c = mesh.vertices[i2];
        let ax = a.x as f64;
        let ay = a.y as f64;
        let az = a.z as f64;
        let bx = b.x as f64;
        let by = b.y as f64;
        let bz = b.z as f64;
        let cx = c.x as f64;
        let cy = c.y as f64;
        let cz = c.z as f64;
        sum += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    }
    sum / 6.0
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use math_explorer::pure_math::differential_geometry::surface::{KleinBottle, Sphere, Torus};

    struct OrientationReversingKlein {
        radius: f64,
    }

    impl ParametricSurface for OrientationReversingKlein {
        fn position(&self, u: f64, v: f64) -> nalgebra::Point3<f64> {
            let r = self.radius;
            let half_u = 0.5 * u;
            let a = r + half_u.cos() * v.sin() - half_u.sin() * (2.0 * v).sin();
            nalgebra::Point3::new(
                a * u.cos(),
                a * u.sin(),
                half_u.sin() * v.sin() + half_u.cos() * (2.0 * v).sin(),
            )
        }
    }

    #[test]
    fn sphere_torus_and_klein_bottle_are_watertight_with_zero_nan_normals() {
        let sphere_mesh = surface_to_mesh(&Sphere { radius: 1.5 }, 24, 18);
        assert_eq!(sphere_mesh.normals[0], Point3D::new(0.0, 0.0, 1.0));
        assert_eq!(sphere_mesh.normals[1], Point3D::new(0.0, 0.0, -1.0));
        assert!(sphere_mesh.normals.iter().all(|n| n.is_finite()));
        assert_eq!(count_open_and_nonmanifold_edges(&sphere_mesh), (0, 0));
        assert_eq!(euler_characteristic(&sphere_mesh), 2);
        assert!(signed_volume(&sphere_mesh) > 0.0);

        let torus_mesh = surface_to_mesh(
            &Torus {
                major_radius: 2.0,
                minor_radius: 0.6,
            },
            24,
            18,
        );
        assert!(torus_mesh.normals.iter().all(|n| n.is_finite()));
        assert_eq!(count_open_and_nonmanifold_edges(&torus_mesh), (0, 0));
        assert_eq!(euler_characteristic(&torus_mesh), 0);
        assert!(signed_volume(&torus_mesh) > 0.0);

        for thickness in [MIN_WALL_THICKNESS, 0.20, MAX_WALL_THICKNESS] {
            let klein_mesh = extrude_surface_wall_shell(
                &KleinBottle { radius: 2.0 },
                24,
                20,
                thickness,
            )
            .expect("KleinBottle wall shell should succeed");
            assert!(klein_mesh.normals.iter().all(|n| n.is_finite()));
            assert_eq!(count_open_and_nonmanifold_edges(&klein_mesh), (0, 0));
            assert_eq!(euler_characteristic(&klein_mesh), 0);
            assert!(signed_volume(&klein_mesh) > 0.0);

            let mobius_klein = extrude_surface_wall_shell(
                &OrientationReversingKlein { radius: 3.0 },
                24,
                20,
                thickness,
            )
            .expect("orientation-reversing figure-8 Klein wall shell should succeed");
            assert!(mobius_klein.normals.iter().all(|n| n.is_finite()));
            assert_eq!(count_open_and_nonmanifold_edges(&mobius_klein), (0, 0));
            assert_eq!(euler_characteristic(&mobius_klein), 0);
            assert!(signed_volume(&mobius_klein) > 0.0);
        }
    }
}
