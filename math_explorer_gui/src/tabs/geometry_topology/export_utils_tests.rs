#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::pure_math::differential_geometry::surface::{
    CompiledHeightfieldSurface, CompiledParametricSurface, KleinBottle, Sphere, Torus,
};
use std::collections::{HashMap, HashSet};

/// Converts any [`ParametricSurface`] on `[0, 2*PI]^2` into a watertight 2-manifold [`Mesh`].
pub fn surface_to_mesh(surface: &dyn ParametricSurface, u_res: usize, v_res: usize) -> Mesh {
    surface_to_mesh_with_domain_and_thickness(
        surface,
        (0.0, TAU),
        (0.0, TAU),
        u_res,
        v_res,
        DEFAULT_WALL_THICKNESS,
    )
    .unwrap_or_else(|_| Mesh {
        vertices: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
    })
}

/// Counts `(open_boundary_edges, non_manifold_edges)` across all directed triangle half-edges of `mesh`.
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
pub fn euler_characteristic(mesh: &Mesh) -> isize {
    let mut edges = HashSet::new();
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

/// Computes the signed volume (`> 0` for outward-wound closed manifolds).
pub fn signed_volume(mesh: &Mesh) -> f64 {
    let mut sum = 0.0_f64;
    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        let (a, b, c) = (mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]);
        let (ax, ay, az) = (a.x as f64, a.y as f64, a.z as f64);
        let (bx, by, bz) = (b.x as f64, b.y as f64, b.z as f64);
        let (cx, cy, cz) = (c.x as f64, c.y as f64, c.z as f64);
        sum += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    }
    sum / 6.0
}

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
        let klein_mesh =
            extrude_surface_wall_shell(&KleinBottle { radius: 2.0 }, 24, 20, thickness)
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

#[test]
fn custom_heightfield_and_open_parametric_patches_extrude_watertight_solids() {
    let saddle_expr = CompiledExpr::compile_with_vars(
        &parse("a * (x^2 - y^2)").expect("saddle parses"),
        &["x", "y"],
    )
    .expect("saddle compiles");
    let heightfield = CompiledHeightfieldSurface::new(saddle_expr, &[0.45]);
    let hf_mesh = surface_to_mesh_with_domain_and_thickness(
        &heightfield,
        (-2.0, 2.0),
        (-2.0, 2.0),
        16,
        16,
        DEFAULT_WALL_THICKNESS,
    )
    .expect("heightfield solid slab builds");
    assert!(hf_mesh.normals.iter().all(|n| n.is_finite()));
    assert_eq!(count_open_and_nonmanifold_edges(&hf_mesh), (0, 0));
    assert_eq!(euler_characteristic(&hf_mesh), 2);
    assert!(signed_volume(&hf_mesh) > 0.0);

    let cx =
        CompiledExpr::compile_with_vars(&parse("(1 + 0.2 * v) * cos(u)").unwrap(), &["u", "v"])
            .unwrap();
    let cy =
        CompiledExpr::compile_with_vars(&parse("(1 + 0.2 * v) * sin(u)").unwrap(), &["u", "v"])
            .unwrap();
    let cz = CompiledExpr::compile_with_vars(&parse("0.4 * u").unwrap(), &["u", "v"]).unwrap();
    let helicoid = CompiledParametricSurface::new(cx, cy, cz, &[]);
    let open_mesh = surface_to_mesh_with_domain_and_thickness(
        &helicoid,
        (0.0, TAU),
        (0.0, TAU),
        18,
        12,
        DEFAULT_WALL_THICKNESS,
    )
    .expect("open parametric patch slab builds");
    assert!(open_mesh.normals.iter().all(|n| n.is_finite()));
    assert_eq!(count_open_and_nonmanifold_edges(&open_mesh), (0, 0));
    assert_eq!(euler_characteristic(&open_mesh), 2);
    assert!(signed_volume(&open_mesh) > 0.0);
}
