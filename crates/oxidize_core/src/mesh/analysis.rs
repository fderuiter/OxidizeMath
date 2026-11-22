//! Physical millimeter bounds, surface area, signed volume, and manifold diagnostics for [`Mesh`].

use super::types::{Mesh, Point3D};
use std::collections::HashMap;

/// Topological and physical millimeter diagnostics for an indexed triangle [`Mesh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshDiagnostics {
    /// Total vertex count in the mesh buffer.
    pub vertex_count: usize,
    /// Total triangle facet count (`indices.len() / 3`).
    pub triangle_count: usize,
    /// Number of directed half-edges without a matching opposite half-edge (plus degenerate edges).
    pub open_edges: usize,
    /// Number of vertex normals containing `NaN`, `Inf`, or near-zero magnitude.
    pub nan_normals: usize,
    /// Minimum physical coordinate corner in millimeters.
    pub min_bounds: Point3D,
    /// Maximum physical coordinate corner in millimeters.
    pub max_bounds: Point3D,
    /// Axis-aligned bounding box span `(dx, dy, dz)` in millimeters.
    pub dimensions_mm: Point3D,
    /// Total surface area in square millimeters (`mm^2`).
    pub area_mm2: f64,
    /// Signed enclosed volume via the divergence theorem in cubic millimeters (`mm^3`).
    pub signed_volume_mm3: f64,
}

impl Default for MeshDiagnostics {
    fn default() -> Self {
        let zero = Point3D::new(0.0, 0.0, 0.0);
        Self {
            vertex_count: 0,
            triangle_count: 0,
            open_edges: 0,
            nan_normals: 0,
            min_bounds: zero,
            max_bounds: zero,
            dimensions_mm: zero,
            area_mm2: 0.0,
            signed_volume_mm3: 0.0,
        }
    }
}

impl MeshDiagnostics {
    /// Returns `true` when the mesh is non-empty, closed (`0` open edges), free of invalid
    /// normals (`0` `NaN` normals), and encloses positive physical volume (`mm^3 > 0`).
    #[must_use]
    pub fn is_watertight(&self) -> bool {
        self.triangle_count > 0
            && self.open_edges == 0
            && self.nan_normals == 0
            && self.signed_volume_mm3 > 0.0
    }
}

/// Computes manifold topology checks and physical millimeter metrics for an indexed [`Mesh`].
#[must_use]
pub fn analyze_mesh(mesh: &Mesh) -> MeshDiagnostics {
    if mesh.vertices.is_empty() {
        return MeshDiagnostics::default();
    }

    let (min_bounds, max_bounds) = compute_bounds(&mesh.vertices);
    let dimensions_mm = max_bounds - min_bounds;
    let nan_normals = count_invalid_normals(mesh);
    let (triangle_count, open_edges, area_mm2, signed_volume_mm3) = accumulate_triangles(mesh);

    MeshDiagnostics {
        vertex_count: mesh.vertices.len(),
        triangle_count,
        open_edges,
        nan_normals,
        min_bounds,
        max_bounds,
        dimensions_mm,
        area_mm2,
        signed_volume_mm3,
    }
}

fn compute_bounds(vertices: &[Point3D]) -> (Point3D, Point3D) {
    let mut min_b = Point3D::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_b = Point3D::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut any_finite = false;

    for v in vertices {
        if v.x.is_finite() && v.y.is_finite() && v.z.is_finite() {
            any_finite = true;
            min_b.x = min_b.x.min(v.x);
            min_b.y = min_b.y.min(v.y);
            min_b.z = min_b.z.min(v.z);
            max_b.x = max_b.x.max(v.x);
            max_b.y = max_b.y.max(v.y);
            max_b.z = max_b.z.max(v.z);
        }
    }

    if any_finite {
        (min_b, max_b)
    } else {
        let zero = Point3D::new(0.0, 0.0, 0.0);
        (zero, zero)
    }
}

fn count_invalid_normals(mesh: &Mesh) -> usize {
    let missing = mesh.vertices.len().saturating_sub(mesh.normals.len());
    let bad_stored = mesh
        .normals
        .iter()
        .filter(|n| {
            !n.x.is_finite()
                || !n.y.is_finite()
                || !n.z.is_finite()
                || (n.x * n.x + n.y * n.y + n.z * n.z) <= 1.0e-12
        })
        .count();
    missing + bad_stored
}

fn accumulate_triangles(mesh: &Mesh) -> (usize, usize, f64, f64) {
    let tri_count = mesh.indices.len() / 3;
    let mut directed_edges: HashMap<(usize, usize), isize> = HashMap::with_capacity(tri_count * 3);
    let mut degenerate_edges = 0usize;
    let mut area = 0.0_f64;
    let mut signed_vol6 = 0.0_f64;
    let v_len = mesh.vertices.len();

    for &[i0, i1, i2] in mesh.indices.as_chunks::<3>().0 {
        if i0 >= v_len || i1 >= v_len || i2 >= v_len || i0 == i1 || i1 == i2 || i2 == i0 {
            degenerate_edges += 3;
            continue;
        }

        record_half_edge(&mut directed_edges, i0, i1);
        record_half_edge(&mut directed_edges, i1, i2);
        record_half_edge(&mut directed_edges, i2, i0);

        let (tri_area, tri_vol6) =
            triangle_area_and_vol6(mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]);
        area += tri_area;
        signed_vol6 += tri_vol6;
    }

    let unmatched: usize = directed_edges
        .values()
        .map(|&balance| balance.unsigned_abs())
        .sum();
    (
        tri_count,
        unmatched + degenerate_edges,
        area,
        signed_vol6 / 6.0,
    )
}

fn record_half_edge(edges: &mut HashMap<(usize, usize), isize>, u: usize, v: usize) {
    let key = if u < v { (u, v) } else { (v, u) };
    let sign = if u < v { 1 } else { -1 };
    *edges.entry(key).or_insert(0) += sign;
}

fn triangle_area_and_vol6(p0: Point3D, p1: Point3D, p2: Point3D) -> (f64, f64) {
    let p0 = [f64::from(p0.x), f64::from(p0.y), f64::from(p0.z)];
    let p1 = [f64::from(p1.x), f64::from(p1.y), f64::from(p1.z)];
    let p2 = [f64::from(p2.x), f64::from(p2.y), f64::from(p2.z)];

    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

    let p1_cross_p2 = [
        p1[1] * p2[2] - p1[2] * p2[1],
        p1[2] * p2[0] - p1[0] * p2[2],
        p1[0] * p2[1] - p1[1] * p2[0],
    ];
    let vol6 = p0[0] * p1_cross_p2[0] + p0[1] * p1_cross_p2[1] + p0[2] * p1_cross_p2[2];
    (area, vol6)
}
