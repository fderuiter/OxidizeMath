//! Discrete Cotangent Laplace-Beltrami Operator in Compressed Sparse Row (CSR) format.
//!
//! Discretizes the intrinsic Riemannian Laplace-Beltrami operator $\Delta_S$ over
//! closed, polar-welded, periodic, and extruded 3D triangle meshes using Meyer et al. (2003)
//! mixed Voronoi vertex areas and cotangent edge weights with `1e-12` denominator guards.
// [cite:freesurfer]

use oxidize_core::mesh::Mesh;

const MIN_DENOM: f64 = 1e-12;

/// Compressed Sparse Row (CSR) discrete cotangent Laplace-Beltrami operator over a 3D triangle mesh.
///
/// Evaluates $(\Delta_S u)_i = \frac{1}{A_i} \sum_{j \in \mathcal{N}(i)} w_{ij} (u_j - u_i)$
/// with zero heap allocations on hot stepping paths.
#[derive(Debug, Clone, PartialEq)]
pub struct CotangentLaplacian {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    weights: Vec<f64>,
    areas: Vec<f64>,
    inv_areas: Vec<f64>,
    mean_edge_length: f64,
}

#[verified_engine::verified]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[verified_engine::verified]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[verified_engine::verified]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[verified_engine::verified]
fn accumulate_mixed_voronoi_area(
    areas: &mut [f64],
    tri: [usize; 3],
    l_sq: [f64; 3],
    cot: [f64; 3],
    tri_area: f64,
) {
    let [i0, i1, i2] = tri;
    if cot[0] < 0.0 {
        areas[i0] += 0.5 * tri_area;
        areas[i1] += 0.25 * tri_area;
        areas[i2] += 0.25 * tri_area;
    } else if cot[1] < 0.0 {
        areas[i1] += 0.5 * tri_area;
        areas[i0] += 0.25 * tri_area;
        areas[i2] += 0.25 * tri_area;
    } else if cot[2] < 0.0 {
        areas[i2] += 0.5 * tri_area;
        areas[i0] += 0.25 * tri_area;
        areas[i1] += 0.25 * tri_area;
    } else {
        areas[i0] += 0.125 * (l_sq[1] * cot[1] + l_sq[2] * cot[2]);
        areas[i1] += 0.125 * (l_sq[2] * cot[2] + l_sq[0] * cot[0]);
        areas[i2] += 0.125 * (l_sq[0] * cot[0] + l_sq[1] * cot[1]);
    }
}

fn coalesce_csr_entries(
    num_vertices: usize,
    mut half_edges: Vec<(usize, usize, f64, f64)>,
) -> (Vec<usize>, Vec<usize>, Vec<f64>, f64) {
    half_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut row_ptr = vec![0_usize; num_vertices + 1];
    let mut col_idx = Vec::with_capacity(half_edges.len() / 2);
    let mut weights = Vec::with_capacity(half_edges.len() / 2);
    let mut edge_len_sum = 0.0;
    let mut edge_count = 0_usize;

    for (r, c, w, len) in half_edges {
        if r == c || r >= num_vertices || c >= num_vertices {
            continue;
        }
        if let Some(&last_c) = col_idx.last()
            && weights.len() > row_ptr[r]
            && last_c == c
        {
            if let Some(last_w) = weights.last_mut() {
                *last_w += w;
            }
        } else {
            col_idx.push(c);
            weights.push(w);
            row_ptr[r + 1] += 1;
            if r < c {
                edge_len_sum += len;
                edge_count += 1;
            }
        }
    }

    for r in 0..num_vertices {
        row_ptr[r + 1] += row_ptr[r];
    }
    let mean_len = if edge_count > 0 {
        (edge_len_sum / edge_count as f64).max(1e-6)
    } else {
        1.0
    };
    (row_ptr, col_idx, weights, mean_len)
}

impl CotangentLaplacian {
    /// Builds a CSR cotangent Laplace-Beltrami operator from indexed 3D vertex positions and triangles.
    ///
    /// Degenerate triangles (`2 * area < 1e-12`) are safely skipped and vertex mixed Voronoi areas
    /// are clamped to at least `1e-12` so polar or isolated vertices never emit `NaN` or `Inf`.
    #[must_use]
    pub fn from_indexed(positions: &[[f64; 3]], triangles: &[[usize; 3]]) -> Self {
        let n = positions.len();
        let mut areas = vec![0.0_f64; n];
        let mut half_edges = Vec::with_capacity(triangles.len() * 6);

        for &[i0, i1, i2] in triangles {
            if i0 >= n || i1 >= n || i2 >= n || i0 == i1 || i1 == i2 || i2 == i0 {
                continue;
            }
            let (p0, p1, p2) = (positions[i0], positions[i1], positions[i2]);
            let (e01, e02, e12) = (sub3(p1, p0), sub3(p2, p0), sub3(p2, p1));
            let cross = cross3(e01, e02);
            let double_area = dot3(cross, cross).sqrt();
            if !double_area.is_finite() || double_area < MIN_DENOM {
                continue;
            }

            let l_sq = [dot3(e12, e12), dot3(e02, e02), dot3(e01, e01)];
            let cots = [
                dot3(e01, e02) / double_area,
                dot3(sub3(p2, p1), sub3(p0, p1)) / double_area,
                dot3(sub3(p0, p2), sub3(p1, p2)) / double_area,
            ];
            accumulate_mixed_voronoi_area(&mut areas, [i0, i1, i2], l_sq, cots, 0.5 * double_area);

            let (w12, w20, w01) = (0.5 * cots[0], 0.5 * cots[1], 0.5 * cots[2]);
            let (len0, len1, len2) = (l_sq[0].sqrt(), l_sq[1].sqrt(), l_sq[2].sqrt());
            half_edges.extend([
                (i1, i2, w12, len0),
                (i2, i1, w12, len0),
                (i2, i0, w20, len1),
                (i0, i2, w20, len1),
                (i0, i1, w01, len2),
                (i1, i0, w01, len2),
            ]);
        }

        let (row_ptr, col_idx, weights, mean_edge_length) = coalesce_csr_entries(n, half_edges);
        let mut inv_areas = vec![0.0_f64; n];
        for (area, inv) in areas.iter_mut().zip(inv_areas.iter_mut()) {
            let clamped = if area.is_finite() {
                (*area).max(MIN_DENOM)
            } else {
                MIN_DENOM
            };
            *area = clamped;
            *inv = 1.0 / clamped;
        }

        Self {
            row_ptr,
            col_idx,
            weights,
            areas,
            inv_areas,
            mean_edge_length,
        }
    }

    /// Builds a CSR cotangent Laplace-Beltrami operator directly from an indexed [`Mesh`].
    #[must_use]
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let positions: Vec<[f64; 3]> = mesh
            .vertices
            .iter()
            .map(|v| [f64::from(v.x), f64::from(v.y), f64::from(v.z)])
            .collect();
        let triangles: Vec<[usize; 3]> = mesh.indices.as_chunks::<3>().0.to_vec();
        Self::from_indexed(&positions, &triangles)
    }

    /// Number of vertices in the discrete manifold.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn num_vertices(&self) -> usize {
        self.areas.len()
    }

    /// Number of non-zero directed off-diagonal CSR entries.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn nnz(&self) -> usize {
        self.weights.len()
    }

    /// Characteristic mean edge length of the mesh (`>= 1e-6`).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_edge_length(&self) -> f64 {
        self.mean_edge_length
    }

    /// Clamped mixed Voronoi area $A_i \ge 10^{-12}$ associated with vertex `v`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn vertex_area(&self, v: usize) -> f64 {
        self.areas[v]
    }

    /// Slice of clamped mixed Voronoi vertex areas across all vertices.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn vertex_areas(&self) -> &[f64] {
        &self.areas
    }

    /// Inverse mixed Voronoi area $A_i^{-1}$ for vertex `v`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn inv_vertex_area(&self, v: usize) -> f64 {
        self.inv_areas[v]
    }

    /// CSR neighbor vertex indices $\mathcal{N}(v)$ adjacent to `v`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn neighbors(&self, v: usize) -> &[usize] {
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        &self.col_idx[start..end]
    }

    /// CSR cotangent edge weights $w_{vj}$ aligned with [`Self::neighbors`].
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn row_weights(&self, v: usize) -> &[f64] {
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        &self.weights[start..end]
    }

    /// Evaluates $(\Delta_S u)_v$ at a single vertex `v` without heap allocation.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn apply_at(&self, field: &[f64], v: usize) -> f64 {
        let u_v = field[v];
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        let mut sum = 0.0;
        for idx in start..end {
            let j = self.col_idx[idx];
            sum += self.weights[idx] * (field[j] - u_v);
        }
        sum * self.inv_areas[v]
    }

    /// Evaluates $\Delta_S u$ across all vertices into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn apply(&self, field: &[f64], out: &mut [f64]) {
        let n = self.num_vertices();
        assert_eq!(field.len(), n, "Input field length mismatch");
        assert_eq!(out.len(), n, "Output field length mismatch");
        for (v, dst) in out.iter_mut().enumerate().take(n) {
            *dst = self.apply_at(field, v);
        }
    }

    /// Advances explicit heat diffusion $u^{k+1} = u^k + \Delta t \, \alpha \, \Delta_S u^k$ into `next`.
    #[verified_engine::verified]
    pub fn step_heat_euler(&self, field: &[f64], next: &mut [f64], diffusivity: f64, dt: f64) {
        let n = self.num_vertices();
        assert_eq!(field.len(), n, "Field length mismatch");
        assert_eq!(next.len(), n, "Next buffer length mismatch");
        let scale = dt * diffusivity;
        for v in 0..n {
            next[v] = field[v] + scale * self.apply_at(field, v);
        }
    }

    /// Computes unsigned discrete mean curvature $|H|_i = \frac{1}{2} \| (\Delta_S \mathbf{x})_i \|$ into `out`.
    #[verified_engine::verified]
    pub fn compute_mean_curvature(&self, positions: &[[f64; 3]], out: &mut [f64]) {
        let n = self.num_vertices();
        assert_eq!(positions.len(), n, "Position count mismatch");
        assert_eq!(out.len(), n, "Curvature output length mismatch");
        for v in 0..n {
            let p_v = positions[v];
            let start = self.row_ptr[v];
            let end = self.row_ptr[v + 1];
            let mut lx = 0.0;
            let mut ly = 0.0;
            let mut lz = 0.0;
            for idx in start..end {
                let p_j = positions[self.col_idx[idx]];
                let w = self.weights[idx];
                lx += w * (p_j[0] - p_v[0]);
                ly += w * (p_j[1] - p_v[1]);
                lz += w * (p_j[2] - p_v[2]);
            }
            let inv_a = self.inv_areas[v];
            let norm = ((lx * inv_a).powi(2) + (ly * inv_a).powi(2) + (lz * inv_a).powi(2)).sqrt();
            out[v] = 0.5 * norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use verified_engine::engine::VerifiedEngine;

    fn build_welded_unit_sphere(nu: usize, nv: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let mut positions = Vec::with_capacity(2 + (nv - 2) * nu);
        positions.push([0.0, 0.0, 1.0]); // North pole (v = 0)
        positions.push([0.0, 0.0, -1.0]); // South pole (v = PI)
        for j in 1..(nv - 1) {
            let v = PI * (j as f64) / ((nv - 1) as f64);
            let (sv, cv) = v.sin_cos();
            for i in 0..nu {
                let u = 2.0 * PI * (i as f64) / (nu as f64);
                let (su, cu) = u.sin_cos();
                positions.push([sv * cu, sv * su, cv]);
            }
        }
        let ring_idx = |i: usize, j: usize| -> usize { 2 + (j - 1) * nu + (i % nu) };
        let mut tris = Vec::new();
        for i in 0..nu {
            tris.push([0, ring_idx(i, 1), ring_idx(i + 1, 1)]);
        }
        for j in 1..(nv - 2) {
            for i in 0..nu {
                let a = ring_idx(i, j);
                let b = ring_idx(i + 1, j);
                let c = ring_idx(i + 1, j + 1);
                let d = ring_idx(i, j + 1);
                tris.push([a, d, c]);
                tris.push([a, c, b]);
            }
        }
        let last_ring = nv - 2;
        for i in 0..nu {
            tris.push([ring_idx(i, last_ring), 1, ring_idx(i + 1, last_ring)]);
        }
        (positions, tris)
    }

    fn spherical_harmonic_l2_error(nu: usize, nv: usize, l: usize) -> f64 {
        let (pos, tris) = build_welded_unit_sphere(nu, nv);
        let lap = CotangentLaplacian::from_indexed(&pos, &tris);
        let eigenval = -((l * (l + 1)) as f64);
        let field: Vec<f64> = pos
            .iter()
            .map(
                |&[_x, _y, z]| {
                    if l == 1 { z } else { 0.5 * (3.0 * z * z - 1.0) }
                },
            )
            .collect();
        let mut out = vec![0.0; pos.len()];
        lap.apply(&field, &mut out);

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for (i, (&approx, &u)) in out.iter().zip(field.iter()).enumerate() {
            assert!(approx.is_finite(), "Non-finite value at vertex {i}");
            let area = lap.vertex_area(i);
            let exact = eigenval * u;
            err_sq += area * (approx - exact).powi(2);
            norm_sq += area * exact.powi(2);
        }
        (err_sq / norm_sq).sqrt()
    }

    #[test]
    fn test_spherical_harmonics_l2_convergence_and_polar_stability() {
        let err_l1_coarse = spherical_harmonic_l2_error(24, 16, 1);
        let err_l1_fine = spherical_harmonic_l2_error(48, 32, 1);
        assert!(
            err_l1_fine < err_l1_coarse * 0.5,
            "l=1 failed to converge: {err_l1_coarse} -> {err_l1_fine}"
        );
        assert!(
            err_l1_fine < 0.02,
            "l=1 fine relative L2 error too high: {err_l1_fine}"
        );

        let err_l2_coarse = spherical_harmonic_l2_error(24, 16, 2);
        let err_l2_fine = spherical_harmonic_l2_error(48, 32, 2);
        assert!(
            err_l2_fine < err_l2_coarse * 0.5,
            "l=2 failed to converge: {err_l2_coarse} -> {err_l2_fine}"
        );
        assert!(
            err_l2_fine < 0.04,
            "l=2 fine relative L2 error too high: {err_l2_fine}"
        );
    }

    #[test]
    fn test_cotangent_laplacian_100_steps_zero_alloc_and_mass_conservation() {
        let (pos, tris) = build_welded_unit_sphere(24, 16);
        let lap = CotangentLaplacian::from_indexed(&pos, &tris);
        let mut u: Vec<f64> = pos.iter().map(|p| p[2] + 0.5 * p[0] * p[1]).collect();
        let mut next = vec![0.0; pos.len()];
        let mut mean_curv = vec![0.0; pos.len()];

        let initial_mass: f64 = u
            .iter()
            .enumerate()
            .map(|(i, &val)| lap.vertex_area(i) * val)
            .sum();

        let ((), _metrics) = VerifiedEngine::run_verified(|| {
            lap.compute_mean_curvature(&pos, &mut mean_curv);
            for _ in 0..100 {
                lap.step_heat_euler(&u, &mut next, 0.05, 1e-3);
                std::mem::swap(&mut u, &mut next);
            }
        });

        let final_mass: f64 = u
            .iter()
            .enumerate()
            .map(|(i, &val)| lap.vertex_area(i) * val)
            .sum();
        assert!((final_mass - initial_mass).abs() < 1e-10);
        assert!(
            mean_curv
                .iter()
                .all(|h| h.is_finite() && (*h - 1.0).abs() < 0.1)
        );
    }

    #[test]
    fn test_degenerate_obtuse_and_isolated_vertices_clamp_without_nan() {
        let pos = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.05, 0.0], // obtuse at vertex 2 (> 160 deg)
            [2.0, 0.0, 0.0],  // collinear with 0 and 1
            [9.0, 9.0, 9.0],  // isolated vertex
        ];
        let tris = [[0, 1, 2], [0, 1, 3], [0, 0, 1]];
        let lap = CotangentLaplacian::from_indexed(&pos, &tris);
        for v in 0..pos.len() {
            assert!(lap.vertex_area(v) >= 1e-12 && lap.vertex_area(v).is_finite());
        }
        let field = [1.0, 2.0, -0.5, 0.25, 3.0];
        let mut out = [0.0; 5];
        let mut next = [0.0; 5];
        let mut curv = [0.0; 5];
        lap.apply(&field, &mut out);
        lap.step_heat_euler(&field, &mut next, 0.1, 1e-3);
        lap.compute_mean_curvature(&pos, &mut curv);
        assert!(out.iter().all(|v| v.is_finite()) && out[4] == 0.0);
        assert!(next.iter().all(|v| v.is_finite()));
        assert!(curv.iter().all(|v| v.is_finite()));
    }
}
