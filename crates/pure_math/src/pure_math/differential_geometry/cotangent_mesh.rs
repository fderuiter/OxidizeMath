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
        debug_assert_eq!(self.weights.len(), self.col_idx.len());
        self.weights.len()
    }

    /// Characteristic mean edge length of the mesh (`>= 1e-6`).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_edge_length(&self) -> f64 {
        debug_assert!(self.mean_edge_length > 0.0);
        self.mean_edge_length
    }

    /// Clamped mixed Voronoi area $A_i \ge 10^{-12}$ associated with vertex `v`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn vertex_area(&self, v: usize) -> f64 {
        debug_assert!(v < self.areas.len());
        self.areas[v]
    }

    /// Slice of clamped mixed Voronoi vertex areas across all vertices.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn vertex_areas(&self) -> &[f64] {
        debug_assert_eq!(self.areas.len(), self.inv_areas.len());
        &self.areas
    }

    /// Inverse mixed Voronoi area $A_i^{-1}$ for vertex `v`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn inv_vertex_area(&self, v: usize) -> f64 {
        debug_assert!(v < self.inv_areas.len());
        self.inv_areas[v]
    }

    /// CSR neighbor vertex indices $\mathcal{N}(v)$ adjacent to `v`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn neighbors(&self, v: usize) -> &[usize] {
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        debug_assert!(start <= end && end <= self.col_idx.len());
        &self.col_idx[start..end]
    }

    /// CSR cotangent edge weights $w_{vj}$ aligned with [`Self::neighbors`].
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn row_weights(&self, v: usize) -> &[f64] {
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        debug_assert!(start <= end && end <= self.weights.len());
        &self.weights[start..end]
    }

    /// Evaluates $(\Delta_S u)_v$ at a single vertex `v` without heap allocation.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn apply_at(&self, field: &[f64], v: usize) -> f64 {
        debug_assert!(v < field.len());
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

    /// Fallible companion to [`apply`](Self::apply).
    #[verified_engine::verified]
    pub fn try_apply(
        &self,
        field: &[f64],
        out: &mut [f64],
    ) -> core::result::Result<(), math_commons::error::MathError> {
        let n = self.num_vertices();
        __ensure_eq_math(&(field.len()), &(n), "apply: equality invariant violated")?;
        __ensure_eq_math(&(out.len()), &(n), "apply: equality invariant violated")?;
        for (v, dst) in out.iter_mut().enumerate().take(n) {
            *dst = self.apply_at(field, v);
        }
        debug_assert_eq!(out.len(), n);
        Ok(())
    }

    /// Evaluates $\Delta_S u$ across all vertices into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn apply(&self, field: &[f64], out: &mut [f64]) {
        self.try_apply(field, out)
            .expect("apply preconditions must hold");
    }

    /// Fallible companion to [`step_heat_euler`](Self::step_heat_euler).
    #[verified_engine::verified]
    pub fn try_step_heat_euler(
        &self,
        field: &[f64],
        next: &mut [f64],
        diffusivity: f64,
        dt: f64,
    ) -> core::result::Result<(), math_commons::error::MathError> {
        let n = self.num_vertices();
        __ensure_eq_math(
            &(field.len()),
            &(n),
            "step_heat_euler: equality invariant violated",
        )?;
        __ensure_eq_math(
            &(next.len()),
            &(n),
            "step_heat_euler: equality invariant violated",
        )?;
        let scale = dt * diffusivity;
        for v in 0..n {
            next[v] = field[v] + scale * self.apply_at(field, v);
        }
        debug_assert_eq!(next.len(), n);
        Ok(())
    }

    /// Advances explicit heat diffusion $u^{k+1} = u^k + \Delta t \, \alpha \, \Delta_S u^k$ into `next`.
    #[verified_engine::verified]
    pub fn step_heat_euler(&self, field: &[f64], next: &mut [f64], diffusivity: f64, dt: f64) {
        self.try_step_heat_euler(field, next, diffusivity, dt)
            .expect("step_heat_euler preconditions must hold");
    }

    /// Fallible companion to [`compute_mean_curvature`](Self::compute_mean_curvature).
    #[verified_engine::verified]
    pub fn try_compute_mean_curvature(
        &self,
        positions: &[[f64; 3]],
        out: &mut [f64],
    ) -> core::result::Result<(), math_commons::error::MathError> {
        let n = self.num_vertices();
        __ensure_eq_math(
            &(positions.len()),
            &(n),
            "compute_mean_curvature: equality invariant violated",
        )?;
        __ensure_eq_math(
            &(out.len()),
            &(n),
            "compute_mean_curvature: equality invariant violated",
        )?;
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
        debug_assert_eq!(out.len(), n);
        Ok(())
    }

    /// Computes unsigned discrete mean curvature $|H|_i = \frac{1}{2} \| (\Delta_S \mathbf{x})_i \|$ into `out`.
    #[verified_engine::verified]
    pub fn compute_mean_curvature(&self, positions: &[[f64; 3]], out: &mut [f64]) {
        self.try_compute_mean_curvature(positions, out)
            .expect("compute_mean_curvature preconditions must hold");
    }
}

#[inline]
fn __ensure_eq_math<A: PartialEq<B> + ?Sized, B: ?Sized>(
    left: &A,
    right: &B,
    reason: &'static str,
) -> core::result::Result<(), math_commons::error::MathError> {
    if left == right {
        Ok(())
    } else {
        Err(math_commons::error::MathError::NumericalError { reason })
    }
}

#[cfg(test)]
mod tests;
