//! Preallocated zero-allocation workspace for Vietoris-Rips filtration and Z₂ 1-homology.

use super::core::Point2D;
use super::persistence::PersistenceInterval;
use super::space3d::Point3D;
use crate::error::TdaError;

/// Summary of simplex counts and Betti numbers at a single filtration radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiltrationStepSummary {
    /// Connected components (`β₀`).
    pub beta0: usize,
    /// Independent 1-cycles over `ℤ₂` (`β₁`).
    pub beta1: usize,
    /// Number of 0-simplices (vertices).
    pub vertices: usize,
    /// Number of 1-simplices (edges).
    pub edges: usize,
    /// Number of 2-simplices (triangles).
    pub triangles: usize,
}

/// Preallocated scratch buffers enabling zero-allocation Vietoris-Rips filtration sweeps
/// and `ℤ₂` boundary-matrix column reduction up to `max_points` vertices.
#[derive(Debug, Clone)]
pub struct TdaFiltrationWorkspace {
    max_points: usize,
    dist: Vec<Vec<f64>>,
    uf_parent: Vec<usize>,
    uf_root_seen: Vec<u8>,
    edges: Vec<[usize; 2]>,
    edge_lookup: Vec<Vec<usize>>,
    triangles: Vec<[usize; 3]>,
    tri_boundaries: Vec<[usize; 3]>,
    pivot_for_row: Vec<usize>,
    basis_words: Vec<Vec<u64>>,
    col_scratch: Vec<u64>,
    component_births: Vec<f64>,
    hole_births: Vec<f64>,
}

impl TdaFiltrationWorkspace {
    /// Allocates reusable filtration and `ℤ₂` reduction buffers for up to `max_points` vertices.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::tda::{Point3D, TdaFiltrationWorkspace};
    ///
    /// let mut ws = TdaFiltrationWorkspace::with_capacity(16);
    /// let pts = [
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 1.0, 0.0),
    ///     Point3D::new(0.0, 1.0, 0.0),
    /// ];
    /// ws.load_points_3d(&pts).unwrap();
    /// let step = ws.evaluate_radius(pts.len(), 1.1).unwrap();
    /// assert_eq!((step.beta0, step.beta1), (1, 1));
    /// ```
    #[verified_engine::verified]
    pub fn with_capacity(max_points: usize) -> Self {
        let cap = max_points.max(1);
        let max_edges = cap.saturating_mul(cap.saturating_sub(1)) / 2;
        let max_tris = cap
            .saturating_mul(cap.saturating_sub(1))
            .saturating_mul(cap.saturating_sub(2))
            / 6;
        let max_words = max_edges.div_ceil(64).max(1);
        Self {
            max_points: cap,
            dist: vec![vec![0.0; cap]; cap],
            uf_parent: vec![0; cap],
            uf_root_seen: vec![0; cap],
            edges: Vec::with_capacity(max_edges),
            edge_lookup: vec![vec![usize::MAX; cap]; cap],
            triangles: Vec::with_capacity(max_tris),
            tri_boundaries: Vec::with_capacity(max_tris),
            pivot_for_row: vec![usize::MAX; max_edges.max(1)],
            basis_words: vec![vec![0; max_words]; max_edges.max(1)],
            col_scratch: vec![0; max_words],
            component_births: Vec::with_capacity(cap),
            hole_births: Vec::with_capacity(max_edges.max(1)),
        }
    }

    /// Returns the maximum vertex count supported without reallocation.
    #[verified_engine::verified]
    pub fn capacity(&self) -> usize {
        self.max_points
    }

    /// Computes pairwise distances for a 2D point slice into the preallocated buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError`] if `points` is empty, exceeds capacity, or has non-finite values.
    #[verified_engine::verified]
    pub fn load_points_2d(&mut self, points: &[Point2D]) -> Result<(), TdaError> {
        let n = points.len();
        if n == 0 {
            return Err(TdaError::EmptyPointCloud);
        }
        if n > self.max_points {
            return Err(TdaError::MatrixError {
                reason: "Point count exceeds workspace capacity".to_string(),
            });
        }
        for (i, pi) in points.iter().enumerate() {
            if !pi.x.is_finite() || !pi.y.is_finite() {
                return Err(TdaError::MatrixError {
                    reason: "Non-finite 2D coordinate".to_string(),
                });
            }
            self.dist[i][i] = 0.0;
            for (j, pj) in points.iter().enumerate().skip(i + 1) {
                let dx = pi.x - pj.x;
                let dy = pi.y - pj.y;
                let d = (dx * dx + dy * dy).sqrt();
                self.dist[i][j] = d;
                self.dist[j][i] = d;
            }
        }
        Ok(())
    }

    /// Computes pairwise distances for a 3D point slice into the preallocated buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError`] if `points` is empty, exceeds capacity, or has non-finite values.
    #[verified_engine::verified]
    pub fn load_points_3d(&mut self, points: &[Point3D]) -> Result<(), TdaError> {
        let n = points.len();
        if n == 0 {
            return Err(TdaError::EmptyPointCloud);
        }
        if n > self.max_points {
            return Err(TdaError::MatrixError {
                reason: "Point count exceeds workspace capacity".to_string(),
            });
        }
        for (i, pi) in points.iter().enumerate() {
            if !pi.x.is_finite() || !pi.y.is_finite() || !pi.z.is_finite() {
                return Err(TdaError::MatrixError {
                    reason: "Non-finite 3D coordinate".to_string(),
                });
            }
            self.dist[i][i] = 0.0;
            for (j, pj) in points.iter().enumerate().skip(i + 1) {
                let dx = pi.x - pj.x;
                let dy = pi.y - pj.y;
                let dz = pi.z - pj.z;
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                self.dist[i][j] = d;
                self.dist[j][i] = d;
            }
        }
        Ok(())
    }

    /// Returns the precomputed pairwise distance `d(i, j)` for an `n`-point cloud.
    #[verified_engine::verified]
    pub fn distance(&self, n: usize, i: usize, j: usize) -> f64 {
        debug_assert!(i < n && j < n);
        self.dist[i][j]
    }

    /// Returns the canonical `[i, j]` 1-simplices from the most recent [`Self::evaluate_radius`] call.
    #[verified_engine::verified]
    pub fn active_edges(&self) -> &[[usize; 2]] {
        &self.edges
    }

    /// Returns the canonical `[i, j, k]` 2-simplices from the most recent [`Self::evaluate_radius`] call.
    #[verified_engine::verified]
    pub fn active_triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }

    /// Evaluates simplex counts and exact `(β₀, β₁)` over `ℤ₂` at `radius` with zero allocations.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::InvalidRadius`] when `radius` is negative or non-finite.
    #[verified_engine::verified]
    pub fn evaluate_radius(
        &mut self,
        n: usize,
        radius: f64,
    ) -> Result<FiltrationStepSummary, TdaError> {
        if radius < 0.0 || !radius.is_finite() {
            return Err(TdaError::InvalidRadius { value: radius });
        }
        if n == 0 || n > self.max_points {
            return Err(TdaError::EmptyPointCloud);
        }
        let beta0 = self.build_edges_and_components(n, radius);
        let e = self.edges.len();
        let cycles = (e + beta0).saturating_sub(n);
        self.collect_triangle_boundaries(n);
        let triangles = self.tri_boundaries.len();
        let rank_d2 = self.reduce_collected_boundaries(cycles);
        Ok(FiltrationStepSummary {
            beta0,
            beta1: cycles.saturating_sub(rank_d2),
            vertices: n,
            edges: e,
            triangles,
        })
    }

    #[verified_engine::verified]
    fn build_edges_and_components(&mut self, n: usize, radius: f64) -> usize {
        for i in 0..n {
            self.uf_parent[i] = i;
            self.uf_root_seen[i] = 0;
        }
        self.edges.clear();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.dist[i][j] <= radius {
                    let idx = self.edges.len();
                    self.edges.push([i, j]);
                    self.edge_lookup[i][j] = idx;
                    let (mut rx, mut ry) = (i, j);
                    while self.uf_parent[rx] != rx {
                        self.uf_parent[rx] = self.uf_parent[self.uf_parent[rx]];
                        rx = self.uf_parent[rx];
                    }
                    while self.uf_parent[ry] != ry {
                        self.uf_parent[ry] = self.uf_parent[self.uf_parent[ry]];
                        ry = self.uf_parent[ry];
                    }
                    if rx != ry {
                        self.uf_parent[ry] = rx;
                    }
                } else {
                    self.edge_lookup[i][j] = usize::MAX;
                }
            }
        }
        let mut components = 0usize;
        for i in 0..n {
            let mut root = i;
            while self.uf_parent[root] != root {
                root = self.uf_parent[root];
            }
            if self.uf_root_seen[root] == 0 {
                self.uf_root_seen[root] = 1;
                components += 1;
            }
        }
        components
    }

    #[verified_engine::verified]
    fn collect_triangle_boundaries(&mut self, n: usize) {
        self.triangles.clear();
        self.tri_boundaries.clear();
        if self.edges.len() < 3 {
            return;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let e_ij = self.edge_lookup[i][j];
                if e_ij == usize::MAX {
                    continue;
                }
                for k in (j + 1)..n {
                    let e_ik = self.edge_lookup[i][k];
                    let e_jk = self.edge_lookup[j][k];
                    if e_ik != usize::MAX && e_jk != usize::MAX {
                        self.triangles.push([i, j, k]);
                        self.tri_boundaries.push([e_ij, e_ik, e_jk]);
                    }
                }
            }
        }
    }

    #[verified_engine::verified]
    fn reduce_collected_boundaries(&mut self, max_rank: usize) -> usize {
        let e = self.edges.len();
        if max_rank == 0 || self.tri_boundaries.is_empty() {
            return 0;
        }
        let words = e.div_ceil(64).max(1);
        self.pivot_for_row[..e].fill(usize::MAX);
        let mut rank = 0usize;
        for &[r0, r1, r2] in &self.tri_boundaries {
            if rank >= max_rank {
                break;
            }
            if self.pivot_for_row[r2] == usize::MAX {
                self.pivot_for_row[r2] = rank;
                let row_words = &mut self.basis_words[rank][..words];
                row_words.fill(0);
                row_words[r0 >> 6] ^= 1u64 << (r0 & 63);
                row_words[r1 >> 6] ^= 1u64 << (r1 & 63);
                row_words[r2 >> 6] ^= 1u64 << (r2 & 63);
                rank += 1;
                continue;
            }
            self.col_scratch[..words].fill(0);
            self.col_scratch[r0 >> 6] ^= 1u64 << (r0 & 63);
            self.col_scratch[r1 >> 6] ^= 1u64 << (r1 & 63);
            self.col_scratch[r2 >> 6] ^= 1u64 << (r2 & 63);
            if reduce_high_pivots(
                words,
                rank,
                &mut self.col_scratch,
                &mut self.pivot_for_row,
                &mut self.basis_words,
            ) {
                rank += 1;
            }
        }
        rank
    }

    /// Sweeps `radii` and writes `PersistenceInterval`s into `out` with zero heap allocations
    /// when `out` has pre-reserved capacity.
    ///
    /// # Errors
    ///
    /// Returns [`TdaError::InvalidRadius`] if `radii` is empty or contains invalid values.
    #[verified_engine::verified]
    pub fn compute_persistence_into(
        &mut self,
        n: usize,
        radii: &[f64],
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        if radii.is_empty() {
            return Err(TdaError::InvalidRadius { value: 0.0 });
        }
        out.clear();
        self.component_births.clear();
        self.hole_births.clear();
        let mut prev_b0 = 0usize;
        let mut prev_b1 = 0usize;
        for &r in radii {
            let step = self.evaluate_radius(n, r)?;
            update_births_and_deaths(&mut self.component_births, prev_b0, step.beta0, r, 0, out)?;
            update_births_and_deaths(&mut self.hole_births, prev_b1, step.beta1, r, 1, out)?;
            prev_b0 = step.beta0;
            prev_b1 = step.beta1;
        }
        let final_r = radii[radii.len() - 1];
        for &birth in &self.component_births {
            if birth < final_r {
                out.push(PersistenceInterval::new(birth, final_r, 0)?);
            }
        }
        for &birth in &self.hole_births {
            if birth < final_r {
                out.push(PersistenceInterval::new(birth, final_r, 1)?);
            }
        }
        Ok(())
    }
}

#[verified_engine::verified]
fn reduce_high_pivots(
    words: usize,
    basis_idx: usize,
    col_scratch: &mut [u64],
    pivot_for_row: &mut [usize],
    basis_words: &mut [Vec<u64>],
) -> bool {
    for wi in (0..words).rev() {
        while col_scratch[wi] != 0 {
            let bit = 63 - col_scratch[wi].leading_zeros() as usize;
            let pivot_row = (wi << 6) | bit;
            let existing = pivot_for_row[pivot_row];
            if existing == usize::MAX {
                pivot_for_row[pivot_row] = basis_idx;
                basis_words[basis_idx][..words].copy_from_slice(&col_scratch[..words]);
                return true;
            }
            let existing_words = &basis_words[existing];
            for w in 0..=wi {
                col_scratch[w] ^= existing_words[w];
            }
        }
    }
    false
}

#[verified_engine::verified]
fn update_births_and_deaths(
    births: &mut Vec<f64>,
    prev_betti: usize,
    curr_betti: usize,
    radius: f64,
    dim: usize,
    out: &mut Vec<PersistenceInterval>,
) -> Result<(), TdaError> {
    if curr_betti > prev_betti {
        for _ in 0..(curr_betti - prev_betti) {
            births.push(radius);
        }
    } else if curr_betti < prev_betti {
        for _ in 0..(prev_betti - curr_betti) {
            if let Some(birth) = births.pop()
                && birth < radius
            {
                out.push(PersistenceInterval::new(birth, radius, dim)?);
            }
        }
    }
    Ok(())
}
