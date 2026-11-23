//! Preallocated zero-allocation workspace for Vietoris-Rips filtration and Z₂ 1-homology.

use super::boundary::FilteredEdge;
use super::core::Point2D;
use super::filtration::{append_edge_triangles_and_reduce, prepare_sorted_workspace_edges};
use super::homology::{UnionFind, betti_1_from_counts};
use super::persistence::PersistenceInterval;
use super::space3d::Point3D;
use crate::error::TdaError;

const DEFAULT_RADII_SCRATCH_CAP: usize = 512;

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
    sorted_edges: Vec<FilteredEdge>,
    adj_words: Vec<Vec<u64>>,
    uf: UnionFind,
    edges: Vec<[usize; 2]>,
    edge_lookup: Vec<Vec<usize>>,
    triangles: Vec<[usize; 3]>,
    pivot_for_row: Vec<usize>,
    basis_words: Vec<Vec<u64>>,
    col_scratch: Vec<u64>,
    component_births: Vec<f64>,
    hole_births: Vec<f64>,
    radii_scratch: Vec<f64>,
    active_n: usize,
    edges_prepared: bool,
    edge_cursor: usize,
    basis_rank: usize,
    prev_radius: Option<f64>,
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
        let max_edge_words = max_edges.div_ceil(64).max(1);
        let max_vertex_words = cap.div_ceil(64).max(1);
        Self {
            max_points: cap,
            dist: vec![vec![0.0; cap]; cap],
            sorted_edges: Vec::with_capacity(max_edges),
            adj_words: vec![vec![0; max_vertex_words]; cap],
            uf: UnionFind::new(cap),
            edges: Vec::with_capacity(max_edges),
            edge_lookup: vec![vec![usize::MAX; cap]; cap],
            triangles: Vec::with_capacity(max_tris),
            pivot_for_row: vec![usize::MAX; max_edges.max(1)],
            basis_words: vec![vec![0; max_edge_words]; max_edges.max(1)],
            col_scratch: vec![0; max_edge_words],
            component_births: Vec::with_capacity(cap),
            hole_births: Vec::with_capacity(max_edges.max(1)),
            radii_scratch: Vec::with_capacity(DEFAULT_RADII_SCRATCH_CAP),
            active_n: 0,
            edges_prepared: false,
            edge_cursor: 0,
            basis_rank: 0,
            prev_radius: None,
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
        self.prepare_edges_for_n(n);
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
        self.prepare_edges_for_n(n);
        Ok(())
    }

    #[verified_engine::verified]
    fn prepare_edges_for_n(&mut self, n: usize) {
        prepare_sorted_workspace_edges(&self.dist, n, &mut self.sorted_edges);
        self.active_n = n;
        self.edges_prepared = true;
        self.prev_radius = None;
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
    /// Advances incrementally from the previous radius when `radius >= prev_radius`, and resets
    /// in-place when `radius < prev_radius`.
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
        if !self.edges_prepared || self.active_n != n {
            self.prepare_edges_for_n(n);
        }
        if self.prev_radius.is_none_or(|prev| radius < prev) {
            self.reset_filtration_state(n);
        }

        self.advance_filtration_to(n, radius);
        self.prev_radius = Some(radius);

        let beta0 = self.uf.count_components();
        let e = self.edges.len();
        Ok(FiltrationStepSummary {
            beta0,
            beta1: betti_1_from_counts(n, e, beta0, self.basis_rank),
            vertices: n,
            edges: e,
            triangles: self.triangles.len(),
        })
    }

    #[verified_engine::verified]
    fn reset_filtration_state(&mut self, n: usize) {
        let n_words = n.div_ceil(64).max(1);
        let total_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
        self.uf.reset_prefix(n);
        for row in self.adj_words.iter_mut().take(n) {
            row[..n_words].fill(0);
        }
        self.edges.clear();
        self.triangles.clear();
        self.pivot_for_row[..total_edges.max(1)].fill(usize::MAX);
        self.edge_cursor = 0;
        self.basis_rank = 0;
    }

    #[verified_engine::verified]
    fn advance_filtration_to(&mut self, n: usize, radius: f64) {
        let n_words = n.div_ceil(64).max(1);
        let total_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
        let edge_words = total_edges.div_ceil(64).max(1);

        while self.edge_cursor < self.sorted_edges.len() {
            let edge = self.sorted_edges[self.edge_cursor];
            if edge.diameter > radius {
                break;
            }
            self.edge_cursor += 1;
            let u = edge.u as usize;
            let v = edge.v as usize;
            let e_uv = self.edges.len();
            self.edges.push([u, v]);
            self.edge_lookup[u][v] = e_uv;
            self.edge_lookup[v][u] = e_uv;

            let merged = self.uf.union(u, v);
            if !merged {
                let curr_cycles =
                    betti_1_from_counts(n, self.edges.len(), self.uf.count_components(), 0);
                append_edge_triangles_and_reduce(
                    u,
                    v,
                    e_uv,
                    n_words,
                    edge_words,
                    curr_cycles,
                    &self.adj_words,
                    &self.edge_lookup,
                    &mut self.triangles,
                    &mut self.col_scratch,
                    &mut self.pivot_for_row,
                    &mut self.basis_words,
                    &mut self.basis_rank,
                );
            }
            self.adj_words[u][v >> 6] |= 1_u64 << (v & 63);
            self.adj_words[v][u >> 6] |= 1_u64 << (u & 63);
        }
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
        let mut sorted = true;
        for (idx, &r) in radii.iter().enumerate() {
            if r < 0.0 || !r.is_finite() {
                return Err(TdaError::InvalidRadius { value: r });
            }
            if idx > 0 && r < radii[idx - 1] {
                sorted = false;
            }
        }
        if n == 0 || n > self.max_points {
            return Err(TdaError::EmptyPointCloud);
        }

        out.clear();
        self.component_births.clear();
        self.hole_births.clear();
        self.prev_radius = None;

        if sorted {
            self.sweep_monotonic_radii(n, radii, out)
        } else if radii.len() <= self.radii_scratch.capacity() {
            self.radii_scratch.clear();
            self.radii_scratch.extend_from_slice(radii);
            self.radii_scratch.sort_unstable_by(f64::total_cmp);
            self.sweep_scratch_radii(n, out)
        } else {
            let mut owned = radii.to_vec();
            owned.sort_unstable_by(f64::total_cmp);
            self.sweep_monotonic_radii(n, &owned, out)
        }
    }

    #[verified_engine::verified]
    fn sweep_monotonic_radii(
        &mut self,
        n: usize,
        radii: &[f64],
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        let mut prev_b0 = 0usize;
        let mut prev_b1 = 0usize;
        for &r in radii {
            let step = self.evaluate_radius(n, r)?;
            update_births_and_deaths(&mut self.component_births, prev_b0, step.beta0, r, 0, out)?;
            update_births_and_deaths(&mut self.hole_births, prev_b1, step.beta1, r, 1, out)?;
            prev_b0 = step.beta0;
            prev_b1 = step.beta1;
        }
        flush_essential_births(
            &self.component_births,
            &self.hole_births,
            radii[radii.len() - 1],
            out,
        )
    }

    #[verified_engine::verified]
    fn sweep_scratch_radii(
        &mut self,
        n: usize,
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        let len = self.radii_scratch.len();
        let mut prev_b0 = 0usize;
        let mut prev_b1 = 0usize;
        for idx in 0..len {
            let r = self.radii_scratch[idx];
            let step = self.evaluate_radius(n, r)?;
            update_births_and_deaths(&mut self.component_births, prev_b0, step.beta0, r, 0, out)?;
            update_births_and_deaths(&mut self.hole_births, prev_b1, step.beta1, r, 1, out)?;
            prev_b0 = step.beta0;
            prev_b1 = step.beta1;
        }
        flush_essential_births(
            &self.component_births,
            &self.hole_births,
            self.radii_scratch[len - 1],
            out,
        )
    }
}

#[verified_engine::verified]
fn flush_essential_births(
    comp_births: &[f64],
    hole_births: &[f64],
    final_r: f64,
    out: &mut Vec<PersistenceInterval>,
) -> Result<(), TdaError> {
    for &birth in comp_births {
        if birth < final_r {
            out.push(PersistenceInterval::new(birth, final_r, 0)?);
        }
    }
    for &birth in hole_births {
        if birth < final_r {
            out.push(PersistenceInterval::new(birth, final_r, 1)?);
        }
    }
    Ok(())
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

// [cite:clinical_trials_statistics]
