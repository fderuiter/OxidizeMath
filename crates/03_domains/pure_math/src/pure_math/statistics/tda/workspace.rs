//! Preallocated zero-allocation workspace for Vietoris-Rips filtration and Z₂ 1-homology.

use super::core::Point2D;
use super::filtration::{
    IncrementalReductionCtx, append_edge_triangles_and_reduce, prepare_sorted_workspace_edges,
};
use super::homology::{betti_1_from_counts, uf_reset_slices, uf_union_slices};
use super::persistence::{PersistenceInterval, flush_essential_births, update_births_and_deaths};
use super::space3d::Point3D;
use super::workspace_layout::{TdaRegions, allocate_tda_arena};
use crate::error::TdaError;
use math_memory::OwnedByteArena;

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

/// Preallocated scratch buffers in a single [`OwnedByteArena`] enabling zero-allocation
/// Vietoris-Rips filtration sweeps and `ℤ₂` boundary-matrix column reduction up to `max_points`.
#[derive(Debug, Clone)]
pub struct TdaFiltrationWorkspace {
    max_points: usize,
    max_vertex_words: usize,
    max_edge_words: usize,
    arena: OwnedByteArena,
    regions: TdaRegions,
    sorted_edges_len: usize,
    edges_len: usize,
    triangles_len: usize,
    uf_components: usize,
    comp_births_len: usize,
    hole_births_len: usize,
    radii_scratch_len: usize,
    active_n: usize,
    edges_prepared: bool,
    edge_cursor: usize,
    basis_rank: usize,
    prev_radius: Option<f64>,
}

impl TdaFiltrationWorkspace {
    /// Allocates reusable filtration and `ℤ₂` reduction buffers in one [`OwnedByteArena`] for up to `max_points` vertices.
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
        let alloc = allocate_tda_arena(max_points);
        Self {
            max_points: alloc.cap,
            max_vertex_words: alloc.max_vertex_words,
            max_edge_words: alloc.max_edge_words,
            arena: alloc.arena,
            regions: alloc.regions,
            sorted_edges_len: 0,
            edges_len: 0,
            triangles_len: 0,
            uf_components: alloc.cap,
            comp_births_len: 0,
            hole_births_len: 0,
            radii_scratch_len: 0,
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
        let stride = self.max_points;
        let dist = self.arena.region_slice_mut(self.regions.dist);
        for (i, pi) in points.iter().enumerate() {
            if !pi.x.is_finite() || !pi.y.is_finite() {
                return Err(TdaError::MatrixError {
                    reason: "Non-finite 2D coordinate".to_string(),
                });
            }
            dist[math_commons::math_kernel::types::flatten_2d_index(i, i, stride)] = 0.0;
            for (j, pj) in points.iter().enumerate().skip(i + 1) {
                let dx = pi.x - pj.x;
                let dy = pi.y - pj.y;
                let d = (dx * dx + dy * dy).sqrt();
                dist[math_commons::math_kernel::types::flatten_2d_index(j, i, stride)] = d;
                dist[math_commons::math_kernel::types::flatten_2d_index(i, j, stride)] = d;
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
        let stride = self.max_points;
        let dist = self.arena.region_slice_mut(self.regions.dist);
        for (i, pi) in points.iter().enumerate() {
            if !pi.x.is_finite() || !pi.y.is_finite() || !pi.z.is_finite() {
                return Err(TdaError::MatrixError {
                    reason: "Non-finite 3D coordinate".to_string(),
                });
            }
            dist[math_commons::math_kernel::types::flatten_2d_index(i, i, stride)] = 0.0;
            for (j, pj) in points.iter().enumerate().skip(i + 1) {
                let dx = pi.x - pj.x;
                let dy = pi.y - pj.y;
                let dz = pi.z - pj.z;
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                dist[math_commons::math_kernel::types::flatten_2d_index(j, i, stride)] = d;
                dist[math_commons::math_kernel::types::flatten_2d_index(i, j, stride)] = d;
            }
        }
        self.prepare_edges_for_n(n);
        Ok(())
    }

    #[verified_engine::verified]
    fn prepare_edges_for_n(&mut self, n: usize) {
        let stride = self.max_points;
        let d_reg = self.regions.dist;
        let se_reg = self.regions.sorted_edges;
        let mut cur = self.arena.raw_cursor();
        let dist = cur.region_mut(d_reg);
        let sorted_edges = cur.region_mut(se_reg);
        self.sorted_edges_len = prepare_sorted_workspace_edges(dist, stride, n, sorted_edges);
        self.active_n = n;
        self.edges_prepared = true;
        self.prev_radius = None;
    }

    /// Returns the precomputed pairwise distance `d(i, j)` for an `n`-point cloud.
    #[verified_engine::verified]
    pub fn distance(&self, n: usize, i: usize, j: usize) -> f64 {
        debug_assert!(i < n && j < n);
        self.arena.region_slice(self.regions.dist)[i * self.max_points + j]
    }

    /// Returns the canonical `[i, j]` 1-simplices from the most recent [`Self::evaluate_radius`] call.
    #[verified_engine::verified]
    pub fn active_edges(&self) -> &[[usize; 2]] {
        &self.arena.region_slice(self.regions.edges)[..self.edges_len]
    }

    /// Returns the canonical `[i, j, k]` 2-simplices from the most recent [`Self::evaluate_radius`] call.
    #[verified_engine::verified]
    pub fn active_triangles(&self) -> &[[usize; 3]] {
        &self.arena.region_slice(self.regions.triangles)[..self.triangles_len]
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
        if !self.edges_prepared || self.active_n != n {
            self.prepare_edges_for_n(n);
        }
        if self.prev_radius.is_none_or(|prev| radius < prev) {
            self.reset_filtration_state(n);
        }

        self.advance_filtration_to(n, radius);
        self.prev_radius = Some(radius);

        let beta0 = self.uf_components;
        let e = self.edges_len;
        Ok(FiltrationStepSummary {
            beta0,
            beta1: betti_1_from_counts(n, e, beta0, self.basis_rank),
            vertices: n,
            edges: e,
            triangles: self.triangles_len,
        })
    }

    #[verified_engine::verified]
    fn reset_filtration_state(&mut self, n: usize) {
        let n_words = n.div_ceil(64).max(1);
        let adj_stride = self.max_vertex_words;
        let total_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
        let r = self.regions;
        let mut cur = self.arena.raw_cursor();
        let adj_words = cur.region_mut(r.adj_words);
        for u in 0..n {
            adj_words[u * adj_stride..u * adj_stride + n_words].fill(0);
        }
        let uf_p = cur.region_mut(r.uf_parent);
        let uf_r = cur.region_mut(r.uf_rank);
        self.uf_components = uf_reset_slices(uf_p, uf_r, n);
        let pivot_for_row = cur.region_mut(r.pivot_for_row);
        pivot_for_row[..total_edges.max(1)].fill(usize::MAX);
        self.edges_len = 0;
        self.triangles_len = 0;
        self.edge_cursor = 0;
        self.basis_rank = 0;
    }

    #[verified_engine::verified]
    fn advance_filtration_to(&mut self, n: usize, radius: f64) {
        let n_words = n.div_ceil(64).max(1);
        let total_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
        let edge_words = total_edges.div_ceil(64).max(1);
        let adj_stride = self.max_vertex_words;
        let basis_stride = self.max_edge_words;
        let lookup_stride = self.max_points;
        let r = self.regions;

        let mut cur = self.arena.raw_cursor();
        let sorted_edges = &cur.region_mut(r.sorted_edges)[..self.sorted_edges_len];
        let adj_words = cur.region_mut(r.adj_words);
        let uf_parent = cur.region_mut(r.uf_parent);
        let uf_rank = cur.region_mut(r.uf_rank);
        let edges = cur.region_mut(r.edges);
        let edge_lookup = cur.region_mut(r.edge_lookup);
        let triangles = cur.region_mut(r.triangles);
        let pivot_for_row = cur.region_mut(r.pivot_for_row);
        let basis_words = cur.region_mut(r.basis_words);
        let col_scratch = cur.region_mut(r.col_scratch);

        while self.edge_cursor < sorted_edges.len() {
            let [diam_bits, uv_packed] = sorted_edges[self.edge_cursor];
            if f64::from_bits(diam_bits) > radius {
                break;
            }
            self.edge_cursor += 1;
            let u = (uv_packed >> 32) as usize;
            let v = (uv_packed & 0xFFFF_FFFF) as usize;
            let e_uv = self.edges_len;
            edges[e_uv] = [u, v];
            self.edges_len += 1;
            edge_lookup[math_commons::math_kernel::types::flatten_2d_index(v, u, lookup_stride)] =
                e_uv;
            edge_lookup[math_commons::math_kernel::types::flatten_2d_index(u, v, lookup_stride)] =
                e_uv;

            let merged = uf_union_slices(uf_parent, uf_rank, &mut self.uf_components, u, v);
            if !merged {
                let curr_cycles = betti_1_from_counts(n, self.edges_len, self.uf_components, 0);
                let mut ctx = IncrementalReductionCtx {
                    n_words,
                    adj_stride,
                    edge_words,
                    basis_stride,
                    lookup_stride,
                    curr_cycles,
                    adj_words,
                    edge_lookup,
                    triangles,
                    triangles_len: &mut self.triangles_len,
                    col_scratch,
                    pivot_for_row,
                    basis_words,
                    basis_rank: &mut self.basis_rank,
                };
                append_edge_triangles_and_reduce(u, v, e_uv, &mut ctx);
            }
            adj_words[u * adj_stride + (v >> 6)] |= 1_u64 << (v & 63);
            adj_words[v * adj_stride + (u >> 6)] |= 1_u64 << (u & 63);
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
        self.comp_births_len = 0;
        self.hole_births_len = 0;
        self.prev_radius = None;

        if sorted {
            self.sweep_monotonic_radii(n, radii, out)
        } else if radii.len() <= self.regions.radii_scratch.len() {
            let len = radii.len();
            let scratch = &mut self.arena.region_slice_mut(self.regions.radii_scratch)[..len];
            scratch.copy_from_slice(radii);
            scratch.sort_unstable_by(f64::total_cmp);
            self.radii_scratch_len = len;
            self.sweep_scratch_radii(n, out)
        } else {
            let mut owned = radii.to_vec();
            owned.sort_unstable_by(f64::total_cmp);
            self.sweep_monotonic_radii(n, &owned, out)
        }
    }

    #[verified_engine::verified]
    fn record_step_births_deaths(
        &mut self,
        prev_b0: usize,
        prev_b1: usize,
        step: FiltrationStepSummary,
        r: f64,
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        let cb_reg = self.regions.component_births;
        let hb_reg = self.regions.hole_births;
        let mut cur = self.arena.raw_cursor();
        let comp_births = cur.region_mut(cb_reg);
        let hole_births = cur.region_mut(hb_reg);
        update_births_and_deaths(
            comp_births,
            &mut self.comp_births_len,
            prev_b0,
            step.beta0,
            r,
            0,
            out,
        )?;
        update_births_and_deaths(
            hole_births,
            &mut self.hole_births_len,
            prev_b1,
            step.beta1,
            r,
            1,
            out,
        )
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
            self.record_step_births_deaths(prev_b0, prev_b1, step, r, out)?;
            prev_b0 = step.beta0;
            prev_b1 = step.beta1;
        }
        let comp = &self.arena.region_slice(self.regions.component_births)[..self.comp_births_len];
        let hole = &self.arena.region_slice(self.regions.hole_births)[..self.hole_births_len];
        flush_essential_births(comp, hole, radii[radii.len() - 1], out)
    }

    #[verified_engine::verified]
    fn sweep_scratch_radii(
        &mut self,
        n: usize,
        out: &mut Vec<PersistenceInterval>,
    ) -> Result<(), TdaError> {
        let len = self.radii_scratch_len;
        let mut prev_b0 = 0usize;
        let mut prev_b1 = 0usize;
        for idx in 0..len {
            let r = self.arena.region_slice(self.regions.radii_scratch)[idx];
            let step = self.evaluate_radius(n, r)?;
            self.record_step_births_deaths(prev_b0, prev_b1, step, r, out)?;
            prev_b0 = step.beta0;
            prev_b1 = step.beta1;
        }
        let final_r = self.arena.region_slice(self.regions.radii_scratch)[len - 1];
        let comp = &self.arena.region_slice(self.regions.component_births)[..self.comp_births_len];
        let hole = &self.arena.region_slice(self.regions.hole_births)[..self.hole_births_len];
        flush_essential_births(comp, hole, final_r, out)
    }
}

// [cite:clinical_trials_statistics]
