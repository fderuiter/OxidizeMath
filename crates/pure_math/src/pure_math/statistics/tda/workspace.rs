//! Preallocated zero-allocation workspace for Vietoris-Rips filtration and Z₂ 1-homology.

use super::core::Point2D;
use super::filtration::{
    EdgeTriangleCtx, append_edge_triangles_and_reduce, prepare_sorted_workspace_edges,
};
use super::homology::betti_1_from_counts;
use super::space3d::Point3D;
use super::workspace_sweep::uf_union;
use crate::error::TdaError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};

pub(super) const DEFAULT_RADII_SCRATCH_CAP: usize = 512;

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

struct FiltrationSlices<'a> {
    dist: &'a mut [f64],
    sorted_edges: &'a mut [[u64; 2]],
    adj_words: &'a mut [u64],
    basis_words: &'a mut [u64],
    col_scratch: &'a mut [u64],
    uf_parent: &'a mut [usize],
    uf_rank: &'a mut [usize],
    edges: &'a mut [[usize; 2]],
    edge_lookup: &'a mut [usize],
    triangles: &'a mut [[usize; 3]],
    pivot_for_row: &'a mut [usize],
}

/// Preallocated scratch buffers enabling zero-allocation Vietoris-Rips filtration sweeps
/// and `ℤ₂` boundary-matrix column reduction up to `max_points` vertices.
#[derive(Debug, Clone)]
pub struct TdaFiltrationWorkspace {
    pub(super) max_points: usize,
    edge_cap: usize,
    tri_cap: usize,
    max_vertex_words: usize,
    max_edge_words: usize,
    pub(super) arena: OwnedByteArena,
    dist_reg: ArenaRegion<f64>,
    pub(super) comp_births_reg: ArenaRegion<f64>,
    pub(super) hole_births_reg: ArenaRegion<f64>,
    pub(super) radii_scratch_reg: ArenaRegion<f64>,
    edges_reg: ArenaRegion<[usize; 2]>,
    triangles_reg: ArenaRegion<[usize; 3]>,
    sorted_edges_len: usize,
    edges_len: usize,
    triangles_len: usize,
    pub(super) comp_births_len: usize,
    pub(super) hole_births_len: usize,
    uf_components: usize,
    active_n: usize,
    edges_prepared: bool,
    edge_cursor: usize,
    basis_rank: usize,
    pub(super) prev_radius: Option<f64>,
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
        let edge_cap = max_edges.max(1);
        let max_tris = cap
            .saturating_mul(cap.saturating_sub(1))
            .saturating_mul(cap.saturating_sub(2))
            / 6;
        let tri_cap = max_tris.max(1);
        let max_edge_words = edge_cap.div_ceil(64).max(1);
        let max_vertex_words = cap.div_ceil(64).max(1);

        let mut layout = ArenaLayout::new();
        let dist_reg = layout.add_slice::<f64>(cap * cap).expect("tda dist");
        let comp_births_reg = layout.add_slice::<f64>(cap).expect("tda comp_births");
        let hole_births_reg = layout.add_slice::<f64>(edge_cap).expect("tda hole_births");
        let radii_scratch_reg = layout
            .add_slice::<f64>(DEFAULT_RADII_SCRATCH_CAP)
            .expect("tda radii_scratch");
        let _ = layout
            .add_slice::<[u64; 2]>(edge_cap)
            .expect("tda sorted_edges");
        let _ = layout
            .add_slice::<u64>(cap * max_vertex_words)
            .expect("tda adj_words");
        let _ = layout
            .add_slice::<u64>(edge_cap * max_edge_words)
            .expect("tda basis_words");
        let _ = layout
            .add_slice::<u64>(max_edge_words)
            .expect("tda col_scratch");
        let _ = layout.add_slice::<usize>(cap).expect("tda uf_parent");
        let _ = layout.add_slice::<usize>(cap).expect("tda uf_rank");
        let edges_reg = layout.add_slice::<[usize; 2]>(edge_cap).expect("tda edges");
        let lookup_reg = layout.add_slice::<usize>(cap * cap).expect("tda lookup");
        let triangles_reg = layout
            .add_slice::<[usize; 3]>(tri_cap)
            .expect("tda triangles");
        let pivot_reg = layout.add_slice::<usize>(edge_cap).expect("tda pivots");

        let mut arena = OwnedByteArena::with_capacity(layout.total_bytes());
        arena.region_mut(lookup_reg).fill(usize::MAX);
        arena.region_mut(pivot_reg).fill(usize::MAX);

        Self {
            max_points: cap,
            edge_cap,
            tri_cap,
            max_vertex_words,
            max_edge_words,
            arena,
            dist_reg,
            comp_births_reg,
            hole_births_reg,
            radii_scratch_reg,
            edges_reg,
            triangles_reg,
            sorted_edges_len: 0,
            edges_len: 0,
            triangles_len: 0,
            comp_births_len: 0,
            hole_births_len: 0,
            uf_components: cap,
            active_n: 0,
            edges_prepared: false,
            edge_cursor: 0,
            basis_rank: 0,
            prev_radius: None,
        }
    }

    fn slices_mut(&mut self) -> FiltrationSlices<'_> {
        let cap = self.max_points;
        let ec = self.edge_cap;
        let tc = self.tri_cap;
        let vw = self.max_vertex_words;
        let ew = self.max_edge_words;
        let mut scope = self.arena.reborrow();
        let dist = scope.retain_slice::<f64>(cap * cap).expect("dist");
        let _ = scope.retain_slice::<f64>(cap).expect("comp_births");
        let _ = scope.retain_slice::<f64>(ec).expect("hole_births");
        let _ = scope
            .retain_slice::<f64>(DEFAULT_RADII_SCRATCH_CAP)
            .expect("radii_scratch");
        FiltrationSlices {
            dist,
            sorted_edges: scope.retain_slice::<[u64; 2]>(ec).expect("sorted_edges"),
            adj_words: scope.retain_slice::<u64>(cap * vw).expect("adj_words"),
            basis_words: scope.retain_slice::<u64>(ec * ew).expect("basis_words"),
            col_scratch: scope.retain_slice::<u64>(ew).expect("col_scratch"),
            uf_parent: scope.retain_slice::<usize>(cap).expect("uf_parent"),
            uf_rank: scope.retain_slice::<usize>(cap).expect("uf_rank"),
            edges: scope.retain_slice::<[usize; 2]>(ec).expect("edges"),
            edge_lookup: scope.retain_slice::<usize>(cap * cap).expect("lookup"),
            triangles: scope.retain_slice::<[usize; 3]>(tc).expect("triangles"),
            pivot_for_row: scope.retain_slice::<usize>(ec).expect("pivots"),
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
        let dist = self.arena.region_mut(self.dist_reg);
        for (i, pi) in points.iter().enumerate() {
            if !pi.x.is_finite() || !pi.y.is_finite() {
                return Err(TdaError::MatrixError {
                    reason: "Non-finite 2D coordinate".to_string(),
                });
            }
            dist[flatten_2d_index(i, i, stride)] = 0.0;
            for (j, pj) in points.iter().enumerate().skip(i + 1) {
                let dx = pi.x - pj.x;
                let dy = pi.y - pj.y;
                let d = (dx * dx + dy * dy).sqrt();
                dist[flatten_2d_index(i, j, stride)] = d;
                dist[flatten_2d_index(j, i, stride)] = d;
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
        let dist = self.arena.region_mut(self.dist_reg);
        for (i, pi) in points.iter().enumerate() {
            if !pi.x.is_finite() || !pi.y.is_finite() || !pi.z.is_finite() {
                return Err(TdaError::MatrixError {
                    reason: "Non-finite 3D coordinate".to_string(),
                });
            }
            dist[flatten_2d_index(i, i, stride)] = 0.0;
            for (j, pj) in points.iter().enumerate().skip(i + 1) {
                let dx = pi.x - pj.x;
                let dy = pi.y - pj.y;
                let dz = pi.z - pj.z;
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                dist[flatten_2d_index(i, j, stride)] = d;
                dist[flatten_2d_index(j, i, stride)] = d;
            }
        }
        self.prepare_edges_for_n(n);
        Ok(())
    }

    #[verified_engine::verified]
    fn prepare_edges_for_n(&mut self, n: usize) {
        let stride = self.max_points;
        let len = {
            let s = self.slices_mut();
            prepare_sorted_workspace_edges(s.dist, stride, n, s.sorted_edges)
        };
        self.sorted_edges_len = len;
        self.active_n = n;
        self.edges_prepared = true;
        self.prev_radius = None;
    }

    /// Returns the precomputed pairwise distance `d(i, j)` for an `n`-point cloud.
    #[verified_engine::verified]
    pub fn distance(&self, n: usize, i: usize, j: usize) -> f64 {
        debug_assert!(i < n && j < n);
        self.arena.region(self.dist_reg)[flatten_2d_index(i, j, self.max_points)]
    }

    /// Returns the canonical `[i, j]` 1-simplices from the most recent [`Self::evaluate_radius`] call.
    #[verified_engine::verified]
    pub fn active_edges(&self) -> &[[usize; 2]] {
        &self.arena.region(self.edges_reg)[..self.edges_len]
    }

    /// Returns the canonical `[i, j, k]` 2-simplices from the most recent [`Self::evaluate_radius`] call.
    #[verified_engine::verified]
    pub fn active_triangles(&self) -> &[[usize; 3]] {
        &self.arena.region(self.triangles_reg)[..self.triangles_len]
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
        let vw = self.max_vertex_words;
        let total_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
        {
            let s = self.slices_mut();
            for (i, slot) in s.uf_parent.iter_mut().enumerate().take(n) {
                *slot = i;
            }
            s.uf_rank[..n].fill(0);
            for u in 0..n {
                s.adj_words[u * vw..u * vw + n_words].fill(0);
            }
            s.pivot_for_row[..total_edges.max(1)].fill(usize::MAX);
        }
        self.uf_components = n;
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
        let vw = self.max_vertex_words;
        let cap = self.max_points;
        let sorted_len = self.sorted_edges_len;

        let mut cursor = self.edge_cursor;
        let mut e_len = self.edges_len;
        let mut t_len = self.triangles_len;
        let mut comps = self.uf_components;
        let mut b_rank = self.basis_rank;

        {
            let s = self.slices_mut();
            while cursor < sorted_len {
                let packed = s.sorted_edges[cursor];
                let diam = f64::from_bits(packed[0]);
                if diam > radius {
                    break;
                }
                cursor += 1;
                let u = (packed[1] >> 16) as usize;
                let v = (packed[1] & 0xFFFF) as usize;
                let e_uv = e_len;
                s.edges[e_len] = [u, v];
                e_len += 1;
                s.edge_lookup[flatten_2d_index(u, v, cap)] = e_uv;
                s.edge_lookup[flatten_2d_index(v, u, cap)] = e_uv;

                let merged = uf_union(s.uf_parent, s.uf_rank, &mut comps, u, v);
                if !merged {
                    let curr_cycles = betti_1_from_counts(n, e_len, comps, 0);
                    let ctx = EdgeTriangleCtx {
                        u,
                        v,
                        e_uv,
                        n_words,
                        vertex_stride: vw,
                        lookup_stride: cap,
                        edge_words,
                        curr_cycles,
                        adj_words: s.adj_words,
                        edge_lookup: s.edge_lookup,
                    };
                    append_edge_triangles_and_reduce(
                        &ctx,
                        s.triangles,
                        &mut t_len,
                        s.col_scratch,
                        s.pivot_for_row,
                        s.basis_words,
                        &mut b_rank,
                    );
                }
                s.adj_words[u * vw + (v >> 6)] |= 1_u64 << (v & 63);
                s.adj_words[v * vw + (u >> 6)] |= 1_u64 << (u & 63);
            }
        }

        self.edge_cursor = cursor;
        self.edges_len = e_len;
        self.triangles_len = t_len;
        self.uf_components = comps;
        self.basis_rank = b_rank;
    }
}

// [cite:clinical_trials_statistics]
