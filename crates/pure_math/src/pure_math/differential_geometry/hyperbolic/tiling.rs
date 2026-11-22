//! Zero-allocation Fuchsian $\{p, q\}$ tessellations and fundamental-domain re-centering.

use super::{
    HyperbolicError, MAX_POLYGON_SIDES, MAX_TILING_DEPTH, MAX_TILING_POLYGONS, Su11Matrix,
    clamp_to_disk, poincare_distance,
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Returns `true` iff $\{p, q\}$ defines a compact hyperbolic tessellation:
/// $3 \le p, q \le 16$ and $(p - 2)(q - 2) > 4$.
#[must_use]
#[verified_engine::verified]
pub fn is_hyperbolic_schlafli(p: usize, q: usize) -> bool {
    if !(3..=MAX_POLYGON_SIDES).contains(&p) || !(3..=MAX_POLYGON_SIDES).contains(&q) {
        return false;
    }
    (p - 2) * (q - 2) > 4
}

/// Returns the Euclidean circumradius $r_0 \in (0, 1)$ in the Poincaré disk of the central
/// regular $\{p, q\}$ polygon.
#[verified_engine::verified]
pub fn regular_polygon_poincare_radius(p: usize, q: usize) -> Result<f64, HyperbolicError> {
    if !is_hyperbolic_schlafli(p, q) {
        return Err(HyperbolicError::NonHyperbolicSchlafli { p, q });
    }
    let cot_p = 1.0 / (PI / (p as f64)).tan();
    let cot_q = 1.0 / (PI / (q as f64)).tan();
    let c = cot_p * cot_q;
    Ok(((c - 1.0) / (c + 1.0)).max(0.0).sqrt())
}

/// Returns the intrinsic hyperbolic inradius $h$ (distance from center to edge midpoint)
/// of a regular $\{p, q\}$ polygon: $\cosh h = \frac{\cos(\pi/q)}{\sin(\pi/p)}$.
#[verified_engine::verified]
pub fn regular_polygon_inradius(p: usize, q: usize) -> Result<f64, HyperbolicError> {
    if !is_hyperbolic_schlafli(p, q) {
        return Err(HyperbolicError::NonHyperbolicSchlafli { p, q });
    }
    let cosh_h = (PI / (q as f64)).cos() / (PI / (p as f64)).sin();
    Ok(cosh_h.max(1.0).acosh())
}

/// Stack-inline hyperbolic polygon with up to [`MAX_POLYGON_SIDES`] Poincaré disk vertices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperbolicPolygon {
    /// Counter-clockwise vertex coordinates in the Poincaré disk (`0..sides` active).
    pub vertices: [Complex64; MAX_POLYGON_SIDES],
    /// Number of active sides $p$.
    pub sides: usize,
    /// Transformed Poincaré coordinate of the tile center.
    pub center: Complex64,
    /// Combinatorial BFS reflection/rotation layer depth from the central tile.
    pub layer: usize,
}

impl Default for HyperbolicPolygon {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            vertices: [Complex64::new(0.0, 0.0); MAX_POLYGON_SIDES],
            sides: 0,
            center: Complex64::new(0.0, 0.0),
            layer: 0,
        }
    }
}

impl HyperbolicPolygon {
    /// Active Poincaré vertex slice of length `self.sides`.
    #[must_use]
    pub fn active_vertices(&self) -> &[Complex64] {
        &self.vertices[..self.sides]
    }
}

/// Pre-allocated Fuchsian $\{p, q\}$ tessellation buffer supporting zero-allocation generation
/// and continuous Möbius panning with fundamental-domain re-centering.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::num_complex::Complex64;
/// use math_explorer::pure_math::differential_geometry::hyperbolic::HyperbolicTilingBuffer;
///
/// let mut buf = HyperbolicTilingBuffer::with_capacity(128);
/// let count = buf.configure(7, 3, 2).expect("valid {7, 3} hyperbolic tiling");
/// assert!(count > 7);
/// buf.apply_pan(Complex64::new(0.25, -0.15));
/// assert_eq!(buf.polygons().len(), count);
/// ```
#[derive(Debug, Clone)]
pub struct HyperbolicTilingBuffer {
    capacity: usize,
    p: usize,
    q: usize,
    depth: usize,
    view: Su11Matrix,
    pan_steps: u64,
    base_vertices: [Complex64; MAX_POLYGON_SIDES],
    generators: [Su11Matrix; MAX_POLYGON_SIDES],
    group_elements: Vec<Su11Matrix>,
    tile_layers: Vec<usize>,
    polygons: Vec<HyperbolicPolygon>,
}

impl Default for HyperbolicTilingBuffer {
    fn default() -> Self {
        let mut buf = Self::with_capacity(256);
        let _ = buf.configure(7, 3, 2);
        buf
    }
}

impl HyperbolicTilingBuffer {
    /// Pre-allocates storage for at most `max_polygons.clamp(1, MAX_TILING_POLYGONS)` tiles.
    #[must_use]
    pub fn with_capacity(max_polygons: usize) -> Self {
        let cap = max_polygons.clamp(1, MAX_TILING_POLYGONS);
        Self {
            capacity: cap,
            p: 7,
            q: 3,
            depth: 2,
            view: Su11Matrix::identity(),
            pan_steps: 0,
            base_vertices: [Complex64::new(0.0, 0.0); MAX_POLYGON_SIDES],
            generators: [Su11Matrix::identity(); MAX_POLYGON_SIDES],
            group_elements: Vec::with_capacity(cap),
            tile_layers: Vec::with_capacity(cap),
            polygons: Vec::with_capacity(cap),
        }
    }

    /// Active Schläfli symbol `(p, q)` and BFS `depth`.
    #[must_use]
    #[verified_engine::verified]
    pub fn params(&self) -> (usize, usize, usize) {
        (self.p, self.q, self.depth)
    }

    /// Current reduced camera transformation in $\mathrm{SU}(1,1)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn view_matrix(&self) -> Su11Matrix {
        self.view
    }

    /// Sets the camera transformation directly, renormalizing and refreshing tile coordinates.
    #[verified_engine::verified]
    pub fn set_view_matrix(&mut self, mut matrix: Su11Matrix) {
        matrix.renormalize();
        self.view = matrix;
        self.recenter_fundamental_domain();
        self.refresh_projected_polygons();
    }

    /// Resets the camera transformation back to the identity matrix.
    #[verified_engine::verified]
    pub fn reset_view(&mut self) {
        self.view = Su11Matrix::identity();
        self.pan_steps = 0;
        self.refresh_projected_polygons();
    }

    /// Projected Poincaré polygons in the current camera view.
    #[must_use]
    pub fn polygons(&self) -> &[HyperbolicPolygon] {
        &self.polygons
    }

    /// Populates the $\{p, q\}$ Fuchsian group elements up to `depth` without exceeding capacity.
    #[verified_engine::verified]
    pub fn configure(
        &mut self,
        p: usize,
        q: usize,
        depth: usize,
    ) -> Result<usize, HyperbolicError> {
        let r0 = regular_polygon_poincare_radius(p, q)?;
        let inradius = regular_polygon_inradius(p, q)?;
        let mid_r = (0.5 * inradius).tanh();
        let bounded_depth = depth.min(MAX_TILING_DEPTH);

        self.p = p;
        self.q = q;
        self.depth = bounded_depth;

        let mut k = 0;
        while k < p {
            let ang = 2.0 * PI * (k as f64) / (p as f64);
            self.base_vertices[k] = clamp_to_disk(Complex64::new(r0 * ang.cos(), r0 * ang.sin()));
            let mid_ang = 2.0 * PI * ((k as f64) + 0.5) / (p as f64);
            let midpoint = Complex64::new(mid_r * mid_ang.cos(), mid_r * mid_ang.sin());
            self.generators[k] = Su11Matrix::edge_half_turn(midpoint);
            k += 1;
        }

        self.build_bfs_group(bounded_depth);
        self.recenter_fundamental_domain();
        self.refresh_projected_polygons();
        Ok(self.polygons.len())
    }

    #[verified_engine::verified]
    fn build_bfs_group(&mut self, bounded_depth: usize) {
        self.group_elements.clear();
        self.tile_layers.clear();
        self.polygons.clear();

        self.group_elements.push(Su11Matrix::identity());
        self.tile_layers.push(0);
        self.polygons.push(HyperbolicPolygon::default());

        let mut cursor = 0;
        while cursor < self.group_elements.len() && self.group_elements.len() < self.capacity {
            let cur_layer = self.tile_layers[cursor];
            if cur_layer >= bounded_depth {
                cursor += 1;
                continue;
            }
            let parent = self.group_elements[cursor];
            self.expand_parent_neighbors(parent, cur_layer + 1);
            cursor += 1;
        }
    }

    #[verified_engine::verified]
    fn expand_parent_neighbors(&mut self, parent: Su11Matrix, next_layer: usize) {
        let origin = Complex64::new(0.0, 0.0);
        let mut side = 0;
        while side < self.p && self.group_elements.len() < self.capacity {
            let mut candidate = parent.compose(&self.generators[side]);
            candidate.renormalize();
            let cand_center = candidate.apply(origin);
            if !self.contains_center(cand_center) {
                self.group_elements.push(candidate);
                self.tile_layers.push(next_layer);
                self.polygons.push(HyperbolicPolygon::default());
            }
            side += 1;
        }
    }

    #[verified_engine::verified]
    fn contains_center(&self, center: Complex64) -> bool {
        let origin = Complex64::new(0.0, 0.0);
        for existing in &self.group_elements {
            let existing_center = existing.apply(origin);
            if poincare_distance(center, existing_center) < 1e-4 {
                return true;
            }
        }
        false
    }

    /// Applies a Möbius pan `delta`, renormalizes `view`, reduces modulo Fuchsian generators for
    /// drift-free infinite panning, and updates all tile vertices in-place with zero heap allocations.
    #[verified_engine::verified]
    pub fn apply_pan(&mut self, delta: Complex64) -> Su11Matrix {
        let step = Su11Matrix::hyperbolic_pan(delta);
        self.view = step.compose(&self.view);
        self.pan_steps = self.pan_steps.wrapping_add(1);
        if self.pan_steps.is_multiple_of(8) {
            self.view.renormalize();
        }
        self.recenter_fundamental_domain();
        self.refresh_projected_polygons();
        step
    }

    #[verified_engine::verified]
    fn recenter_fundamental_domain(&mut self) {
        let origin = Complex64::new(0.0, 0.0);
        let mut iter = 0;
        while iter < 6 {
            let current_norm = self.view.apply(origin).norm_sqr();
            let mut best_side = None;
            let mut best_norm = current_norm - 1e-7;
            let mut side = 0;
            while side < self.p {
                let candidate = self.view.compose(&self.generators[side]);
                let norm = candidate.apply(origin).norm_sqr();
                if norm < best_norm {
                    best_norm = norm;
                    best_side = Some(side);
                }
                side += 1;
            }
            if let Some(winner) = best_side {
                self.view = self.view.compose(&self.generators[winner]);
                self.view.renormalize();
                iter += 1;
            } else {
                break;
            }
        }
    }

    #[verified_engine::verified]
    fn refresh_projected_polygons(&mut self) {
        let origin = Complex64::new(0.0, 0.0);
        let count = self.group_elements.len();
        let mut idx = 0;
        while idx < count {
            let m = self.view.compose(&self.group_elements[idx]);
            let mut poly = HyperbolicPolygon {
                vertices: [origin; MAX_POLYGON_SIDES],
                sides: self.p,
                center: m.apply(origin),
                layer: self.tile_layers[idx],
            };
            let mut v = 0;
            while v < self.p {
                poly.vertices[v] = m.apply(self.base_vertices[v]);
                v += 1;
            }
            self.polygons[idx] = poly;
            idx += 1;
        }
    }
}

// [cite:modular_polynomials_review]
