//! Flat-slice discrete 2-manifolds and zero-allocation Mean Curvature Flow (`MeanCurvatureFlow`).

pub use super::buckled_flow::{BuckledSurfaceFlow, SurfaceTopology};
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::{Point3, Vector3};

/// A discrete 2-manifold represented by a contiguous flat `(nu, nv)` grid of 3D vertices.
///
/// Vertex `(i, j)` for `i in 0..nu` and `j in 0..nv` is stored at
/// `flatten_2d_index(j, i, nv)` (`i * nv + j`) so surface evolutions run with zero inner-row
/// heap allocations.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteSurface {
    /// Contiguous row-major vertex buffer of length `nu * nv`.
    pub points: Vec<Point3<f64>>,
    /// Grid resolution along parameter $u$ (`nu >= 3`).
    pub nu: usize,
    /// Grid resolution along parameter $v$ (`nv >= 3`).
    pub nv: usize,
    /// Whether the $u$-boundary wraps periodically (`rem_euclid(nu)`).
    pub closed_u: bool,
    /// Whether the $v$-boundary wraps periodically (`rem_euclid(nv)`).
    pub closed_v: bool,
}

impl DiscreteSurface {
    /// Creates a flat-buffered discrete surface of dimensions `(nu, nv)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(
        points: Vec<Point3<f64>>,
        nu: usize,
        nv: usize,
        closed_u: bool,
        closed_v: bool,
    ) -> Self {
        assert!(
            nu >= 2 && nv >= 2,
            "DiscreteSurface requires nu >= 2 and nv >= 2"
        );
        assert_eq!(
            points.len(),
            nu * nv,
            "DiscreteSurface flat points length must equal nu * nv"
        );
        Self {
            points,
            nu,
            nv,
            closed_u,
            closed_v,
        }
    }

    /// Samples a parametric map `f(u, v)` onto a contiguous flat `(nu, nv)` surface grid.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_parametric<F>(
        u_range: (f64, f64),
        v_range: (f64, f64),
        nu: usize,
        nv: usize,
        closed_u: bool,
        closed_v: bool,
        f: F,
    ) -> Self
    where
        F: Fn(f64, f64) -> Point3<f64>,
    {
        assert!(nu >= 2 && nv >= 2);
        let u_denom = if closed_u { nu as f64 } else { (nu - 1) as f64 };
        let v_denom = if closed_v { nv as f64 } else { (nv - 1) as f64 };
        let du = (u_range.1 - u_range.0) / u_denom;
        let dv = (v_range.1 - v_range.0) / v_denom;

        let mut points = vec![Point3::origin(); nu * nv];
        for i in 0..nu {
            let u = u_range.0 + i as f64 * du;
            for j in 0..nv {
                let v = v_range.0 + j as f64 * dv;
                let idx = flatten_2d_index(j, i, nv);
                points[idx] = f(u, v);
            }
        }
        Self::new(points, nu, nv, closed_u, closed_v)
    }

    /// Returns `(nu, nv)` grid dimensions.
    #[must_use]
    #[inline]
    pub const fn dims(&self) -> (usize, usize) {
        (self.nu, self.nv)
    }

    /// Returns the flat vertex slice of length `nu * nv`.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[Point3<f64>] {
        &self.points
    }

    /// Returns vertex `(i, j)` via [`flatten_2d_index`].
    #[must_use]
    #[inline]
    pub fn point(&self, i: usize, j: usize) -> Point3<f64> {
        self.points[flatten_2d_index(j, i, self.nv)]
    }

    #[verified_engine::verified]
    fn get(&self, i: i32, j: i32) -> Point3<f64> {
        let nu = self.nu as i32;
        let nv = self.nv as i32;
        assert!(nu >= 2 && nv >= 2);

        let i_idx = if self.closed_u {
            i.rem_euclid(nu)
        } else {
            i.clamp(0, nu - 1)
        } as usize;
        let j_idx = if self.closed_v {
            j.rem_euclid(nv)
        } else {
            j.clamp(0, nv - 1)
        } as usize;

        self.points[flatten_2d_index(j_idx, i_idx, self.nv)]
    }

    /// Computes the discrete Mean Curvature Vector $\vec{H} = H \mathbf{n}$ at grid node `(i, j)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_curvature_vector(&self, i: usize, j: usize) -> Vector3<f64> {
        assert!(i < self.nu && j < self.nv);
        let idx_i = i as i32;
        let idx_j = j as i32;
        let p = self.get(idx_i, idx_j);

        let p_ip = self.get(idx_i + 1, idx_j);
        let p_im = self.get(idx_i - 1, idx_j);
        let p_jp = self.get(idx_i, idx_j + 1);
        let p_jm = self.get(idx_i, idx_j - 1);

        let r_u = (p_ip - p_im) / 2.0;
        let r_v = (p_jp - p_jm) / 2.0;

        let r_uu = p_ip.coords - p.coords * 2.0 + p_im.coords;
        let r_vv = p_jp.coords - p.coords * 2.0 + p_jm.coords;

        let p_pp = self.get(idx_i + 1, idx_j + 1);
        let p_mm = self.get(idx_i - 1, idx_j - 1);
        let p_pm = self.get(idx_i + 1, idx_j - 1);
        let p_mp = self.get(idx_i - 1, idx_j + 1);
        let r_uv = (p_pp.coords - p_mp.coords - p_pm.coords + p_mm.coords) / 4.0;

        let e = r_u.dot(&r_u);
        let f = r_u.dot(&r_v);
        let g = r_v.dot(&r_v);
        let det_g = e * g - f * f;

        let n_unnorm = r_u.cross(&r_v);
        let n_norm = n_unnorm.norm();
        if det_g.abs() < 1e-12 || n_norm < 1e-12 || !n_norm.is_finite() {
            return Vector3::zeros();
        }
        let n = n_unnorm / n_norm;

        let l = r_uu.dot(&n);
        let m = r_uv.dot(&n);
        let n_coeff = r_vv.dot(&n);

        let h_val = (e * n_coeff - 2.0 * f * m + g * l) / (2.0 * det_g);
        if !h_val.is_finite() {
            return Vector3::zeros();
        }
        n * h_val
    }

    /// Approximates total discrete surface area across all `(nu, nv)` quads.
    #[must_use]
    #[verified_engine::verified]
    pub fn surface_area(&self) -> f64 {
        assert_eq!(self.points.len(), self.nu * self.nv);
        let i_end = if self.closed_u { self.nu } else { self.nu - 1 };
        let j_end = if self.closed_v { self.nv } else { self.nv - 1 };
        let mut area = 0.0;
        for i in 0..i_end {
            for j in 0..j_end {
                let p00 = self.get(i as i32, j as i32);
                let p10 = self.get(i as i32 + 1, j as i32);
                let p01 = self.get(i as i32, j as i32 + 1);
                let p11 = self.get(i as i32 + 1, j as i32 + 1);
                area += 0.5 * (p10 - p00).cross(&(p11 - p00)).norm();
                area += 0.5 * (p11 - p00).cross(&(p01 - p00)).norm();
            }
        }
        assert!(area.is_finite() && area >= 0.0);
        area
    }
}

/// Zero-allocation explicit Mean Curvature Flow solver ($\partial_t \mathbf{r} = \vec{H}$)
/// backed by flat ping-pong vertex buffers.
#[derive(Debug, Clone)]
pub struct MeanCurvatureFlow {
    /// Active evolving discrete surface.
    pub surface: DiscreteSurface,
    /// Pre-allocated flat scratch buffer (`len = surface.nu * surface.nv`) swapped on each step.
    pub next_points: Vec<Point3<f64>>,
    /// Pre-allocated flat scalar mean curvature norms $\|\vec{H}\|$ updated on each step.
    pub curvature_norms: Vec<f64>,
}

impl MeanCurvatureFlow {
    /// Initializes a zero-allocation `MeanCurvatureFlow` solver from `surface`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(surface: DiscreteSurface) -> Self {
        let len = surface.points.len();
        assert_eq!(len, surface.nu * surface.nv);
        let next_points = surface.points.clone();
        let curvature_norms = vec![0.0; len];
        let mut solver = Self {
            surface,
            next_points,
            curvature_norms,
        };
        solver.refresh_curvature_norms();
        solver
    }

    /// Recomputes `curvature_norms` in place without advancing surface positions.
    #[verified_engine::verified]
    pub fn refresh_curvature_norms(&mut self) {
        let (nu, nv) = self.surface.dims();
        assert_eq!(self.curvature_norms.len(), nu * nv);
        for i in 0..nu {
            for j in 0..nv {
                let idx = flatten_2d_index(j, i, nv);
                self.curvature_norms[idx] = self.surface.mean_curvature_vector(i, j).norm();
            }
        }
    }

    /// Estimates a conservative explicit parabolic CFL time-step bound based on minimum edge length.
    #[must_use]
    #[verified_engine::verified]
    pub fn recommended_cfl_dt(&self) -> f64 {
        let (nu, nv) = self.surface.dims();
        assert!(nu >= 2 && nv >= 2);
        let mut min_ds2 = f64::INFINITY;
        for i in 0..(nu - 1) {
            for j in 0..(nv - 1) {
                let p = self.surface.point(i, j);
                let pu = self.surface.point(i + 1, j);
                let pv = self.surface.point(i, j + 1);
                let du2 = (pu - p).norm_squared();
                let dv2 = (pv - p).norm_squared();
                if du2 > 1e-10 {
                    min_ds2 = min_ds2.min(du2);
                }
                if dv2 > 1e-10 {
                    min_ds2 = min_ds2.min(dv2);
                }
            }
        }
        if !min_ds2.is_finite() {
            return 1e-3;
        }
        // Note: `mean_curvature_vector` uses unit index finite differences, where
        // |h_vec| ~ H ~ 1/R while second-difference amplification scales as dt / min_ds2.
        (0.2 * min_ds2).clamp(1e-5, 0.05)
    }

    /// Advances the discrete surface by one explicit Mean Curvature Flow step `dt` with
    /// **zero heap allocations**.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        assert!(
            dt.is_finite() && dt >= 0.0,
            "dt must be finite and non-negative"
        );
        let (nu, nv) = self.surface.dims();
        assert_eq!(self.next_points.len(), nu * nv);
        assert_eq!(self.curvature_norms.len(), nu * nv);

        for i in 0..nu {
            for j in 0..nv {
                let idx = flatten_2d_index(j, i, nv);
                let h_vec = self.surface.mean_curvature_vector(i, j);
                self.curvature_norms[idx] = h_vec.norm();
                self.next_points[idx] = self.surface.points[idx] + h_vec * dt;
            }
        }

        std::mem::swap(&mut self.surface.points, &mut self.next_points);
    }
}
