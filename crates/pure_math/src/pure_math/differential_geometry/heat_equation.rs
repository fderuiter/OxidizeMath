//! Flat-slice zero-allocation Heat Equation solver ($\partial_t u = \alpha \Delta_S u$) on parametric surfaces.

use super::buckled_flow::{SurfaceTopology, detect_twisted_seam_map};
use super::surface::{ParametricSurface, SurfaceAnalysis};
use math_commons::math_kernel::types::flatten_2d_index;

/// Solves the Heat Equation $\frac{\partial u}{\partial t} = \alpha \Delta_S u$ on a parametric surface
/// using contiguous row-major flat buffers indexed via [`flatten_2d_index`].
pub struct HeatEquationSolver<'a, S: ParametricSurface> {
    /// Reference to the underlying curved parametric manifold.
    pub surface: &'a S,
    /// Thermal diffusivity coefficient $\alpha \ge 0$.
    pub alpha: f64,
    /// Current flat temperature buffer of length `grid_res.0 * grid_res.1`.
    pub u_grid: Vec<f64>,
    /// Pre-allocated scratch buffer swapped on each step with zero heap allocations.
    pub next_grid: Vec<f64>,
    /// Grid resolution `(nu, nv)` along `(u, v)`.
    pub grid_res: (usize, usize),
    /// Parameter interval `(u_min, u_max)`.
    pub range_u: (f64, f64),
    /// Parameter interval `(v_min, v_max)`.
    pub range_v: (f64, f64),
    /// Boundary topology applied across parameter seams and polar caps.
    pub topology: SurfaceTopology,
    seam_fwd: Vec<usize>,
    seam_rev: Vec<usize>,
}

impl<'a, S: ParametricSurface> HeatEquationSolver<'a, S> {
    /// Creates a new flat-buffered Heat Equation solver on `surface`.
    ///
    /// # Examples
    ///
    /// Simulating heat diffusion on a unit [`Sphere`](super::surface::Sphere):
    ///
    /// ```rust
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::differential_geometry::surface::Sphere;
    /// use math_explorer::pure_math::differential_geometry::heat_equation::HeatEquationSolver;
    /// use std::f64::consts::PI;
    ///
    /// let sphere = Sphere { radius: 1.0 };
    /// let initial_hot_spot = |_u: f64, v: f64| -> f64 {
    ///     if v < 0.5 { 100.0 } else { 0.0 }
    /// };
    ///
    /// let mut solver = HeatEquationSolver::new(
    ///     &sphere,
    ///     0.01,
    ///     (0.0, 2.0 * PI),
    ///     (0.05, PI - 0.05),
    ///     (20, 20),
    ///     initial_hot_spot,
    /// );
    ///
    /// let initial_max = solver.u_grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    /// assert!(initial_max > 99.0);
    ///
    /// for _ in 0..10 {
    ///     solver.step(0.001);
    /// }
    ///
    /// let final_max = solver.u_grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    /// assert!(final_max < initial_max);
    /// ```
    #[verified_engine::verified]
    pub fn new(
        surface: &'a S,
        alpha: f64,
        range_u: (f64, f64),
        range_v: (f64, f64),
        grid_res: (usize, usize),
        initial_condition: impl Fn(f64, f64) -> f64,
    ) -> Self {
        let (nu, nv) = grid_res;
        assert!(
            nu >= 2 && nv >= 2,
            "HeatEquationSolver requires grid_res >= (2, 2)"
        );
        assert!(alpha.is_finite() && alpha >= 0.0);

        let du = (range_u.1 - range_u.0) / (nu as f64 - 1.0);
        let dv = (range_v.1 - range_v.0) / (nv as f64 - 1.0);

        let mut grid = vec![0.0; nu * nv];
        for i in 0..nu {
            let u = range_u.0 + i as f64 * du;
            for j in 0..nv {
                let v = range_v.0 + j as f64 * dv;
                let idx = flatten_2d_index(j, i, nv);
                grid[idx] = initial_condition(u, v);
            }
        }

        Self {
            surface,
            alpha,
            next_grid: grid.clone(),
            u_grid: grid,
            grid_res,
            range_u,
            range_v,
            topology: SurfaceTopology::OpenNeumann,
            seam_fwd: Vec::new(),
            seam_rev: Vec::new(),
        }
    }

    /// Wraps pre-allocated flat `u_grid` and `next_grid` buffers with **zero heap allocations**.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_buffers(
        surface: &'a S,
        alpha: f64,
        range_u: (f64, f64),
        range_v: (f64, f64),
        grid_res: (usize, usize),
        u_grid: Vec<f64>,
        next_grid: Vec<f64>,
    ) -> Self {
        let (nu, nv) = grid_res;
        assert!(nu >= 2 && nv >= 2);
        assert_eq!(u_grid.len(), nu * nv);
        assert_eq!(next_grid.len(), nu * nv);
        assert!(alpha.is_finite() && alpha >= 0.0);
        Self {
            surface,
            alpha,
            u_grid,
            next_grid,
            grid_res,
            range_u,
            range_v,
            topology: SurfaceTopology::OpenNeumann,
            seam_fwd: Vec::new(),
            seam_rev: Vec::new(),
        }
    }

    /// Configures intrinsic seam/polar boundary topology on this solver.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_topology(mut self, topology: SurfaceTopology) -> Self {
        let (nu, nv) = self.grid_res;
        self.topology = topology;
        if topology == SurfaceTopology::TwistedKlein {
            let (fwd, rev) = detect_twisted_seam_map(self.surface, self.range_u, self.range_v, nv);
            self.seam_fwd = fwd;
            self.seam_rev = rev;
        }
        if topology == SurfaceTopology::PolarSphere && nu > 0 && nv >= 2 {
            for pole_j in [0_usize, nv - 1] {
                let cap_mean: f64 = (0..nu).map(|i| self.value(i, pole_j)).sum::<f64>() / nu as f64;
                for i in 0..nu {
                    let idx = flatten_2d_index(pole_j, i, nv);
                    self.u_grid[idx] = cap_mean;
                    self.next_grid[idx] = cap_mean;
                }
            }
        }
        self
    }

    /// Reads temperature at `(i, j)` via [`flatten_2d_index`].
    #[must_use]
    #[inline]
    pub fn value(&self, i: usize, j: usize) -> f64 {
        self.u_grid[flatten_2d_index(j, i, self.grid_res.1)]
    }

    /// Overwrites temperature at `(i, j)` via [`flatten_2d_index`].
    #[inline]
    pub fn set_value(&mut self, i: usize, j: usize, value: f64) {
        let idx = flatten_2d_index(j, i, self.grid_res.1);
        self.u_grid[idx] = value;
    }

    /// Computes surface-integrated thermal energy $\iint_S u(u, v)\,\sqrt{g}\,du\,dv$.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_heat(&self) -> f64 {
        let (nu, nv) = self.grid_res;
        assert_eq!(self.u_grid.len(), nu * nv);
        let (du, dv) = self.grid_spacings();
        let mut sum = 0.0;
        for i in 0..nu {
            let u = self.range_u.0 + i as f64 * du;
            let wu = if self.topology == SurfaceTopology::OpenNeumann && (i == 0 || i + 1 == nu) {
                0.5
            } else {
                1.0
            };
            for j in 0..nv {
                let v = self.range_v.0 + j as f64 * dv;
                let wv = if matches!(
                    self.topology,
                    SurfaceTopology::OpenNeumann | SurfaceTopology::PolarSphere
                ) && (j == 0 || j + 1 == nv)
                {
                    0.5
                } else {
                    1.0
                };
                let area = self.surface.area_element(u, v);
                if area.is_finite() {
                    sum += wu * wv * self.value(i, j) * area * du * dv;
                }
            }
        }
        assert!(sum.is_finite());
        sum
    }

    /// Estimates a conservative explicit CFL time-step bound for Laplace-Beltrami diffusion.
    #[must_use]
    #[verified_engine::verified]
    pub fn recommended_cfl_dt(&self) -> f64 {
        let (nu, nv) = self.grid_res;
        assert!(nu >= 2 && nv >= 2);
        let alpha_eff = self.alpha.max(1e-6);
        let (du, dv) = self.grid_spacings();
        let mut max_rate = 0.0_f64;

        for i in 0..nu {
            let u = self.range_u.0 + i as f64 * du;
            for j in 0..nv {
                let v = self.range_v.0 + j as f64 * dv;
                let (e, f, g) = self.surface.first_fundamental_form(u, v);
                let det_g = (e * g - f * f).max(1e-8);
                let rate = (g / (det_g * du * du)) + (e / (det_g * dv * dv));
                if rate.is_finite() {
                    max_rate = max_rate.max(rate);
                }
            }
        }

        if max_rate <= 1e-12 {
            0.01
        } else {
            (0.22 / (alpha_eff * max_rate)).clamp(1e-6, 0.1)
        }
    }

    #[inline]
    fn grid_spacings(&self) -> (f64, f64) {
        self.topology
            .grid_spacings(self.range_u, self.range_v, self.grid_res.0, self.grid_res.1)
    }

    #[inline]
    fn sample_cell(&self, i: i32, j: i32) -> f64 {
        let (nu, nv) = (self.grid_res.0 as i32, self.grid_res.1 as i32);
        let (wi, wj) = match self.topology {
            SurfaceTopology::PeriodicTorus => {
                (i.rem_euclid(nu) as usize, j.rem_euclid(nv) as usize)
            }
            SurfaceTopology::TwistedKlein => {
                let base_j = j.rem_euclid(nv) as usize;
                if i < 0 {
                    ((i + nu) as usize, self.seam_rev[base_j])
                } else if i >= nu {
                    ((i - nu) as usize, self.seam_fwd[base_j])
                } else {
                    (i as usize, base_j)
                }
            }
            SurfaceTopology::PolarSphere => {
                (i.rem_euclid(nu) as usize, j.clamp(0, nv - 1) as usize)
            }
            SurfaceTopology::OpenNeumann => {
                (i.clamp(0, nu - 1) as usize, j.clamp(0, nv - 1) as usize)
            }
        };
        self.u_grid[flatten_2d_index(wj, wi, self.grid_res.1)]
    }

    #[inline]
    fn compute_flux_u(&self, i_base: i32, j_base: i32, du: f64, dv: f64) -> f64 {
        let nu = self.grid_res.0 as i32;
        if self.topology == SurfaceTopology::OpenNeumann && (i_base < 0 || i_base + 1 >= nu) {
            return 0.0;
        }
        let u_half = self.range_u.0 + (i_base as f64 + 0.5) * du;
        let v_curr = self.range_v.0 + j_base as f64 * dv;
        let (e, f, g_metric) = self.surface.first_fundamental_form(u_half, v_curr);
        let det_g = e * g_metric - f * f;
        if det_g <= 1e-12 || !det_g.is_finite() {
            return 0.0;
        }
        let sqrt_g = det_g.sqrt();
        let f_u = (self.sample_cell(i_base + 1, j_base) - self.sample_cell(i_base, j_base)) / du;
        let nv = self.grid_res.1;
        let seam_reverses = self.seam_fwd.len() == nv
            && nv >= 3
            && (self.seam_fwd[1] + nv - self.seam_fwd[0]) % nv == nv - 1;
        let v_sign = if self.topology == SurfaceTopology::TwistedKlein
            && seam_reverses
            && (i_base < 0 || i_base + 1 >= nu)
        {
            -1.0
        } else {
            1.0
        };
        let f_v_i = (self.sample_cell(i_base, j_base + 1) - self.sample_cell(i_base, j_base - 1))
            / (2.0 * dv);
        let f_v_ip1 = v_sign
            * (self.sample_cell(i_base + 1, j_base + 1) - self.sample_cell(i_base + 1, j_base - 1))
            / (2.0 * dv);
        (g_metric * f_u - f * 0.5 * (f_v_i + f_v_ip1)) / sqrt_g
    }

    #[inline]
    fn compute_flux_v(&self, i_base: i32, j_base: i32, du: f64, dv: f64) -> f64 {
        let nv = self.grid_res.1 as i32;
        if self.topology == SurfaceTopology::OpenNeumann && (j_base < 0 || j_base + 1 >= nv) {
            return 0.0;
        }
        let u_curr = self.range_u.0 + i_base as f64 * du;
        let v_half = self.range_v.0 + (j_base as f64 + 0.5) * dv;
        let (e, f, g_metric) = self.surface.first_fundamental_form(u_curr, v_half);
        let det_g = e * g_metric - f * f;
        if det_g <= 1e-12 || !det_g.is_finite() {
            return 0.0;
        }
        let sqrt_g = det_g.sqrt();
        let f_v = (self.sample_cell(i_base, j_base + 1) - self.sample_cell(i_base, j_base)) / dv;
        let f_u_j = (self.sample_cell(i_base + 1, j_base) - self.sample_cell(i_base - 1, j_base))
            / (2.0 * du);
        let f_u_jp1 = (self.sample_cell(i_base + 1, j_base + 1)
            - self.sample_cell(i_base - 1, j_base + 1))
            / (2.0 * du);
        (e * f_v - f * 0.5 * (f_u_j + f_u_jp1)) / sqrt_g
    }

    /// Advances the simulation by time step `dt` with **zero heap allocations**.
    ///
    /// Discretizes $\Delta_S f = \frac{1}{\sqrt{g}} \left[ \partial_u \left(\frac{G f_u - F f_v}{\sqrt{g}}\right) + \partial_v \left(\frac{E f_v - F f_u}{\sqrt{g}}\right) \right]$.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        assert!(
            dt.is_finite() && dt >= 0.0,
            "dt must be finite and non-negative"
        );
        let (nu, nv) = self.grid_res;
        assert_eq!(self.u_grid.len(), nu * nv);
        assert_eq!(self.next_grid.len(), nu * nv);

        let (du, dv) = self.grid_spacings();
        let j_range = if self.topology == SurfaceTopology::PolarSphere {
            1..nv.saturating_sub(1)
        } else {
            0..nv
        };

        for i in 0..nu {
            let u = self.range_u.0 + i as f64 * du;
            for j in j_range.clone() {
                let v = self.range_v.0 + j as f64 * dv;
                let idx = flatten_2d_index(j, i, nv);

                let area = self.surface.area_element(u, v);
                let laplacian = if area <= 1e-12 || !area.is_finite() {
                    0.0
                } else {
                    let flux_u_plus = self.compute_flux_u(i as i32, j as i32, du, dv);
                    let flux_u_minus = self.compute_flux_u(i as i32 - 1, j as i32, du, dv);
                    let flux_v_plus = self.compute_flux_v(i as i32, j as i32, du, dv);
                    let flux_v_minus = self.compute_flux_v(i as i32, j as i32 - 1, du, dv);
                    ((flux_u_plus - flux_u_minus) / du + (flux_v_plus - flux_v_minus) / dv) / area
                };

                self.next_grid[idx] = self.u_grid[idx] + dt * self.alpha * laplacian;
            }
        }
        if self.topology == SurfaceTopology::PolarSphere && nv >= 3 {
            self.step_polar_caps(dt, dv);
        }
        std::mem::swap(&mut self.u_grid, &mut self.next_grid);
    }

    fn step_polar_caps(&mut self, dt: f64, dv: f64) {
        let (nu, nv) = self.grid_res;
        for (pole_j, ring_j, probe_v) in [
            (0_usize, 1_usize, self.range_v.0 + 1e-4),
            (nv - 1, nv - 2, self.range_v.1 - 1e-4),
        ] {
            let (_, _, g_mer) = self.surface.first_fundamental_form(self.range_u.0, probe_v);
            let ring_mean: f64 =
                (0..nu).map(|i| self.value(i, ring_j)).sum::<f64>() / nu.max(1) as f64;
            let pole_val = self.value(0, pole_j);
            let lap = 4.0 * (ring_mean - pole_val) / (g_mer.max(1e-6) * dv * dv);
            let updated = pole_val + dt * self.alpha * lap;
            for i in 0..nu {
                self.next_grid[flatten_2d_index(pole_j, i, nv)] = updated;
            }
        }
    }
}
