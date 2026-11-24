use super::buckled_flow::{SurfaceTopology, detect_twisted_seam_map};
use super::surface::{ParametricSurface, SurfaceAnalysis};

/// Solves the Heat Equation $\frac{\partial u}{\partial t} = \alpha \Delta_S u$ on a parametric surface.
/// Uses explicit Euler time-stepping with a discrete divergence-form stencil for the Laplace-Beltrami operator.
pub struct HeatEquationSolver<'a, S: ParametricSurface> {
    #[allow(missing_docs)]
    pub surface: &'a S,
    #[allow(missing_docs)]
    pub alpha: f64, // Thermal diffusivity
    #[allow(missing_docs)]
    pub u_grid: Vec<Vec<f64>>, // Current temperature state
    #[allow(missing_docs)]
    pub next_grid: Vec<Vec<f64>>, // Bolt Optimization: Buffer to avoid allocations in step
    #[allow(missing_docs)]
    pub grid_res: (usize, usize), // (n_u, n_v)
    #[allow(missing_docs)]
    pub range_u: (f64, f64),
    #[allow(missing_docs)]
    pub range_v: (f64, f64),
    /// Boundary topology applied across parameter seams and polar caps.
    pub topology: SurfaceTopology,
    seam_fwd: Vec<usize>,
    seam_rev: Vec<usize>,
}

impl<'a, S: ParametricSurface> HeatEquationSolver<'a, S> {
    /// Creates a new Heat Equation solver for a given parametric surface.
    ///
    /// Initializes the temperature grid $u(u, v)$ based on the provided `initial_condition`.
    /// The solver uses a uniform grid in the parameter space $(u, v)$.
    ///
    /// # Arguments
    ///
    /// * `surface` - The parametric surface geometry (e.g., Sphere, Torus).
    /// * `alpha` - Thermal diffusivity coefficient $\alpha$. Higher values mean heat spreads faster.
    /// * `range_u` - The domain range for the $u$ parameter $(u_{min}, u_{max})$.
    /// * `range_v` - The domain range for the $v$ parameter $(v_{min}, v_{max})$.
    /// * `grid_res` - The resolution of the simulation grid $(n_u, n_v)$. Must be at least $(2, 2)$ to avoid division by zero or infinity.
    /// * `initial_condition` - A closure `Fn(u, v) -> temp` defining the initial temperature at $t=0$.
    ///
    /// # Examples
    ///
    /// Simulating heat diffusion on a unit Sphere:
    ///
    /// ```rust
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::differential_geometry::surface::Sphere;
    /// use math_explorer::pure_math::differential_geometry::heat_equation::HeatEquationSolver;
    /// use std::f64::consts::PI;
    ///
    /// // 1. Define the surface
    /// let sphere = Sphere { radius: 1.0 };
    ///
    /// // 2. Define initial temperature: A hot spot at the north pole (v close to 0)
    /// let initial_hot_spot = |_u: f64, v: f64| -> f64 {
    ///     if v < 0.5 { 100.0 } else { 0.0 }
    /// };
    ///
    /// // 3. Initialize the solver
    /// let mut solver = HeatEquationSolver::new(
    ///     &sphere,
    ///     0.01,             // Alpha (diffusivity)
    ///     (0.0, 2.0 * PI), // u range (0 to 2pi)
    ///     (0.01, PI - 0.01), // v range (avoid poles due to singularity)
    ///     (20, 20),        // Grid resolution
    ///     initial_hot_spot
    /// );
    ///
    /// // 4. Run the simulation
    /// let initial_max = solver.u_grid.iter().flatten().fold(0.0/0.0, |a: f64, b| a.max(*b));
    /// assert!(initial_max > 99.0);
    ///
    /// // Run with small time step to ensure stability (CFL condition)
    /// for _ in 0..10 {
    ///     solver.step(0.001);
    /// }
    ///
    /// let final_max = solver.u_grid.iter().flatten().fold(0.0/0.0, |a: f64, b| a.max(*b));
    ///
    /// // Heat should have diffused, lowering the peak temperature
    /// assert!(final_max < initial_max);
    /// ```
    #[verified_engine::verified(alloc_ok = "constructor pre-allocates state")]
    pub fn new(
        surface: &'a S,
        alpha: f64,
        range_u: (f64, f64),
        range_v: (f64, f64),
        grid_res: (usize, usize),
        initial_condition: impl Fn(f64, f64) -> f64,
    ) -> Self {
        let (nu, nv) = grid_res;
        let mut grid = vec![vec![0.0; nv]; nu];

        let du = (range_u.1 - range_u.0) / (nu as f64 - 1.0);
        let dv = (range_v.1 - range_v.0) / (nv as f64 - 1.0);

        for (i, row) in grid.iter_mut().enumerate().take(nu) {
            for (j, val) in row.iter_mut().enumerate().take(nv) {
                let u = range_u.0 + i as f64 * du;
                let v = range_v.0 + j as f64 * dv;
                *val = initial_condition(u, v);
            }
        }

        Self {
            surface,
            alpha,
            u_grid: grid.clone(),
            next_grid: grid,
            grid_res,
            range_u,
            range_v,
            topology: SurfaceTopology::OpenNeumann,
            seam_fwd: (0..nv).collect(),
            seam_rev: (0..nv).collect(),
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
                let cap_mean: f64 =
                    (0..nu).map(|i| self.u_grid[i][pole_j]).sum::<f64>() / nu as f64;
                for row in self.u_grid.iter_mut().take(nu) {
                    row[pole_j] = cap_mean;
                }
            }
        }
        self
    }

    fn grid_spacings(&self) -> (f64, f64) {
        self.topology
            .grid_spacings(self.range_u, self.range_v, self.grid_res.0, self.grid_res.1)
    }

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
        self.u_grid[wi][wj]
    }

    fn compute_flux_u(&self, i_base: i32, j_base: i32, du: f64, dv: f64) -> f64 {
        let nu = self.grid_res.0 as i32;
        if self.topology == SurfaceTopology::OpenNeumann && (i_base < 0 || i_base + 1 >= nu) {
            return 0.0;
        }
        let u_half = self.range_u.0 + (i_base as f64 + 0.5) * du;
        let v_curr = self.range_v.0 + j_base as f64 * dv;
        let (e, f, g_metric) = self.surface.first_fundamental_form(u_half, v_curr);
        let sqrt_g = (e * g_metric - f * f).max(1e-12).sqrt();
        let f_u = (self.sample_cell(i_base + 1, j_base) - self.sample_cell(i_base, j_base)) / du;
        let nv = self.grid_res.1;
        let seam_reverses = nv >= 3 && (self.seam_fwd[1] + nv - self.seam_fwd[0]) % nv == nv - 1;
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

    fn compute_flux_v(&self, i_base: i32, j_base: i32, du: f64, dv: f64) -> f64 {
        let nv = self.grid_res.1 as i32;
        if self.topology == SurfaceTopology::OpenNeumann && (j_base < 0 || j_base + 1 >= nv) {
            return 0.0;
        }
        let u_curr = self.range_u.0 + i_base as f64 * du;
        let v_half = self.range_v.0 + (j_base as f64 + 0.5) * dv;
        let (e, f, g_metric) = self.surface.first_fundamental_form(u_curr, v_half);
        let sqrt_g = (e * g_metric - f * f).max(1e-12).sqrt();
        let f_v = (self.sample_cell(i_base, j_base + 1) - self.sample_cell(i_base, j_base)) / dv;
        let f_u_j = (self.sample_cell(i_base + 1, j_base) - self.sample_cell(i_base - 1, j_base))
            / (2.0 * du);
        let f_u_jp1 = (self.sample_cell(i_base + 1, j_base + 1)
            - self.sample_cell(i_base - 1, j_base + 1))
            / (2.0 * du);
        (e * f_v - f * 0.5 * (f_u_j + f_u_jp1)) / sqrt_g
    }

    /// Advances the simulation by time step `dt`.
    /// Uses $\Delta_S f = \frac{1}{\sqrt{g}} [ \partial_u (\frac{G f_u - F f_v}{\sqrt{g}}) + \partial_v (\frac{E f_v - F f_u}{\sqrt{g}}) ]$
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let (nu, nv) = self.grid_res;
        let (du, dv) = self.grid_spacings();
        let j_range = if self.topology == SurfaceTopology::PolarSphere {
            1..nv.saturating_sub(1)
        } else {
            0..nv
        };

        #[allow(clippy::needless_range_loop)]
        for i in 0..nu {
            for j in j_range.clone() {
                let u = self.range_u.0 + i as f64 * du;
                let v = self.range_v.0 + j as f64 * dv;
                let flux_u_plus = self.compute_flux_u(i as i32, j as i32, du, dv);
                let flux_u_minus = self.compute_flux_u(i as i32 - 1, j as i32, du, dv);
                let flux_v_plus = self.compute_flux_v(i as i32, j as i32, du, dv);
                let flux_v_minus = self.compute_flux_v(i as i32, j as i32 - 1, du, dv);
                let area = self.surface.area_element(u, v).max(1e-12);
                let laplacian =
                    ((flux_u_plus - flux_u_minus) / du + (flux_v_plus - flux_v_minus) / dv) / area;
                self.next_grid[i][j] = self.u_grid[i][j] + dt * self.alpha * laplacian;
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
                (0..nu).map(|i| self.u_grid[i][ring_j]).sum::<f64>() / nu.max(1) as f64;
            let pole_val = self.u_grid[0][pole_j];
            let lap = 4.0 * (ring_mean - pole_val) / (g_mer.max(1e-6) * dv * dv);
            let updated = pole_val + dt * self.alpha * lap;
            for row in self.next_grid.iter_mut().take(nu) {
                row[pole_j] = updated;
            }
        }
    }
}
