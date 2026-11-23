use super::surface::{ParametricSurface, SurfaceAnalysis};

/// Solves the Heat Equation $\frac{\partial u}{\partial t} = \alpha \Delta_S u$ on a parametric surface.
/// Uses explicit Euler time-stepping with a discrete 5-point stencil for the Laplace-Beltrami operator.
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
        }
    }

    #[verified_engine::verified]
    fn sample_clamped(&self, i: i32, j: i32) -> f64 {
        let (nu, nv) = self.grid_res;
        let i_idx = i.clamp(0, (nu - 1) as i32) as usize;
        let j_idx = j.clamp(0, (nv - 1) as i32) as usize;
        self.u_grid[i_idx][j_idx]
    }

    #[verified_engine::verified]
    fn flux_u_at(&self, i_base: i32, j_base: i32, du: f64, dv: f64) -> f64 {
        let u_half = self.range_u.0 + (i_base as f64 + 0.5) * du;
        let v_curr = self.range_v.0 + j_base as f64 * dv;
        let (e, f, g_metric) = self.surface.first_fundamental_form(u_half, v_curr);
        let det_g = e * g_metric - f * f;
        if det_g.abs() < 1e-12 {
            return 0.0;
        }
        let sqrt_g = det_g.sqrt();
        let f_u =
            (self.sample_clamped(i_base + 1, j_base) - self.sample_clamped(i_base, j_base)) / du;
        let f_v_i = (self.sample_clamped(i_base, j_base + 1)
            - self.sample_clamped(i_base, j_base - 1))
            / (2.0 * dv);
        let f_v_ip1 = (self.sample_clamped(i_base + 1, j_base + 1)
            - self.sample_clamped(i_base + 1, j_base - 1))
            / (2.0 * dv);
        let f_v = 0.5 * (f_v_i + f_v_ip1);
        (g_metric * f_u - f * f_v) / sqrt_g
    }

    #[verified_engine::verified]
    fn flux_v_at(&self, i_base: i32, j_base: i32, du: f64, dv: f64) -> f64 {
        let u_curr = self.range_u.0 + i_base as f64 * du;
        let v_half = self.range_v.0 + (j_base as f64 + 0.5) * dv;
        let (e, f, g_metric) = self.surface.first_fundamental_form(u_curr, v_half);
        let det_g = e * g_metric - f * f;
        if det_g.abs() < 1e-12 {
            return 0.0;
        }
        let sqrt_g = det_g.sqrt();
        let f_v =
            (self.sample_clamped(i_base, j_base + 1) - self.sample_clamped(i_base, j_base)) / dv;
        let f_u_j = (self.sample_clamped(i_base + 1, j_base)
            - self.sample_clamped(i_base - 1, j_base))
            / (2.0 * du);
        let f_u_jp1 = (self.sample_clamped(i_base + 1, j_base + 1)
            - self.sample_clamped(i_base - 1, j_base + 1))
            / (2.0 * du);
        let f_u = 0.5 * (f_u_j + f_u_jp1);
        (e * f_v - f * f_u) / sqrt_g
    }

    /// Advances the simulation by time step `dt`.
    /// Uses $\Delta_S f = \frac{1}{\sqrt{g}} [ \partial_u (\frac{G f_u - F f_v}{\sqrt{g}}) + \partial_v (\frac{E f_v - F f_u}{\sqrt{g}}) ]$
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let (nu, nv) = self.grid_res;
        if nu < 2 || nv < 2 {
            return;
        }
        let du = (self.range_u.1 - self.range_u.0) / (nu as f64 - 1.0);
        let dv = (self.range_v.1 - self.range_v.0) / (nv as f64 - 1.0);

        #[allow(clippy::needless_range_loop)]
        for i in 0..nu {
            for j in 0..nv {
                let u = self.range_u.0 + i as f64 * du;
                let v = self.range_v.0 + j as f64 * dv;
                let (e, f, g_metric) = self.surface.first_fundamental_form(u, v);
                let det_g = e * g_metric - f * f;
                let laplacian = if det_g.abs() < 1e-12 {
                    0.0
                } else {
                    let flux_u_plus = self.flux_u_at(i as i32, j as i32, du, dv);
                    let flux_u_minus = self.flux_u_at(i as i32 - 1, j as i32, du, dv);
                    let flux_v_plus = self.flux_v_at(i as i32, j as i32, du, dv);
                    let flux_v_minus = self.flux_v_at(i as i32, j as i32 - 1, du, dv);
                    let term1 = (flux_u_plus - flux_u_minus) / du;
                    let term2 = (flux_v_plus - flux_v_minus) / dv;
                    let area = self.surface.area_element(u, v).max(1e-12);
                    (term1 + term2) / area
                };

                self.next_grid[i][j] = self.u_grid[i][j] + dt * self.alpha * laplacian;
            }
        }

        std::mem::swap(&mut self.u_grid, &mut self.next_grid);
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use crate::pure_math::differential_geometry::surface::Sphere;
    use std::f64::consts::PI;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_heat_equation_solver_polar_domain_no_nan_and_zero_alloc() {
        let sphere = Sphere { radius: 1.0 };
        let mut solver = HeatEquationSolver::new(
            &sphere,
            0.01,
            (0.0, 2.0 * PI),
            (0.0, PI),
            (16, 16),
            |u, v| v.cos() + 0.25 * u.sin() * v.sin(),
        );

        let ((), _metrics) = VerifiedEngine::run_verified(|| {
            for _ in 0..25 {
                solver.step(1e-4);
            }
        });

        for row in &solver.u_grid {
            for &val in row {
                assert!(
                    val.is_finite(),
                    "Pole-inclusive grid produced non-finite value {val}"
                );
            }
        }
    }
}
