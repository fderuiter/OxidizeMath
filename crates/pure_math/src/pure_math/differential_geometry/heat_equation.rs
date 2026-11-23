use super::surface::{ParametricSurface, SurfaceAnalysis};

/// Solves the Heat Equation $\frac{\partial u}{\partial t} = \alpha \Delta_S u$ on a parametric surface.
/// Uses explicit Euler time-stepping with a discrete 5-point stencil for the Laplace-Beltrami operator.
#[derive(Debug, Clone, PartialEq)]
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

    /// Advances the simulation by time step `dt`.
    /// Uses $\Delta_S f = \frac{1}{\sqrt{g}} [ \partial_u (\frac{G f_u - F f_v}{\sqrt{g}}) + \partial_v (\frac{E f_v - F f_u}{\sqrt{g}}) ]$
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let (nu, nv) = self.grid_res;
        let du = (self.range_u.1 - self.range_u.0) / (nu as f64 - 1.0);
        let dv = (self.range_v.1 - self.range_v.0) / (nv as f64 - 1.0);

        // Bolt Optimization: Use pre-allocated buffer instead of cloning the grid every step

        // Safe getter with wrapping for periodicity (assuming closed surface or periodic domain like torus)
        // For general surfaces, Neumann conditions (replicate boundary) are often safer,
        // but for Sphere/Torus physics, wrapping is often desired on angles.
        // Let's implement clamp for generic stability in this library context.
        let get = |grid: &Vec<Vec<f64>>, i: i32, j: i32| -> f64 {
            let i_idx = i.clamp(0, (nu - 1) as i32) as usize;
            let j_idx = j.clamp(0, (nv - 1) as i32) as usize;
            grid[i_idx][j_idx]
        };

        #[allow(clippy::needless_range_loop)]
        for i in 0..nu {
            for j in 0..nv {
                let u = self.range_u.0 + i as f64 * du;
                let v = self.range_v.0 + j as f64 * dv;

                // 1. Calculate derivatives of f at (i, j)
                // We need half-step derivatives for the divergence.
                // Or we can use central differences for the whole operator.
                // Let's use central differences for the outer derivatives and central for inner.
                // Standard discretization of divergence form:
                // d_u (A f_u) ~ (A_{i+1/2} (f_{i+1} - f_i)/du - A_{i-1/2} (f_i - f_{i-1})/du) / du

                // Let's compute flux terms at half-points.

                // Flux U terms: A = (G f_u - F f_v) / sqrt(g)
                // We need A at (i+0.5, j) and (i-0.5, j)

                let compute_flux_u = |i_base: i32, j_base: i32| -> f64 {
                    // Evaluate metric at half step u
                    let u_half = self.range_u.0 + (i_base as f64 + 0.5) * du;
                    let v_curr = self.range_v.0 + j_base as f64 * dv;
                    let (e, f, g_metric) = self.surface.first_fundamental_form(u_half, v_curr);
                    let sqrt_g = (e * g_metric - f * f).sqrt();

                    // f_u at i+0.5 is (f_{i+1} - f_i) / du
                    let f_u = (get(&self.u_grid, i_base + 1, j_base)
                        - get(&self.u_grid, i_base, j_base))
                        / du;

                    // f_v at i+0.5 is average of f_v at i and i+1?
                    // f_v at i = (f_{i, j+1} - f_{i, j-1}) / 2dv
                    let f_v_i = (get(&self.u_grid, i_base, j_base + 1)
                        - get(&self.u_grid, i_base, j_base - 1))
                        / (2.0 * dv);
                    let f_v_ip1 = (get(&self.u_grid, i_base + 1, j_base + 1)
                        - get(&self.u_grid, i_base + 1, j_base - 1))
                        / (2.0 * dv);
                    let f_v = 0.5 * (f_v_i + f_v_ip1);

                    (g_metric * f_u - f * f_v) / sqrt_g
                };

                let flux_u_plus = compute_flux_u(i as i32, j as i32);
                let flux_u_minus = compute_flux_u(i as i32 - 1, j as i32);

                // Flux V terms: B = (E f_v - F f_u) / sqrt(g)
                // We need B at (i, j+0.5) and (i, j-0.5)

                let compute_flux_v = |i_base: i32, j_base: i32| -> f64 {
                    let u_curr = self.range_u.0 + i_base as f64 * du;
                    let v_half = self.range_v.0 + (j_base as f64 + 0.5) * dv;
                    let (e, f, g_metric) = self.surface.first_fundamental_form(u_curr, v_half);
                    let sqrt_g = (e * g_metric - f * f).sqrt();

                    // f_v at j+0.5 is (f_{j+1} - f_j) / dv
                    let f_v = (get(&self.u_grid, i_base, j_base + 1)
                        - get(&self.u_grid, i_base, j_base))
                        / dv;

                    // f_u at j+0.5 is average
                    let f_u_j = (get(&self.u_grid, i_base + 1, j_base)
                        - get(&self.u_grid, i_base - 1, j_base))
                        / (2.0 * du);
                    let f_u_jp1 = (get(&self.u_grid, i_base + 1, j_base + 1)
                        - get(&self.u_grid, i_base - 1, j_base + 1))
                        / (2.0 * du);
                    let f_u = 0.5 * (f_u_j + f_u_jp1);

                    (e * f_v - f * f_u) / sqrt_g
                };

                let flux_v_plus = compute_flux_v(i as i32, j as i32);
                let flux_v_minus = compute_flux_v(i as i32, j as i32 - 1);

                // Divergence
                let term1 = (flux_u_plus - flux_u_minus) / du;
                let term2 = (flux_v_plus - flux_v_minus) / dv;

                let area = self.surface.area_element(u, v);
                let laplacian = (1.0 / area) * (term1 + term2);

                self.next_grid[i][j] = self.u_grid[i][j] + dt * self.alpha * laplacian;
            }
        }

        std::mem::swap(&mut self.u_grid, &mut self.next_grid);
    }
}
