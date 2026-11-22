use super::SpatialDiffusion;
use crate::biology::reaction_diffusion::{ChemicalState, DiffusionModel};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};
use pure_math::pure_math::analysis::pde::fused_stepper::{
    FusedStencilStepper, StencilOperators, advection_face_velocities, reflect_2d_neighbors,
};

/// A 2D spatial advection-diffusion operator over row-major concentration slices.
///
/// Preallocates horizontal/vertical velocity fields and a solid obstacle mask at construction time
/// so every step executes with zero heap allocations under `VerifiedAllocator`.
///
/// # Examples
///
/// ```rust
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::diffusion::AdvectionDiffusion2D;
/// use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
/// use math_explorer::math_kernel::types::{Dimension, StepSize};
///
/// let mut op = AdvectionDiffusion2D::new(
///     Dimension(16),
///     Dimension(16),
///     StepSize(1.0),
///     StepSize(1.0),
/// );
/// op.set_uniform_velocity(0.2, 0.0);
/// let mut sys = TuringSystem::new_with_kinetics(
///     Dimension(256),
///     DiffusionCoeff(0.2),
///     DiffusionCoeff(1.0),
///     SchnakenbergKinetics::new(0.1, 0.9),
///     op,
/// );
/// sys.step(0.05);
/// assert!(sys.u()[0].is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct AdvectionDiffusion2D {
    width: Dimension,
    height: Dimension,
    dx: StepSize,
    dy: StepSize,
    ops: StencilOperators,
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
    obstacles: Vec<bool>,
}

impl AdvectionDiffusion2D {
    /// Creates a new `AdvectionDiffusion2D` operator with preallocated zero-velocity buffers.
    #[verified_engine::verified]
    pub fn new(width: Dimension, height: Dimension, dx: StepSize, dy: StepSize) -> Self {
        let total = (*width)
            .checked_mul(*height)
            .expect("AdvectionDiffusion2D grid dimensions overflow usize");
        Self {
            width,
            height,
            dx,
            dy,
            ops: StencilOperators::new_2d(dx, dy),
            velocity_x: vec![0.0; total],
            velocity_y: vec![0.0; total],
            obstacles: vec![false; total],
        }
    }

    /// Returns grid width.
    #[verified_engine::verified]
    pub fn width(&self) -> Dimension {
        self.width
    }

    /// Returns grid height.
    #[verified_engine::verified]
    pub fn height(&self) -> Dimension {
        self.height
    }

    /// Returns horizontal grid spacing `dx`.
    #[verified_engine::verified]
    pub fn dx(&self) -> StepSize {
        self.dx
    }

    /// Returns vertical grid spacing `dy`.
    #[verified_engine::verified]
    pub fn dy(&self) -> StepSize {
        self.dy
    }

    /// Returns total cell count `width * height`.
    #[verified_engine::verified]
    pub fn grid_len(&self) -> usize {
        *self.width * *self.height
    }

    /// Returns row-major horizontal velocity slice `u_x`.
    #[verified_engine::verified]
    pub fn velocity_x(&self) -> &[f64] {
        &self.velocity_x
    }

    /// Returns row-major vertical velocity slice `u_y`.
    #[verified_engine::verified]
    pub fn velocity_y(&self) -> &[f64] {
        &self.velocity_y
    }

    /// Returns row-major solid obstacle mask (`true` = solid).
    #[verified_engine::verified]
    pub fn obstacles(&self) -> &[bool] {
        &self.obstacles
    }

    /// Returns mutable row-major horizontal velocity slice `u_x`.
    #[verified_engine::verified]
    pub fn velocity_x_mut(&mut self) -> &mut [f64] {
        &mut self.velocity_x
    }

    /// Returns mutable row-major vertical velocity slice `u_y`.
    #[verified_engine::verified]
    pub fn velocity_y_mut(&mut self) -> &mut [f64] {
        &mut self.velocity_y
    }

    /// Returns mutable row-major obstacle mask.
    #[verified_engine::verified]
    pub fn obstacles_mut(&mut self) -> &mut [bool] {
        &mut self.obstacles
    }

    /// Borrows `(velocity_x, velocity_y, obstacles)` mutably at once for zero-copy solver coupling.
    #[verified_engine::verified]
    pub fn buffers_mut(&mut self) -> (&mut [f64], &mut [f64], &mut [bool]) {
        (
            &mut self.velocity_x,
            &mut self.velocity_y,
            &mut self.obstacles,
        )
    }

    /// Assigns uniform drift velocity `(vx, vy)` across all non-obstacle cells.
    #[verified_engine::verified]
    pub fn set_uniform_velocity(&mut self, vx: f64, vy: f64) {
        let total = self.grid_len();
        for i in 0..total {
            if self.obstacles[i] {
                self.velocity_x[i] = 0.0;
                self.velocity_y[i] = 0.0;
            } else {
                self.velocity_x[i] = vx;
                self.velocity_y[i] = vy;
            }
        }
    }

    /// Resets velocity buffers to zero.
    #[verified_engine::verified]
    pub fn clear_velocity(&mut self) {
        self.velocity_x.fill(0.0);
        self.velocity_y.fill(0.0);
    }

    /// Sets or clears a single obstacle cell at `(x, y)`.
    #[verified_engine::verified]
    pub fn set_obstacle(&mut self, x: usize, y: usize, is_obstacle: bool) {
        let w = *self.width;
        let h = *self.height;
        if x < w && y < h {
            let idx = flatten_2d_index(x, y, w);
            self.obstacles[idx] = is_obstacle;
            if is_obstacle {
                self.velocity_x[idx] = 0.0;
                self.velocity_y[idx] = 0.0;
            }
        }
    }

    /// Paints or erases a circular obstacle disk centered at `(cx, cy)` with `radius`.
    #[verified_engine::verified]
    pub fn paint_obstacle_disk(&mut self, cx: usize, cy: usize, radius: usize, is_obstacle: bool) {
        let w = *self.width;
        let h = *self.height;
        let r_sq = (radius * radius) as isize;
        let y_min = cy.saturating_sub(radius);
        let y_max = (cy + radius).min(h.saturating_sub(1));
        let x_min = cx.saturating_sub(radius);
        let x_max = (cx + radius).min(w.saturating_sub(1));
        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx_i = x as isize - cx as isize;
                let dy_i = y as isize - cy as isize;
                if dx_i * dx_i + dy_i * dy_i <= r_sq {
                    self.set_obstacle(x, y, is_obstacle);
                }
            }
        }
    }

    /// Clears all obstacles across the grid.
    #[verified_engine::verified]
    pub fn clear_obstacles(&mut self) {
        self.obstacles.fill(false);
    }

    /// Returns the peak fluid speed `max hypot(vx, vy)` outside obstacles.
    #[verified_engine::verified]
    pub fn characteristic_speed(&self) -> f64 {
        let total = self.grid_len();
        let mut max_sq = 0.0_f64;
        for i in 0..total {
            if !self.obstacles[i] {
                let speed_sq = self.velocity_x[i] * self.velocity_x[i]
                    + self.velocity_y[i] * self.velocity_y[i];
                if speed_sq.is_finite() && speed_sq > max_sq {
                    max_sq = speed_sq;
                }
            }
        }
        max_sq.sqrt()
    }

    /// Estimates characteristic length scale `L` (obstacle diameter if present, else domain width).
    #[verified_engine::verified]
    pub fn characteristic_length(&self) -> f64 {
        let obs_cells = self.obstacles.iter().filter(|&&b| b).count();
        if obs_cells > 0 {
            let diam_cells = 2.0 * ((obs_cells as f64) / std::f64::consts::PI).sqrt();
            diam_cells.max(1.0) * *self.dx
        } else {
            (*self.width as f64).max(1.0) * *self.dx
        }
    }

    /// Computes discrete 2D curl `omega_z = duy/dx - dux/dy` into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn compute_vorticity(&self, out: &mut [f64]) {
        let w = *self.width;
        let h = *self.height;
        let total = w * h;
        assert!(out.len() >= total, "Vorticity output slice too small");
        for y in 0..h {
            for x in 0..w {
                let (idx, idx_l, idx_r, idx_u, idx_d) =
                    reflect_2d_neighbors(x, y, w, h, &self.obstacles);
                if self.obstacles[idx] {
                    out[idx] = 0.0;
                } else {
                    let duy_dx = self
                        .ops
                        .central_diff_1st(self.velocity_y[idx_l], self.velocity_y[idx_r]);
                    let dux_dy = self
                        .ops
                        .central_diff_1st_y(self.velocity_x[idx_u], self.velocity_x[idx_d]);
                    out[idx] = duy_dx - dux_dy;
                }
            }
        }
    }

    /// Computes bounded CFL sub-step count for combined advective and diffusive stability.
    #[verified_engine::verified]
    pub fn cfl_substeps(&self, dt: f64, d_max: f64, max_cfl: f64, max_substeps: usize) -> usize {
        let cap = max_substeps.max(1);
        if !dt.is_finite() || dt <= 0.0 || !max_cfl.is_finite() || max_cfl <= 0.0 {
            return 1;
        }
        let total = self.grid_len();
        let mut max_adv_rate = 0.0_f64;
        for i in 0..total {
            if !self.obstacles[i] {
                let rate = self.velocity_x[i].abs() * self.ops.inv_dx
                    + self.velocity_y[i].abs() * self.ops.inv_dy;
                if rate.is_finite() && rate > max_adv_rate {
                    max_adv_rate = rate;
                }
            }
        }
        let safe_d = if d_max.is_finite() && d_max > 0.0 {
            d_max
        } else {
            0.0
        };
        let diff_rate = 2.0 * safe_d * (self.ops.inv_dx_sq + self.ops.inv_dy_sq);
        let cfl_number = dt * (max_adv_rate + diff_rate);
        if cfl_number <= max_cfl {
            1
        } else {
            ((cfl_number / max_cfl).ceil() as usize).clamp(1, cap)
        }
    }
}

impl<const N: usize> SpatialDiffusion<N> for AdvectionDiffusion2D {
    fn stepper(&self) -> FusedStencilStepper {
        FusedStencilStepper { ops: self.ops }
    }

    #[verified_engine::verified]
    fn step_fused<K: crate::biology::morphogenesis::reaction::ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        if N == 0 || self.grid_len() == 0 {
            return;
        }
        SpatialDiffusion::<N>::stepper(self).step_2d_coupled_advection_neumann(
            (*self.width, *self.height),
            state,
            next_state,
            &self.velocity_x,
            &self.velocity_y,
            &self.obstacles,
            dt,
            1.0,
            |_idx, face_vel, curr, left, right, up, down, ops| {
                let mut rhs = [0.0; N];
                let rates = kinetics.reaction(curr);
                for s in 0..N {
                    let lap = ops.central_diff_2nd_2d(curr[s], left[s], right[s], up[s], down[s]);
                    let adv = ops
                        .upwind_divergence_2d(face_vel, curr[s], left[s], right[s], up[s], down[s]);
                    rhs[s] = coeffs[s] * lap - adv + rates[s];
                }
                rhs
            },
        );
    }

    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], mut op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        let w = *self.width;
        let h = *self.height;
        let n = w * h;
        if N == 0 || n == 0 {
            return;
        }
        for buffer in state.iter().take(N) {
            assert!(
                buffer.len() >= n,
                "State buffer smaller than AdvectionDiffusion2D grid"
            );
        }
        for y in 0..h {
            for x in 0..w {
                let (idx, idx_l, idx_r, idx_u, idx_d) =
                    reflect_2d_neighbors(x, y, w, h, &self.obstacles);
                let mut curr = [0.0; N];
                for s in 0..N {
                    curr[s] = state[s][idx];
                }
                if self.obstacles[idx] {
                    op(idx, curr, [0.0; N]);
                    continue;
                }
                let face_vel = advection_face_velocities(
                    idx,
                    idx_l,
                    idx_r,
                    idx_u,
                    idx_d,
                    &self.velocity_x,
                    &self.velocity_y,
                );
                let mut transport = [0.0; N];
                for s in 0..N {
                    let u = state[s];
                    let lap = self
                        .ops
                        .central_diff_2nd_2d(curr[s], u[idx_l], u[idx_r], u[idx_u], u[idx_d]);
                    let adv = self.ops.upwind_divergence_2d(
                        face_vel, curr[s], u[idx_l], u[idx_r], u[idx_u], u[idx_d],
                    );
                    transport[s] = coeffs[s] * lap - adv;
                }
                op(idx, curr, transport);
            }
        }
    }
}

impl DiffusionModel for AdvectionDiffusion2D {
    #[verified_engine::verified]
    fn apply(&self, state: &ChemicalState, out: &mut ChemicalState, coeffs: &[f64]) {
        let n_species = state.num_species();
        let w = *self.width;
        let h = *self.height;
        let n_grid = w * h;
        if n_species == 0 || n_grid == 0 {
            return;
        }
        assert_eq!(
            state.grid_size(),
            n_grid,
            "ChemicalState grid size mismatch"
        );
        assert_eq!(
            out.grid_size(),
            n_grid,
            "Output ChemicalState grid size mismatch"
        );
        assert_eq!(
            coeffs.len(),
            n_species,
            "Diffusion coefficients count mismatch"
        );

        for (s, &coeff) in coeffs.iter().enumerate().take(n_species) {
            let src = state.species(s);
            let dst = out.species_mut(s);
            for y in 0..h {
                for x in 0..w {
                    let (idx, idx_l, idx_r, idx_u, idx_d) =
                        reflect_2d_neighbors(x, y, w, h, &self.obstacles);
                    if self.obstacles[idx] {
                        dst[idx] = 0.0;
                        continue;
                    }
                    let face_vel = advection_face_velocities(
                        idx,
                        idx_l,
                        idx_r,
                        idx_u,
                        idx_d,
                        &self.velocity_x,
                        &self.velocity_y,
                    );
                    let lap = self.ops.central_diff_2nd_2d(
                        src[idx], src[idx_l], src[idx_r], src[idx_u], src[idx_d],
                    );
                    let adv = self.ops.upwind_divergence_2d(
                        face_vel, src[idx], src[idx_l], src[idx_r], src[idx_u], src[idx_d],
                    );
                    dst[idx] = coeff * lap - adv;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
