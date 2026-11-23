//! Zero-allocation 2D Störmer-Verlet wave tank and Kirchhoff-Love biharmonic Chladni plate solver.
//!
//! Evaluates 2D wave propagation and thin-plate bending dynamics on pre-allocated contiguous
//! `Vec<f64>` grids using [`crate::pure_math::analysis::pde::fused_stepper::StencilOperators::new_2d`] without post-initialization heap allocations.
//!
//! # Example
//!
//! ```
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::analysis::pde::wave_2d::{
//!     BoundaryMode2D, Wave2DMedium, Wave2DParams, Wave2DSolver,
//! };
//!
//! let mut solver = Wave2DSolver::new_unit_square(32);
//! solver.inject_gaussian_pulse(0.5, 0.5, 0.08, 1.0);
//! let params = Wave2DParams {
//!     medium: Wave2DMedium::AcousticWave,
//!     boundary: BoundaryMode2D::Dirichlet,
//!     wave_speed: 1.0,
//!     plate_stiffness: 0.0,
//!     damping: 0.01,
//! };
//! let substeps = solver.step(&params, 0.004);
//! assert!(substeps >= 1);
//! assert!(solver.displacement().iter().all(|v| v.is_finite()));
//! ```

// [cite:advanced_linear_algebra]

use std::f64::consts::PI;

use math_commons::math_kernel::types::{StepSize, flatten_2d_index};

use super::fused_stepper::StencilOperators;

/// Configuration enums, parameter structs, and analytic reference formulas.
pub mod config;
/// Mask construction utilities for sponge layers, slits, lenses, and plate clamps.
pub mod masks;
/// Chladni sand particle drift and 2D Marching-Squares nodal contour extraction.
pub mod particles_and_contours;
/// Pre-planned zero-allocation 2D FFT spectral analyzer.
pub mod spectral;

#[cfg(test)]
mod tests;

pub use config::{
    BoundaryMode2D, DriverConfig2D, MAX_BIHARMONIC_NUMBER, MAX_COURANT_NUMBER, MAX_VERLET_SUBSTEPS,
    Wave2DMedium, Wave2DParams, analytic_clamped_square_omega, young_double_slit_fringe_spacing,
};
pub use masks::{
    apply_biconvex_lens, apply_circular_drumhead_mask, apply_multi_slit_barrier, apply_point_clamp,
    apply_sponge_layer,
};
pub use particles_and_contours::{ChladniSandSystem, MarchingSquares2D};
pub use spectral::{SpectralAnalyzer2D, SpectralMode2D, SpectralPeak2D};

/// Zero-allocation 2D Verlet PDE solver with pre-allocated mask and stencil buffers.
#[derive(Debug, Clone)]
pub struct Wave2DSolver {
    width: usize,
    height: usize,
    ops: StencilOperators,
    time: f64,
    u_prev: Vec<f64>,
    u_curr: Vec<f64>,
    u_next: Vec<f64>,
    laplacian_buf: Vec<f64>,
    energy_ema: Vec<f64>,
    modal_field: Vec<f64>,
    /// Solid wall / clamp mask (`1.0` = free medium, `0.0` = fixed nodal obstacle).
    pub wall_mask: Vec<f64>,
    /// Spatially varying absorbing sponge damping rate $\sigma(x, y) \ge 0$.
    pub sponge_mask: Vec<f64>,
    /// Spatially varying relative wave speed $s(x, y) = 1 / n(x, y)$.
    pub wave_speed_mask: Vec<f64>,
    detector_accum: Vec<f64>,
    detector_samples: usize,
    stationary_start: bool,
}

impl Wave2DSolver {
    /// Constructs a pre-allocated 2D solver with explicit cell spacings `dx` and `dy`.
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize, dx: f64, dy: f64) -> Self {
        assert!(width >= 4 && height >= 4, "Grid dimensions must be >= 4");
        assert!(dx > 0.0 && dy > 0.0, "Grid spacings must be positive");
        let len = width * height;
        Self {
            width,
            height,
            ops: StencilOperators::new_2d(StepSize(dx), StepSize(dy)),
            time: 0.0,
            u_prev: vec![0.0; len],
            u_curr: vec![0.0; len],
            u_next: vec![0.0; len],
            laplacian_buf: vec![0.0; len],
            energy_ema: vec![0.0; len],
            modal_field: vec![0.0; len],
            wall_mask: vec![1.0; len],
            sponge_mask: vec![0.0; len],
            wave_speed_mask: vec![1.0; len],
            detector_accum: vec![0.0; height],
            detector_samples: 0,
            stationary_start: false,
        }
    }

    /// Constructs a pre-allocated `size x size` solver over the unit square $[0, 1]^2$.
    #[verified_engine::verified]
    pub fn new_unit_square(size: usize) -> Self {
        assert!(size >= 4, "Square grid size must be >= 4");
        let h = 1.0 / ((size - 1) as f64);
        Self::new(size, size, h, h)
    }

    /// Grid width $W$ in cells.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }
    /// Grid height $H$ in cells.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }
    /// Horizontal cell spacing $\Delta x$.
    #[verified_engine::verified]
    pub fn dx(&self) -> f64 {
        *self.ops.dx
    }
    /// Vertical cell spacing $\Delta y$.
    #[verified_engine::verified]
    pub fn dy(&self) -> f64 {
        *self.ops.dy
    }
    /// Elapsed simulation time $t$.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Current displacement field $u(x, y, t)$.
    #[verified_engine::verified]
    pub fn displacement(&self) -> &[f64] {
        &self.u_curr
    }
    /// Exponentially smoothed wave-energy field $\langle u^2 \rangle$.
    #[verified_engine::verified]
    pub fn energy_field(&self) -> &[f64] {
        &self.energy_ema
    }
    /// Phase-demodulated standing-wave field for steady nodal contour extraction.
    #[verified_engine::verified]
    pub fn modal_field(&self) -> &[f64] {
        &self.modal_field
    }

    /// Initializes a stationary displacement field ($u(0) = \text{values}, u_t(0) = 0$).
    #[verified_engine::verified]
    pub fn set_stationary_displacement(&mut self, values: &[f64]) {
        assert_eq!(values.len(), self.u_curr.len(), "Field length mismatch");
        self.u_curr.copy_from_slice(values);
        self.u_prev.copy_from_slice(values);
        self.modal_field.copy_from_slice(values);
        for (e, &v) in self.energy_ema.iter_mut().zip(values.iter()) {
            *e = v * v;
        }
        self.u_next.fill(0.0);
        self.time = 0.0;
        self.stationary_start = true;
    }

    /// Clears wave displacement, energy envelope, and detector accumulators while keeping masks.
    #[verified_engine::verified]
    pub fn clear_wave_state(&mut self) {
        self.u_prev.fill(0.0);
        self.u_curr.fill(0.0);
        self.u_next.fill(0.0);
        self.laplacian_buf.fill(0.0);
        self.energy_ema.fill(0.0);
        self.modal_field.fill(0.0);
        self.detector_accum.fill(0.0);
        self.detector_samples = 0;
        self.time = 0.0;
        self.stationary_start = false;
    }

    /// Resets all masks (`wall_mask = 1`, `sponge_mask = 0`, `wave_speed_mask = 1`) and clears state.
    #[verified_engine::verified]
    pub fn reset_all(&mut self) {
        self.wall_mask.fill(1.0);
        self.sponge_mask.fill(0.0);
        self.wave_speed_mask.fill(1.0);
        self.clear_wave_state();
    }

    /// Resets only the time-integrated detector intensity accumulator.
    #[verified_engine::verified]
    pub fn reset_detector(&mut self) {
        self.detector_accum.fill(0.0);
        self.detector_samples = 0;
    }

    /// Computes `(substeps, dt_sub)` guaranteed to satisfy both wave Courant (`<= 0.65`)
    /// and biharmonic plate stability (`<= 0.18`) bounds.
    #[verified_engine::verified]
    pub fn safe_substep_plan(&self, params: &Wave2DParams, requested_dt: f64) -> (usize, f64) {
        let min_h = self.dx().min(self.dy());
        assert!(min_h > 0.0, "Minimum grid step must be positive");
        let dt_pos = if requested_dt.is_finite() && requested_dt > 0.0 {
            requested_dt
        } else {
            1.0e-4
        };
        let mut max_speed_scale = 1.0_f64;
        for &s in &self.wave_speed_mask {
            if s > max_speed_scale {
                max_speed_scale = s;
            }
        }
        let c_eff = params.wave_speed.abs() * max_speed_scale;
        let dt_wave = if c_eff > 1.0e-12 {
            MAX_COURANT_NUMBER * min_h / c_eff
        } else {
            dt_pos
        };
        let dt_plate =
            if params.medium == Wave2DMedium::BiharmonicPlate && params.plate_stiffness > 1.0e-18 {
                let d_sqrt = params.plate_stiffness.sqrt();
                let inv_sum = self.ops.inv_dx_sq + self.ops.inv_dy_sq;
                MAX_BIHARMONIC_NUMBER / (d_sqrt * inv_sum)
            } else {
                dt_pos
            };
        let dt_limit = dt_wave.min(dt_plate);
        if dt_pos <= dt_limit {
            return (1, dt_pos);
        }
        let needed = (dt_pos / dt_limit).ceil() as usize;
        let substeps = needed.clamp(1, MAX_VERLET_SUBSTEPS);
        let dt_sub = (dt_pos / (substeps as f64)).min(dt_limit);
        (substeps, dt_sub)
    }

    /// Effective wave Courant number $c \Delta t_{\text{sub}} / \min(\Delta x, \Delta y)$.
    #[verified_engine::verified]
    pub fn effective_courant(&self, params: &Wave2DParams, requested_dt: f64) -> f64 {
        let (_substeps, dt_sub) = self.safe_substep_plan(params, requested_dt);
        let min_h = self.dx().min(self.dy());
        params.wave_speed.abs() * dt_sub / min_h
    }

    /// Advances the 2D solver without external forcing over `requested_dt`, returning executed substeps.
    #[verified_engine::verified]
    pub fn step(&mut self, params: &Wave2DParams, requested_dt: f64) -> usize {
        let idle_driver = DriverConfig2D {
            enabled: false,
            ..DriverConfig2D::default()
        };
        self.step_driven(params, &idle_driver, requested_dt, None)
    }

    /// Advances the 2D solver with optional harmonic forcing and detector accumulation.
    #[verified_engine::verified]
    pub fn step_driven(
        &mut self,
        params: &Wave2DParams,
        driver: &DriverConfig2D,
        requested_dt: f64,
        detector_x: Option<usize>,
    ) -> usize {
        assert_eq!(self.u_curr.len(), self.width * self.height);
        let (substeps, dt_sub) = self.safe_substep_plan(params, requested_dt);
        for _ in 0..substeps {
            self.compute_laplacian_pass(params.boundary);
            self.verlet_substep(params, driver, dt_sub);
            if let Some(det_col) = detector_x {
                self.accumulate_detector_column(det_col);
            }
        }
        substeps
    }

    #[verified_engine::verified]
    fn compute_laplacian_pass(&mut self, boundary: BoundaryMode2D) {
        let w = self.width;
        let h = self.height;
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                self.laplacian_buf[idx] = self.eval_cell_laplacian(x, y, idx, boundary);
            }
        }
    }

    #[verified_engine::verified]
    fn eval_cell_laplacian(&self, x: usize, y: usize, idx: usize, boundary: BoundaryMode2D) -> f64 {
        let w = self.width;
        let h = self.height;
        let on_edge = x == 0 || x == w - 1 || y == 0 || y == h - 1;
        if self.wall_mask[idx] <= 0.5 && !on_edge {
            return 0.0;
        }
        if on_edge {
            return self.eval_boundary_laplacian(x, y, idx, boundary);
        }
        let curr = self.u_curr[idx];
        let left = self.u_curr[idx - 1];
        let right = self.u_curr[idx + 1];
        let up = self.u_curr[idx - w];
        let down = self.u_curr[idx + w];
        self.ops.central_diff_2nd_2d(curr, left, right, up, down)
    }

    #[verified_engine::verified]
    fn eval_boundary_laplacian(
        &self,
        x: usize,
        y: usize,
        idx: usize,
        boundary: BoundaryMode2D,
    ) -> f64 {
        let w = self.width;
        let h = self.height;
        match boundary {
            BoundaryMode2D::Dirichlet | BoundaryMode2D::Sponge => 0.0,
            BoundaryMode2D::Clamped => {
                let corner = (x == 0 || x == w - 1) && (y == 0 || y == h - 1);
                if corner {
                    0.0
                } else if x == 0 {
                    2.0 * self.u_curr[flatten_2d_index(1, y, w)] * self.ops.inv_dx_sq
                } else if x == w - 1 {
                    2.0 * self.u_curr[flatten_2d_index(w - 2, y, w)] * self.ops.inv_dx_sq
                } else if y == 0 {
                    2.0 * self.u_curr[flatten_2d_index(x, 1, w)] * self.ops.inv_dy_sq
                } else {
                    2.0 * self.u_curr[flatten_2d_index(x, h - 2, w)] * self.ops.inv_dy_sq
                }
            }
            BoundaryMode2D::Neumann => {
                if self.wall_mask[idx] <= 0.5 {
                    return 0.0;
                }
                let xl = if x > 0 { x - 1 } else { 1 };
                let xr = if x + 1 < w { x + 1 } else { w - 2 };
                let yu = if y > 0 { y - 1 } else { 1 };
                let yd = if y + 1 < h { y + 1 } else { h - 2 };
                self.ops.central_diff_2nd_2d(
                    self.u_curr[idx],
                    self.u_curr[flatten_2d_index(xl, y, w)],
                    self.u_curr[flatten_2d_index(xr, y, w)],
                    self.u_curr[flatten_2d_index(x, yu, w)],
                    self.u_curr[flatten_2d_index(x, yd, w)],
                )
            }
        }
    }

    #[verified_engine::verified]
    fn eval_biharmonic_at(&self, x: usize, y: usize, idx: usize, boundary: BoundaryMode2D) -> f64 {
        let w = self.width;
        let h = self.height;
        let on_edge = x == 0 || x == w - 1 || y == 0 || y == h - 1;
        if on_edge && boundary != BoundaryMode2D::Neumann {
            return 0.0;
        }
        let xl = if x > 0 { x - 1 } else { 1 };
        let xr = if x + 1 < w { x + 1 } else { w - 2 };
        let yu = if y > 0 { y - 1 } else { 1 };
        let yd = if y + 1 < h { y + 1 } else { h - 2 };
        self.ops.central_diff_2nd_2d(
            self.laplacian_buf[idx],
            self.laplacian_buf[flatten_2d_index(xl, y, w)],
            self.laplacian_buf[flatten_2d_index(xr, y, w)],
            self.laplacian_buf[flatten_2d_index(x, yu, w)],
            self.laplacian_buf[flatten_2d_index(x, yd, w)],
        )
    }

    #[verified_engine::verified]
    fn verlet_substep(&mut self, params: &Wave2DParams, driver: &DriverConfig2D, dt: f64) {
        let w = self.width;
        let h = self.height;
        let dt_sq = dt * dt;
        let phase = 2.0 * PI * driver.frequency * self.time;
        let osc = phase.sin();
        let ref_cos = phase.cos();
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                self.u_next[idx] =
                    self.advance_single_cell(x, y, idx, params, driver, osc, dt, dt_sq);
                let next_val = self.u_next[idx];
                self.energy_ema[idx] = 0.985 * self.energy_ema[idx] + 0.015 * (next_val * next_val);
                let lock_signal = if driver.enabled {
                    next_val * osc + next_val * ref_cos
                } else {
                    next_val
                };
                self.modal_field[idx] = 0.96 * self.modal_field[idx] + 0.04 * lock_signal;
            }
        }
        std::mem::swap(&mut self.u_prev, &mut self.u_curr);
        std::mem::swap(&mut self.u_curr, &mut self.u_next);
        self.stationary_start = false;
        self.time += dt;
    }

    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    fn advance_single_cell(
        &self,
        x: usize,
        y: usize,
        idx: usize,
        params: &Wave2DParams,
        driver: &DriverConfig2D,
        osc: f64,
        dt: f64,
        dt_sq: f64,
    ) -> f64 {
        let on_edge = x == 0 || x == self.width - 1 || y == 0 || y == self.height - 1;
        if self.wall_mask[idx] <= 0.5 || (on_edge && params.boundary != BoundaryMode2D::Neumann) {
            return 0.0;
        }
        let local_c = params.wave_speed * self.wave_speed_mask[idx];
        let mut accel = (local_c * local_c) * self.laplacian_buf[idx];
        if params.medium == Wave2DMedium::BiharmonicPlate && params.plate_stiffness > 0.0 {
            accel -= params.plate_stiffness * self.eval_biharmonic_at(x, y, idx, params.boundary);
        }
        if driver.enabled {
            accel += masks::driver_force_at(self.width, self.height, x, y, driver, osc);
        }
        let gamma = (params.damping.max(0.0) + self.sponge_mask[idx].max(0.0)) * 0.5 * dt;
        let curr = self.u_curr[idx];
        if self.stationary_start {
            return curr + 0.5 * dt_sq * accel / (1.0 + gamma);
        }
        let prev = self.u_prev[idx];
        (2.0 * curr - (1.0 - gamma) * prev + dt_sq * accel) / (1.0 + gamma)
    }

    #[verified_engine::verified]
    fn accumulate_detector_column(&mut self, det_x: usize) {
        let col = det_x.min(self.width - 1);
        for y in 0..self.height {
            let val = self.u_curr[flatten_2d_index(col, y, self.width)];
            self.detector_accum[y] += val * val;
        }
        self.detector_samples += 1;
    }

    /// Writes the time-averaged intensity profile $I(y)$ along the detector screen into `out`.
    #[verified_engine::verified]
    pub fn read_detector_profile(&self, det_x: usize, out: &mut [f64]) {
        assert!(
            out.len() >= self.height,
            "Output buffer shorter than grid height"
        );
        if self.detector_samples > 0 {
            let inv = 1.0 / (self.detector_samples as f64);
            for (dst, &acc) in out.iter_mut().zip(self.detector_accum.iter()) {
                *dst = acc * inv;
            }
        } else {
            let col = det_x.min(self.width - 1);
            for (y, dst) in out.iter_mut().enumerate().take(self.height) {
                *dst = self.energy_ema[flatten_2d_index(col, y, self.width)];
            }
        }
    }

    /// Injects a smooth 2D Gaussian displacement pulse centered at normalized `(cx_norm, cy_norm)`.
    #[verified_engine::verified]
    pub fn inject_gaussian_pulse(
        &mut self,
        cx_norm: f64,
        cy_norm: f64,
        sigma_norm: f64,
        amplitude: f64,
    ) {
        masks::inject_gaussian_pulse_into(
            &mut self.u_curr,
            &mut self.u_prev,
            &self.wall_mask,
            self.width,
            self.height,
            cx_norm,
            cy_norm,
            sigma_norm,
            amplitude,
        );
    }
}
