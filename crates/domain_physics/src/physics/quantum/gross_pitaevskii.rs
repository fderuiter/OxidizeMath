//! Zero-allocation 2D Strang split-step Fourier solver for Gross-Pitaevskii superfluid
//! Bose-Einstein condensates and quantized $2\pi$ topological vortices.
//!
//! Pre-plans 1D row and column `rustfft` passes and pre-allocates complex wavefunctions,
//! transpose matrices, scratch buffers, and plaquette vortex storage at initialization so
//! [`GrossPitaevskiiSolver2D::step`] and [`GrossPitaevskiiSolver2D::detect_vortices`] execute
//! with zero dynamic heap allocations under `VerifiedAllocator::lock_allocations()`.
//!
//! # Example
//!
//! ```
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::quantum::gross_pitaevskii::{
//!     GpeParams2D, GpeTrapConfig, GrossPitaevskiiSolver2D,
//! };
//!
//! let mut solver = GrossPitaevskiiSolver2D::new(64, 64, 16.0, 16.0);
//! solver.configure_trap(GpeTrapConfig::default());
//! solver.seed_condensate(200.0, 1.0);
//! solver.imprint_vortex_dipole(-1.2, 0.0, 1.2, 0.0, 0.35);
//! let summary0 = solver.detect_vortices(0.02);
//! assert_eq!(summary0.positive_count, 1);
//! assert_eq!(summary0.negative_count, 1);
//! assert_eq!(summary0.net_charge, 0);
//!
//! let inv0 = solver.compute_invariants(200.0);
//! solver.step(&GpeParams2D { interaction_g: 200.0, imaginary_time_cooling: false }, 0.005);
//! let inv1 = solver.compute_invariants(200.0);
//! assert!((inv1.norm - inv0.norm).abs() < 1.0e-12);
//! ```

// [cite:quantum_mechanics]

use std::sync::Arc;

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Configuration structs, trapping potentials, conserved invariants, and vortex core types.
pub mod config;
/// Zero-allocation $2\times 2$ plaquette vortex detection and Parseval invariant computation.
pub mod vortices;

#[cfg(test)]
mod tests;

pub use config::{
    GpeInvariants2D, GpeParams2D, GpeTrapConfig, GpeTrapKind, MAX_GPE_DIM, MAX_TRACKED_VORTICES,
    VortexChargeSummary, VortexCore2D, thomas_fermi_chemical_potential_2d,
};
pub use vortices::wrap_phase_difference;

/// Pre-planned 2D Strang split-step Fourier solver for Gross-Pitaevskii condensates.
#[derive(Clone)]
pub struct GrossPitaevskiiSolver2D {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) length_x: f64,
    pub(crate) length_y: f64,
    pub(crate) dx: f64,
    pub(crate) dy: f64,
    pub(crate) time: f64,
    pub(crate) target_norm: f64,
    pub(crate) trap: GpeTrapConfig,
    pub(crate) x_coords: Vec<f64>,
    pub(crate) y_coords: Vec<f64>,
    pub(crate) kx_deriv: Vec<f64>,
    pub(crate) ky_deriv: Vec<f64>,
    pub(crate) kx_sq: Vec<f64>,
    pub(crate) ky_sq: Vec<f64>,
    pub(crate) potential: Vec<f64>,
    pub(crate) psi: Vec<Complex<f64>>,
    pub(crate) kspace: Vec<Complex<f64>>,
    pub(crate) transposed: Vec<Complex<f64>>,
    pub(crate) fft_scratch: Vec<Complex<f64>>,
    pub(crate) density_buf: Vec<f64>,
    pub(crate) phase_buf: Vec<f64>,
    pub(crate) detected_vortices: Vec<VortexCore2D>,
    pub(crate) fft_row_fwd: Arc<dyn Fft<f64>>,
    pub(crate) fft_col_fwd: Arc<dyn Fft<f64>>,
    pub(crate) fft_row_inv: Arc<dyn Fft<f64>>,
    pub(crate) fft_col_inv: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for GrossPitaevskiiSolver2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrossPitaevskiiSolver2D")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("length_x", &self.length_x)
            .field("length_y", &self.length_y)
            .field("time", &self.time)
            .finish()
    }
}

impl GrossPitaevskiiSolver2D {
    /// Creates a pre-planned 2D Gross-Pitaevskii solver on $[-L_x/2, L_x/2) \times [-L_y/2, L_y/2)$
    /// with grid dimensions clamped into even `[8, MAX_GPE_DIM]`.
    #[must_use]
    pub fn new(width: usize, height: usize, length_x: f64, length_y: f64) -> Self {
        let (w, h) = (
            config::clamp_even_dim(width),
            config::clamp_even_dim(height),
        );
        let lx = config::sanitize_domain_len(length_x);
        let ly = config::sanitize_domain_len(length_y);
        let (dx, dy) = (lx / (w as f64), ly / (h as f64));
        let x_coords: Vec<f64> = (0..w).map(|i| -0.5 * lx + (i as f64) * dx).collect();
        let y_coords: Vec<f64> = (0..h).map(|j| -0.5 * ly + (j as f64) * dy).collect();
        let (kx_deriv, kx_sq) = config::build_1d_wavenumbers(w, lx);
        let (ky_deriv, ky_sq) = config::build_1d_wavenumbers(h, ly);
        let (fft_row_fwd, fft_col_fwd, fft_row_inv, fft_col_inv, scratch_len) = plan_2d_ffts(w, h);
        let total = w * h;
        let zero_c = Complex::new(0.0, 0.0);
        let trap = GpeTrapConfig::default();
        let mut solver = Self {
            width: w,
            height: h,
            length_x: lx,
            length_y: ly,
            dx,
            dy,
            time: 0.0,
            target_norm: 1.0,
            trap,
            x_coords,
            y_coords,
            kx_deriv,
            ky_deriv,
            kx_sq,
            ky_sq,
            potential: vec![0.0; total],
            psi: vec![zero_c; total],
            kspace: vec![zero_c; total],
            transposed: vec![zero_c; total],
            fft_scratch: vec![zero_c; scratch_len],
            density_buf: vec![0.0; total],
            phase_buf: vec![0.0; total],
            detected_vortices: Vec::with_capacity(MAX_TRACKED_VORTICES),
            fft_row_fwd,
            fft_col_fwd,
            fft_row_inv,
            fft_col_inv,
        };
        solver.configure_trap(trap);
        solver
    }

    /// Grid width $W$.
    #[must_use]
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height $H$.
    #[must_use]
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Horizontal grid spacing $\Delta x$.
    #[must_use]
    #[verified_engine::verified]
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// Vertical grid spacing $\Delta y$.
    #[must_use]
    #[verified_engine::verified]
    pub fn dy(&self) -> f64 {
        self.dy
    }

    /// Horizontal domain length $L_x$.
    #[must_use]
    #[verified_engine::verified]
    pub fn length_x(&self) -> f64 {
        self.length_x
    }

    /// Vertical domain length $L_y$.
    #[must_use]
    #[verified_engine::verified]
    pub fn length_y(&self) -> f64 {
        self.length_y
    }

    /// Elapsed real or imaginary time.
    #[must_use]
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Spatial $x$ coordinates across columns.
    #[must_use]
    #[verified_engine::verified]
    pub fn x_coords(&self) -> &[f64] {
        &self.x_coords
    }

    /// Spatial $y$ coordinates across rows.
    #[must_use]
    #[verified_engine::verified]
    pub fn y_coords(&self) -> &[f64] {
        &self.y_coords
    }

    /// Complex condensate order parameter $\psi(x, y)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn wavefunction(&self) -> &[Complex<f64>] {
        &self.psi
    }

    /// Cached condensate density $|\psi(x, y)|^2$ updated by [`Self::refresh_density_and_phase`] or [`Self::detect_vortices`].
    #[must_use]
    #[verified_engine::verified]
    pub fn density(&self) -> &[f64] {
        &self.density_buf
    }

    /// Cached condensate phase $\arg\psi(x, y) \in [-\pi, \pi]$ updated by [`Self::refresh_density_and_phase`] or [`Self::detect_vortices`].
    #[must_use]
    #[verified_engine::verified]
    pub fn phase(&self) -> &[f64] {
        &self.phase_buf
    }

    /// External trap potential values $V(x, y)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn potential(&self) -> &[f64] {
        &self.potential
    }

    /// Slice of quantized vortex cores detected during the most recent [`Self::detect_vortices`] pass.
    #[must_use]
    #[verified_engine::verified]
    pub fn vortices(&self) -> &[VortexCore2D] {
        &self.detected_vortices
    }

    /// Updates the external trapping potential $V(x, y)$ in-place without heap allocation.
    #[verified_engine::verified]
    pub fn configure_trap(&mut self, trap: GpeTrapConfig) {
        self.trap = trap;
        let w = self.width;
        for y in 0..self.height {
            let yc = self.y_coords[y];
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                self.potential[idx] = trap.evaluate(self.x_coords[x], yc);
            }
        }
    }

    /// Seeds a smooth regularized Thomas-Fermi or Gaussian ground-state condensate normalized to `target_norm`.
    #[verified_engine::verified]
    pub fn seed_condensate(&mut self, interaction_g: f64, target_norm: f64) {
        self.time = 0.0;
        self.target_norm = target_norm.max(1.0e-6);
        let w = self.width;
        let h = self.height;
        let r_cut_x = 0.42 * self.length_x;
        let r_cut_y = 0.42 * self.length_y;
        let mu_tf = thomas_fermi_chemical_potential_2d(
            interaction_g,
            self.target_norm,
            self.trap.omega_x,
            self.trap.omega_y,
        );

        for y in 0..h {
            let yc = self.y_coords[y];
            for x in 0..w {
                let xc = self.x_coords[x];
                let idx = flatten_2d_index(x, y, w);
                let amp = if self.trap.kind == GpeTrapKind::BoxPeriodic {
                    1.0
                } else {
                    let v = self.potential[idx];
                    let diff = mu_tf - v;
                    let smooth_pos = 0.5 * ((diff * diff + 0.36).sqrt() + diff);
                    let window = (-((xc / r_cut_x).powi(4) + (yc / r_cut_y).powi(4))).exp();
                    smooth_pos.sqrt() * window
                };
                self.psi[idx] = Complex::new(amp, 0.0);
            }
        }
        self.renormalize_in_place(self.target_norm);
        let _ = self.refresh_density_and_phase();
    }

    /// Imprints a quantized topological vortex of integer charge `charge` (`+1` or `-1`) at `(x0, y0)`
    /// with Padé core healing length `healing_len` and preserves total norm with zero heap allocations.
    #[verified_engine::verified]
    pub fn imprint_vortex(&mut self, x0: f64, y0: f64, charge: i32, healing_len: f64) {
        if charge == 0 {
            return;
        }
        let xi = healing_len.max(0.5 * self.dx.min(self.dy));
        let xi_sq2 = 2.0 * xi * xi;
        let abs_q = charge.unsigned_abs() as i32;
        // Offset infinitesimally off exact grid intersections so atan2 is regular on all plaquette vertices.
        let vx = x0 + 1.1e-4 * self.dx;
        let vy = y0 + 1.3e-4 * self.dy;
        let w = self.width;

        for y in 0..self.height {
            let dy = self.y_coords[y] - vy;
            for x in 0..w {
                let dx = self.x_coords[x] - vx;
                let r_sq = dx * dx + dy * dy;
                let core = (r_sq / (r_sq + xi_sq2)).sqrt().powi(abs_q);
                let angle = (charge as f64) * dy.atan2(dx);
                let factor = Complex::new(core * angle.cos(), core * angle.sin());
                let idx = flatten_2d_index(x, y, w);
                self.psi[idx] *= factor;
            }
        }
        self.renormalize_in_place(self.target_norm);
    }

    /// Imprints a counter-rotating vortex-antivortex dipole (`+1` at `(x_pos, y_pos)` and `-1` at `(x_neg, y_neg)`).
    #[verified_engine::verified]
    pub fn imprint_vortex_dipole(
        &mut self,
        x_pos: f64,
        y_pos: f64,
        x_neg: f64,
        y_neg: f64,
        healing_len: f64,
    ) {
        self.imprint_vortex(x_pos, y_pos, 1, healing_len);
        self.imprint_vortex(x_neg, y_neg, -1, healing_len);
    }

    /// Advances the 2D Gross-Pitaevskii equation by `dt` via Strang split-step Fourier integration
    /// with zero dynamic heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, params: &GpeParams2D, dt: f64) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let half_dt = 0.5 * dt;
        let g = params.interaction_g;
        let cooling = params.imaginary_time_cooling;

        self.apply_potential_nonlinear_half_step(half_dt, g, cooling);
        self.apply_kinetic_full_step(dt, cooling);
        self.apply_potential_nonlinear_half_step(half_dt, g, cooling);

        if cooling {
            self.renormalize_in_place(self.target_norm);
        }
        self.time += dt;
    }

    fn apply_potential_nonlinear_half_step(&mut self, half_dt: f64, g: f64, cooling: bool) {
        if cooling {
            for i in 0..self.psi.len() {
                let v_eff = self.potential[i] + g * self.psi[i].norm_sqr();
                let damp = (-half_dt * v_eff).clamp(-40.0, 40.0).exp();
                self.psi[i] *= damp;
            }
        } else {
            for i in 0..self.psi.len() {
                let theta = -half_dt * (self.potential[i] + g * self.psi[i].norm_sqr());
                self.psi[i] *= Complex::new(theta.cos(), theta.sin());
            }
        }
    }

    fn apply_kinetic_full_step(&mut self, dt: f64, cooling: bool) {
        let w = self.width;
        let h = self.height;
        let inv_wh = 1.0 / ((w * h) as f64);

        self.kspace.copy_from_slice(&self.psi);
        self.fft_row_fwd
            .process_with_scratch(&mut self.kspace, &mut self.fft_scratch);
        self.transpose_kspace_to_cols();
        self.fft_col_fwd
            .process_with_scratch(&mut self.transposed, &mut self.fft_scratch);

        if cooling {
            for x in 0..w {
                let kx2 = self.kx_sq[x];
                for y in 0..h {
                    let idx = flatten_2d_index(y, x, h);
                    let k2_half = 0.5 * (kx2 + self.ky_sq[y]);
                    self.transposed[idx] *= (-dt * k2_half).exp() * inv_wh;
                }
            }
        } else {
            for x in 0..w {
                let kx2 = self.kx_sq[x];
                for y in 0..h {
                    let idx = flatten_2d_index(y, x, h);
                    let theta = -0.5 * (kx2 + self.ky_sq[y]) * dt;
                    self.transposed[idx] *= Complex::new(theta.cos(), theta.sin()) * inv_wh;
                }
            }
        }

        self.fft_col_inv
            .process_with_scratch(&mut self.transposed, &mut self.fft_scratch);
        self.transpose_cols_to_psi();
        self.fft_row_inv
            .process_with_scratch(&mut self.psi, &mut self.fft_scratch);
    }

    pub(crate) fn transpose_kspace_to_cols(&mut self) {
        let w = self.width;
        let h = self.height;
        for y in 0..h {
            for x in 0..w {
                let src = flatten_2d_index(x, y, w);
                let dst = flatten_2d_index(y, x, h);
                self.transposed[dst] = self.kspace[src];
            }
        }
    }

    fn transpose_cols_to_psi(&mut self) {
        let w = self.width;
        let h = self.height;
        for x in 0..w {
            for y in 0..h {
                let src = flatten_2d_index(y, x, h);
                let dst = flatten_2d_index(x, y, w);
                self.psi[dst] = self.transposed[src];
            }
        }
    }

    fn renormalize_in_place(&mut self, target_norm: f64) {
        let mut sum_rho = 0.0_f64;
        for z in &self.psi {
            sum_rho += z.norm_sqr();
        }
        let current_norm = sum_rho * self.dx * self.dy;
        if current_norm > 1.0e-20 && target_norm > 0.0 {
            let scale = (target_norm / current_norm).sqrt();
            for z in &mut self.psi {
                *z *= scale;
            }
        }
    }
}

type Planned2DFfts = (
    Arc<dyn Fft<f64>>,
    Arc<dyn Fft<f64>>,
    Arc<dyn Fft<f64>>,
    Arc<dyn Fft<f64>>,
    usize,
);

fn plan_2d_ffts(w: usize, h: usize) -> Planned2DFfts {
    let mut planner = FftPlanner::new();
    let row_fwd = planner.plan_fft_forward(w);
    let col_fwd = planner.plan_fft_forward(h);
    let row_inv = planner.plan_fft_inverse(w);
    let col_inv = planner.plan_fft_inverse(h);
    let scratch_len = row_fwd
        .get_inplace_scratch_len()
        .max(col_fwd.get_inplace_scratch_len())
        .max(row_inv.get_inplace_scratch_len())
        .max(col_inv.get_inplace_scratch_len())
        .max(1);
    (row_fwd, col_fwd, row_inv, col_inv, scratch_len)
}
