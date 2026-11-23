//! Zero-allocation 1D periodic pseudospectral solver for Korteweg-de Vries (KdV) solitons,
//! linear Airy dispersion, and 1D cubic Nonlinear Schrödinger (NLS) wave envelopes.
//!
//! Pre-plans forward and inverse `rustfft` passes and pre-allocates spectral Integrating-Factor
//! RK4 (Lawson IFRK4), Orszag $2/3$-dealiasing, and Strang split-step work slices at construction
//! so every time step executes with zero dynamic heap allocations under
//! `VerifiedAllocator::lock_allocations()`.
//!
//! # Example
//!
//! ```
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::analysis::pde::kdv::{
//!     KdvEquationMode, KdvParams, KdvSolver1D, SolitonSpec, kdv_two_soliton_phase_shifts,
//! };
//!
//! let mut solver = KdvSolver1D::new(256, 40.0);
//! solver.seed_kdv_solitons(&[
//!     SolitonSpec { speed_c: 16.0, x0: -10.0 },
//!     SolitonSpec { speed_c: 4.0, x0: -2.0 },
//! ]);
//! let params = KdvParams {
//!     mode: KdvEquationMode::KdvSoliton,
//!     nonlinear_enabled: true,
//!     nls_coupling: 1.0,
//! };
//! let inv0 = solver.compute_invariants(&params);
//! solver.step(&params, 0.02);
//! let inv1 = solver.compute_invariants(&params);
//! assert!(inv1.relative_drift(inv0)[0] < 1.0e-6);
//! let shifts = kdv_two_soliton_phase_shifts(16.0, 4.0);
//! assert!(shifts.delta_fast > 0.5 && shifts.delta_slow < -1.0);
//! ```

// [cite:advanced_linear_algebra]

use std::f64::consts::TAU;
use std::sync::Arc;

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Configuration structs, conserved invariant types, and exact analytical soliton formulas.
pub mod config;
/// Zero-allocation invariant computation and sub-cell parabolic soliton peak tracking.
pub mod diagnostics;

#[cfg(test)]
mod tests;

pub use config::{
    DetectedPulsePeak, KDV_IFRK4_CFL_TARGET, KdvEquationMode, KdvParams, MAX_KDV_POINTS,
    MAX_KDV_SUBSTEPS, SolitonInvariants, SolitonSpec, TwoSolitonPhaseShift,
    kdv_single_soliton_profile, kdv_two_soliton_phase_shifts, nls_bright_soliton_sample,
    wrap_periodic_coord,
};

/// Pre-planned 1D periodic pseudospectral solver for KdV solitons, linear Airy waves, and cubic NLS.
#[derive(Clone)]
pub struct KdvSolver1D {
    pub(crate) num_points: usize,
    pub(crate) length: f64,
    pub(crate) dx: f64,
    pub(crate) time: f64,
    pub(crate) x_coords: Vec<f64>,
    pub(crate) u: Vec<f64>,
    pub(crate) u_initial: Vec<f64>,
    pub(crate) psi: Vec<Complex<f64>>,
    pub(crate) k_deriv: Vec<f64>,
    pub(crate) k_sq: Vec<f64>,
    pub(crate) dealias_mask: Vec<bool>,
    pub(crate) fft_fwd: Arc<dyn Fft<f64>>,
    pub(crate) fft_inv: Arc<dyn Fft<f64>>,
    pub(crate) fft_scratch: Vec<Complex<f64>>,
    u_hat: Vec<Complex<f64>>,
    stage_a: Vec<Complex<f64>>,
    stage_b: Vec<Complex<f64>>,
    stage_c: Vec<Complex<f64>>,
    stage_d: Vec<Complex<f64>>,
    stage_state: Vec<Complex<f64>>,
    pub(crate) work_spec: Vec<Complex<f64>>,
    half_disp: Vec<Complex<f64>>,
    full_disp: Vec<Complex<f64>>,
    cached_ifrk4_h: f64,
}

impl std::fmt::Debug for KdvSolver1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KdvSolver1D")
            .field("num_points", &self.num_points)
            .field("length", &self.length)
            .field("dx", &self.dx)
            .field("time", &self.time)
            .finish()
    }
}

impl KdvSolver1D {
    /// Constructs a pre-planned periodic pseudospectral 1D solver on $[-L/2, L/2)$
    /// with `num_points` clamped into even `[8, MAX_KDV_POINTS]`.
    #[must_use]
    pub fn new(num_points: usize, length: f64) -> Self {
        let mut n = num_points.clamp(8, MAX_KDV_POINTS);
        if !n.is_multiple_of(2) {
            n -= 1;
        }
        let domain_len = if length.is_finite() && length > 0.5 {
            length
        } else {
            40.0
        };
        let dx = domain_len / (n as f64);
        let x_coords: Vec<f64> = (0..n)
            .map(|j| -0.5 * domain_len + (j as f64) * dx)
            .collect();
        let (k_deriv, k_sq, dealias_mask) = config::build_spectral_tables(n, domain_len);
        let mut planner = FftPlanner::new();
        let fft_fwd = planner.plan_fft_forward(n);
        let fft_inv = planner.plan_fft_inverse(n);
        let scratch_len = fft_fwd
            .get_inplace_scratch_len()
            .max(fft_inv.get_inplace_scratch_len())
            .max(1);
        let zero_c = Complex::new(0.0, 0.0);
        Self {
            num_points: n,
            length: domain_len,
            dx,
            time: 0.0,
            x_coords,
            u: vec![0.0; n],
            u_initial: vec![0.0; n],
            psi: vec![zero_c; n],
            k_deriv,
            k_sq,
            dealias_mask,
            fft_fwd,
            fft_inv,
            fft_scratch: vec![zero_c; scratch_len],
            u_hat: vec![zero_c; n],
            stage_a: vec![zero_c; n],
            stage_b: vec![zero_c; n],
            stage_c: vec![zero_c; n],
            stage_d: vec![zero_c; n],
            stage_state: vec![zero_c; n],
            work_spec: vec![zero_c; n],
            half_disp: vec![Complex::new(1.0, 0.0); n],
            full_disp: vec![Complex::new(1.0, 0.0); n],
            cached_ifrk4_h: f64::NAN,
        }
    }

    /// Number of periodic grid nodes $N$.
    #[must_use]
    #[verified_engine::verified]
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// Periodic domain length $L$.
    #[must_use]
    #[verified_engine::verified]
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Spatial grid spacing $\Delta x = L / N$.
    #[must_use]
    #[verified_engine::verified]
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// Elapsed simulation time $t$.
    #[must_use]
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Periodic spatial grid coordinates $x_j \in [-L/2, L/2)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn x_coords(&self) -> &[f64] {
        &self.x_coords
    }

    /// Current real scalar wave field $u(x, t)$ (or $|\psi(x, t)|$ envelope in NLS mode).
    #[must_use]
    #[verified_engine::verified]
    pub fn field(&self) -> &[f64] {
        &self.u
    }

    /// Initial scalar profile $u(x, 0)$ stored at seeding time.
    #[must_use]
    #[verified_engine::verified]
    pub fn initial_field(&self) -> &[f64] {
        &self.u_initial
    }

    /// Complex wave envelope $\psi(x, t)$ evolved in [`KdvEquationMode::CubicNls1D`].
    #[must_use]
    #[verified_engine::verified]
    pub fn psi_field(&self) -> &[Complex<f64>] {
        &self.psi
    }

    /// Seeds a superposition of KdV $\operatorname{sech}^2$ solitons and resets $t = 0$.
    #[verified_engine::verified]
    pub fn seed_kdv_solitons(&mut self, solitons: &[SolitonSpec]) {
        self.time = 0.0;
        for i in 0..self.num_points {
            let x = self.x_coords[i];
            let mut sum = 0.0_f64;
            for s in solitons {
                sum += kdv_single_soliton_profile(x, s.x0, s.speed_c, self.length);
            }
            self.u[i] = sum;
            self.u_initial[i] = sum;
            self.psi[i] = Complex::new(sum, 0.0);
        }
    }

    /// Seeds a Zabusky-Kruskal smooth cosine initial wave $u(x, 0) = A \cos(2\pi m x / L)$ and resets $t = 0$.
    #[verified_engine::verified]
    pub fn seed_cosine_wave(&mut self, amplitude: f64, harmonic_m: usize) {
        self.time = 0.0;
        let k0 = TAU * (harmonic_m.max(1) as f64) / self.length;
        for i in 0..self.num_points {
            let val = amplitude * (k0 * self.x_coords[i]).cos();
            self.u[i] = val;
            self.u_initial[i] = val;
            self.psi[i] = Complex::new(val, 0.0);
        }
    }

    /// Seeds a superposition of 1D cubic NLS bright envelope solitons $(\eta_k, v_k, x_{0,k})$ and resets $t = 0$.
    #[verified_engine::verified]
    pub fn seed_nls_bright_solitons(&mut self, solitons: &[(f64, f64, f64)]) {
        self.time = 0.0;
        for i in 0..self.num_points {
            let x = self.x_coords[i];
            let mut z = Complex::new(0.0, 0.0);
            for &(eta, carrier_k, x0) in solitons {
                z += nls_bright_soliton_sample(x, x0, eta, carrier_k, self.length);
            }
            self.psi[i] = z;
            let mag = z.norm();
            self.u[i] = mag;
            self.u_initial[i] = mag;
        }
    }

    /// Injects an additive localized $\operatorname{sech}^2$ soliton pulse centered at `x0` without heap allocation.
    #[verified_engine::verified]
    pub fn inject_soliton_pulse(&mut self, x0: f64, speed_c: f64, mode: KdvEquationMode) {
        for i in 0..self.num_points {
            let x = self.x_coords[i];
            match mode {
                KdvEquationMode::KdvSoliton => {
                    self.u[i] += kdv_single_soliton_profile(x, x0, speed_c, self.length);
                }
                KdvEquationMode::CubicNls1D => {
                    let eta = (0.5 * speed_c).sqrt().max(0.4);
                    self.psi[i] += nls_bright_soliton_sample(x, x0, eta, 0.0, self.length);
                    self.u[i] = self.psi[i].norm();
                }
            }
        }
    }

    /// Advances the active 1D dispersive PDE by `dt` with zero dynamic heap allocations,
    /// returning the number of internal substeps executed.
    #[verified_engine::verified]
    pub fn step(&mut self, params: &KdvParams, dt: f64) -> usize {
        if !dt.is_finite() || dt <= 0.0 {
            return 0;
        }
        match params.mode {
            KdvEquationMode::KdvSoliton => self.step_kdv(dt, params.nonlinear_enabled),
            KdvEquationMode::CubicNls1D => {
                let sigma = if params.nonlinear_enabled {
                    params.nls_coupling
                } else {
                    0.0
                };
                self.step_nls(dt, sigma)
            }
        }
    }

    /// Advances $u_t + 6 \alpha u u_x + u_{xxx} = 0$ by `dt` via dealiased Lawson IFRK4 in Fourier space.
    #[verified_engine::verified]
    pub fn step_kdv(&mut self, dt: f64, nonlinear_enabled: bool) -> usize {
        let n = self.num_points;
        let inv_n = 1.0 / (n as f64);
        for i in 0..n {
            self.u_hat[i] = Complex::new(self.u[i], 0.0);
        }
        self.fft_fwd
            .process_with_scratch(&mut self.u_hat, &mut self.fft_scratch);

        if !nonlinear_enabled {
            for i in 0..n {
                let k = self.k_deriv[i];
                let theta = k * k * k * dt;
                self.u_hat[i] *= Complex::new(theta.cos(), theta.sin());
            }
            self.fft_inv
                .process_with_scratch(&mut self.u_hat, &mut self.fft_scratch);
            for i in 0..n {
                self.u[i] = self.u_hat[i].re * inv_n;
            }
            self.time += dt;
            return 1;
        }

        let substeps = self.safe_kdv_substeps(dt);
        let h = dt / (substeps as f64);
        self.update_ifrk4_factors(h);
        self.apply_dealias_in_place();

        for _ in 0..substeps {
            self.advance_ifrk4_substep(h);
        }

        self.work_spec.copy_from_slice(&self.u_hat);
        self.fft_inv
            .process_with_scratch(&mut self.work_spec, &mut self.fft_scratch);
        for i in 0..n {
            self.u[i] = self.work_spec[i].re * inv_n;
        }
        self.time += dt;
        substeps
    }

    /// Advances $i \psi_t + \psi_{xx} + 2 \sigma |\psi|^2 \psi = 0$ by `dt` via unitary Strang splitting.
    #[verified_engine::verified]
    pub fn step_nls(&mut self, dt: f64, coupling_sigma: f64) -> usize {
        let max_rho = self.psi.iter().fold(0.0_f64, |m, z| m.max(z.norm_sqr()));
        let nl_rate = (2.0 * coupling_sigma.abs() * max_rho).max(1.0);
        let substeps = ((dt * nl_rate / 0.25).ceil() as usize).clamp(1, MAX_KDV_SUBSTEPS);
        let h = dt / (substeps as f64);
        let half_nl = coupling_sigma * h;
        let inv_n = 1.0 / (self.num_points as f64);

        for _ in 0..substeps {
            if half_nl != 0.0 {
                apply_nls_nonlinear_kick(&mut self.psi, half_nl);
            }
            self.fft_fwd
                .process_with_scratch(&mut self.psi, &mut self.fft_scratch);
            for i in 0..self.num_points {
                let theta = -self.k_sq[i] * h;
                self.psi[i] *= Complex::new(theta.cos(), theta.sin()) * inv_n;
            }
            self.fft_inv
                .process_with_scratch(&mut self.psi, &mut self.fft_scratch);
            if half_nl != 0.0 {
                apply_nls_nonlinear_kick(&mut self.psi, half_nl);
            }
        }
        for i in 0..self.num_points {
            self.u[i] = self.psi[i].norm();
        }
        self.time += dt;
        substeps
    }

    fn safe_kdv_substeps(&self, dt: f64) -> usize {
        let u_max = self.u.iter().fold(
            1.0e-3_f64,
            |m, v| if v.is_finite() { m.max(v.abs()) } else { m },
        );
        let k_dealias = TAU * ((self.num_points / 3) as f64) / self.length;
        let lambda = 6.0 * u_max * k_dealias;
        let cfl_steps = (dt * lambda / KDV_IFRK4_CFL_TARGET).ceil() as usize;
        let acc_steps = (dt / 2.5e-4).ceil() as usize;
        cfl_steps.max(acc_steps).clamp(1, MAX_KDV_SUBSTEPS)
    }

    fn update_ifrk4_factors(&mut self, h: f64) {
        if (self.cached_ifrk4_h - h).abs() <= 1.0e-15 {
            return;
        }
        for i in 0..self.num_points {
            let k = self.k_deriv[i];
            let k3_half = 0.5 * k * k * k * h;
            let e_half = Complex::new(k3_half.cos(), k3_half.sin());
            self.half_disp[i] = e_half;
            self.full_disp[i] = e_half * e_half;
        }
        self.cached_ifrk4_h = h;
    }

    fn apply_dealias_in_place(&mut self) {
        for i in 0..self.num_points {
            if !self.dealias_mask[i] {
                self.u_hat[i] = Complex::new(0.0, 0.0);
            }
        }
    }

    fn advance_ifrk4_substep(&mut self, h: f64) {
        let n = self.num_points;
        self.stage_state.copy_from_slice(&self.u_hat);
        self.eval_kdv_nonlinear_rhs(h, StageSlot::A);

        for i in 0..n {
            self.stage_state[i] = self.half_disp[i] * (self.u_hat[i] + 0.5 * self.stage_a[i]);
        }
        self.eval_kdv_nonlinear_rhs(h, StageSlot::B);

        for i in 0..n {
            self.stage_state[i] = self.half_disp[i] * self.u_hat[i] + 0.5 * self.stage_b[i];
        }
        self.eval_kdv_nonlinear_rhs(h, StageSlot::C);

        for i in 0..n {
            self.stage_state[i] =
                self.full_disp[i] * self.u_hat[i] + self.half_disp[i] * self.stage_c[i];
        }
        self.eval_kdv_nonlinear_rhs(h, StageSlot::D);

        let inv6 = 1.0 / 6.0;
        for i in 0..n {
            if self.dealias_mask[i] {
                let weighted = self.full_disp[i] * self.stage_a[i]
                    + 2.0 * self.half_disp[i] * (self.stage_b[i] + self.stage_c[i])
                    + self.stage_d[i];
                self.u_hat[i] = self.full_disp[i] * self.u_hat[i] + inv6 * weighted;
            } else {
                self.u_hat[i] = Complex::new(0.0, 0.0);
            }
        }
    }

    fn eval_kdv_nonlinear_rhs(&mut self, h: f64, slot: StageSlot) {
        let n = self.num_points;
        let inv_n = 1.0 / (n as f64);
        for i in 0..n {
            self.work_spec[i] = if self.dealias_mask[i] {
                self.stage_state[i]
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        self.fft_inv
            .process_with_scratch(&mut self.work_spec, &mut self.fft_scratch);
        for z in &mut self.work_spec {
            let u_phys = z.re * inv_n;
            *z = Complex::new(u_phys * u_phys, 0.0);
        }
        self.fft_fwd
            .process_with_scratch(&mut self.work_spec, &mut self.fft_scratch);

        let out = match slot {
            StageSlot::A => &mut self.stage_a,
            StageSlot::B => &mut self.stage_b,
            StageSlot::C => &mut self.stage_c,
            StageSlot::D => &mut self.stage_d,
        };
        for (i, dst) in out.iter_mut().enumerate().take(n) {
            if self.dealias_mask[i] {
                let factor = Complex::new(0.0, -3.0 * h * self.k_deriv[i]);
                *dst = factor * self.work_spec[i];
            } else {
                *dst = Complex::new(0.0, 0.0);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum StageSlot {
    A,
    B,
    C,
    D,
}

fn apply_nls_nonlinear_kick(psi: &mut [Complex<f64>], half_nl: f64) {
    for z in psi {
        let theta = half_nl * z.norm_sqr();
        *z *= Complex::new(theta.cos(), theta.sin());
    }
}
