//! Zero-allocation Gerchberg-Saxton iterative phase-holography solver.
//!
//! Synthesizes unit-modulus spatial light modulator (SLM) phase masks $e^{i\phi(x, y)}$ seeded
//! deterministically via [`oxidize_core::rng::OxidizeRng`], eliminating conjugate twin-image
//! symmetry while reporting normalized root-mean-square amplitude discrepancy, diffraction
//! efficiency, and conjugate twin-image leakage.

// [cite:graph_parameters_rust]

use std::f64::consts::PI;

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

use super::diffraction::AngularSpectrumPropagator;

/// Optical propagation regime coupling the SLM hologram plane to the target replay plane.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HologramRegime {
    /// Far-field Fraunhofer / Fourier-lens focal replay plane via centered unitary 2D FFT.
    FourierLens,
    /// Near-field lensless free-space replay plane at axial distance `distance_z` via Helmholtz ASM.
    FresnelAngularSpectrum {
        /// Axial distance $z > 0$ from SLM plane to replay plane.
        distance_z: f64,
    },
}

/// Asymmetric target amplitude patterns used to benchmark twin-image suppression in phase holography.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum HologramTargetPreset {
    /// Off-axis upper-left ring paired with an off-axis upper-right focal spot.
    #[default]
    OffAxisRingAndSpot,
    /// Single-quadrant asymmetric L-shaped reticle.
    AsymmetricReticle,
    /// Three off-axis focal spots arranged in the upper half-plane.
    UpperTriadSpots,
    /// Asymmetric letter `F` confined to the upper-left quadrant.
    OffAxisGlyphF,
}

/// Quantitative convergence and twin-suppression telemetry for a synthesized phase hologram.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HologramMetrics {
    /// Completed Gerchberg-Saxton iteration count.
    pub iteration: usize,
    /// Scale-normalized root-mean-square amplitude discrepancy against the target (`< 0.32` beats the `0.707` real-scalar limit).
    pub target_nrmse: f64,
    /// Fraction $\eta \in [0, 1]$ of total replay optical power concentrated inside the target support.
    pub diffraction_efficiency: f64,
    /// Fraction $\eta_{\text{twin}} \in [0, 1]$ of total replay optical power leaking into the centrosymmetric twin region.
    pub twin_leakage: f64,
}

/// Fills `out_target` with an asymmetric non-negative amplitude target on a `width x height` grid.
#[verified_engine::verified]
pub fn fill_target_amplitude_into(
    preset: HologramTargetPreset,
    width: usize,
    height: usize,
    out_target: &mut [f64],
) {
    assert_eq!(
        out_target.len(),
        width * height,
        "Target amplitude slice mismatch"
    );
    out_target.fill(0.0);
    let w = width as f64;
    let h = height as f64;

    for y in 0..height {
        let ny = ((y as f64) - 0.5 * h) / h;
        for x in 0..width {
            let nx = ((x as f64) - 0.5 * w) / w;
            let amp = sample_target_preset(preset, nx, ny);
            out_target[flatten_2d_index(x, y, width)] = amp;
        }
    }
}

#[verified_engine::verified]
fn sample_target_preset(preset: HologramTargetPreset, nx: f64, ny: f64) -> f64 {
    match preset {
        HologramTargetPreset::OffAxisRingAndSpot => {
            let r_ring = (nx + 0.20).hypot(ny + 0.18);
            let ring = (-((r_ring - 0.11) / 0.032).powi(2)).exp();
            let r_spot = (nx - 0.19).hypot(ny + 0.22);
            let spot = (-(r_spot / 0.048).powi(2)).exp();
            ring.max(spot)
        }
        HologramTargetPreset::AsymmetricReticle => {
            let horiz = (-0.32..=-0.06).contains(&nx) && (-0.28..=-0.21).contains(&ny);
            let vert = (-0.32..=-0.25).contains(&nx) && (-0.28..=-0.04).contains(&ny);
            let dot = (nx + 0.12).hypot(ny + 0.12) <= 0.04;
            if horiz || vert || dot { 1.0 } else { 0.0 }
        }
        HologramTargetPreset::UpperTriadSpots => {
            let d1 = (nx + 0.24).hypot(ny + 0.20);
            let d2 = (nx - 0.02).hypot(ny + 0.27);
            let d3 = (nx - 0.23).hypot(ny + 0.14);
            let d_min = d1.min(d2).min(d3);
            (-(d_min / 0.045).powi(2)).exp()
        }
        HologramTargetPreset::OffAxisGlyphF => {
            let spine = (-0.30..=-0.24).contains(&nx) && (-0.34..=-0.06).contains(&ny);
            let top = (-0.30..=-0.10).contains(&nx) && (-0.34..=-0.28).contains(&ny);
            let mid = (-0.30..=-0.14).contains(&nx) && (-0.22..=-0.17).contains(&ny);
            if spine || top || mid { 1.0 } else { 0.0 }
        }
    }
}

/// Pre-allocated Gerchberg-Saxton phase-retrieval solver for unit-modulus SLM holograms.
#[derive(Debug, Clone)]
pub struct GerchbergSaxtonSolver {
    width: usize,
    height: usize,
    regime: HologramRegime,
    propagator: AngularSpectrumPropagator,
    target_unit: Vec<f64>,
    signal_mask: Vec<u8>,
    twin_mask: Vec<u8>,
    slm_phase: Vec<f64>,
    slm_field: Vec<Complex<f64>>,
    replay_field: Vec<Complex<f64>>,
    back_field: Vec<Complex<f64>>,
    metrics: HologramMetrics,
}

impl GerchbergSaxtonSolver {
    /// Pre-allocates all complex wavefield, mask, and 2D FFT scratch buffers for `(width, height)`.
    #[must_use]
    pub fn new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        wavelength: f64,
        regime: HologramRegime,
    ) -> Self {
        let total = width * height;
        let propagator = AngularSpectrumPropagator::new(width, height, dx, dy, wavelength);
        let mut solver = Self {
            width,
            height,
            regime,
            propagator,
            target_unit: vec![0.0; total],
            signal_mask: vec![0; total],
            twin_mask: vec![0; total],
            slm_phase: vec![0.0; total],
            slm_field: vec![Complex::new(1.0, 0.0); total],
            replay_field: vec![Complex::new(0.0, 0.0); total],
            back_field: vec![Complex::new(0.0, 0.0); total],
            metrics: HologramMetrics::default(),
        };
        let mut default_target = vec![0.0; total];
        fill_target_amplitude_into(
            HologramTargetPreset::OffAxisRingAndSpot,
            width,
            height,
            &mut default_target,
        );
        solver.set_target_amplitude(&default_target);
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

    /// Active propagation regime.
    #[must_use]
    #[verified_engine::verified]
    pub fn regime(&self) -> HologramRegime {
        self.regime
    }

    /// Updates optical parameters and propagation regime without reallocating buffers.
    #[verified_engine::verified]
    pub fn reconfigure(&mut self, dx: f64, dy: f64, wavelength: f64, regime: HologramRegime) {
        self.propagator.reconfigure(dx, dy, wavelength);
        self.regime = regime;
    }

    /// Latest convergence and twin-leakage telemetry.
    #[must_use]
    #[verified_engine::verified]
    pub fn metrics(&self) -> HologramMetrics {
        self.metrics
    }

    /// Synthesized wrapped SLM phase mask $\phi(x, y) \in (-\pi, \pi]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn slm_phase(&self) -> &[f64] {
        &self.slm_phase
    }

    /// Unit-modulus complex SLM aperture field $e^{i\phi(x, y)}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn slm_field(&self) -> &[Complex<f64>] {
        &self.slm_field
    }

    /// Complex wavefield reconstructed in the target replay plane.
    #[must_use]
    #[verified_engine::verified]
    pub fn replay_field(&self) -> &[Complex<f64>] {
        &self.replay_field
    }

    /// Energy-normalized target amplitude distribution ($L^2$ norm equal to $\sqrt{W \cdot H}$).
    #[must_use]
    #[verified_engine::verified]
    pub fn target_amplitude(&self) -> &[f64] {
        &self.target_unit
    }

    /// Loads and energy-normalizes `target` ($\sum \hat{T}^2 = W \cdot H$) and builds twin masks
    /// with zero dynamic heap allocations.
    #[verified_engine::verified]
    pub fn set_target_amplitude(&mut self, target: &[f64]) {
        let total = self.width * self.height;
        assert_eq!(target.len(), total, "Target slice length mismatch");

        let mut energy = 0.0_f64;
        let mut peak = 0.0_f64;
        for &v in target {
            let pos = if v.is_finite() && v > 0.0 { v } else { 0.0 };
            energy += pos * pos;
            if pos > peak {
                peak = pos;
            }
        }
        let scale = if energy > 1.0e-24 {
            ((total as f64) / energy).sqrt()
        } else {
            0.0
        };
        let threshold = 0.10 * peak * scale;
        for (i, &v) in target.iter().enumerate() {
            let u = if v.is_finite() && v > 0.0 {
                v * scale
            } else {
                0.0
            };
            self.target_unit[i] = u;
            self.signal_mask[i] = u8::from(u > threshold && threshold > 0.0);
        }
        self.rebuild_twin_mask();
    }

    #[verified_engine::verified]
    fn rebuild_twin_mask(&mut self) {
        self.twin_mask.fill(0);
        for y in 0..self.height {
            let ty = (self.height - y) % self.height;
            for x in 0..self.width {
                let idx = flatten_2d_index(x, y, self.width);
                if self.signal_mask[idx] != 0 {
                    let tx = (self.width - x) % self.width;
                    let twin_idx = flatten_2d_index(tx, ty, self.width);
                    if self.signal_mask[twin_idx] == 0 {
                        self.twin_mask[twin_idx] = 1;
                    }
                }
            }
        }
    }

    /// Seeds uniform random SLM phases in $(-\pi, \pi]$ via `OxidizeRng` with zero heap allocations.
    #[verified_engine::verified]
    pub fn seed_random_phase(&mut self, rng: &mut OxidizeRng) {
        const INV_U64: f64 = 1.0 / (u64::MAX as f64);
        for (phi_slot, field_slot) in self.slm_phase.iter_mut().zip(self.slm_field.iter_mut()) {
            let u = (rng.next_u64() as f64) * INV_U64;
            let phi = (2.0 * u - 1.0) * PI;
            *phi_slot = phi;
            *field_slot = Complex::new(phi.cos(), phi.sin());
        }
        self.forward_slm_to_replay();
        self.metrics = self.compute_current_metrics(0);
    }

    /// Advances `steps` zero-allocation Gerchberg-Saxton iterations and returns updated [`HologramMetrics`].
    #[verified_engine::verified]
    pub fn step_iterations(&mut self, steps: usize) -> HologramMetrics {
        for _ in 0..steps {
            self.impose_replay_amplitude_constraint();
            self.backward_replay_to_slm();
            self.project_unit_modulus_slm_phase();
            self.forward_slm_to_replay();
            self.metrics = self.compute_current_metrics(self.metrics.iteration + 1);
        }
        self.metrics
    }

    #[verified_engine::verified]
    fn forward_slm_to_replay(&mut self) {
        match self.regime {
            HologramRegime::FourierLens => {
                self.propagator.centered_unitary_fft2d_into(
                    &self.slm_field,
                    false,
                    &mut self.replay_field,
                );
            }
            HologramRegime::FresnelAngularSpectrum { distance_z } => {
                self.propagator
                    .propagate_into(&self.slm_field, distance_z, &mut self.replay_field);
            }
        }
    }

    #[verified_engine::verified]
    fn impose_replay_amplitude_constraint(&mut self) {
        for (rep, &target_amp) in self.replay_field.iter_mut().zip(self.target_unit.iter()) {
            let mag = rep.norm();
            if mag > 1.0e-15 {
                *rep *= target_amp / mag;
            } else {
                *rep = Complex::new(target_amp, 0.0);
            }
        }
    }

    #[verified_engine::verified]
    fn backward_replay_to_slm(&mut self) {
        match self.regime {
            HologramRegime::FourierLens => {
                self.propagator.centered_unitary_fft2d_into(
                    &self.replay_field,
                    true,
                    &mut self.back_field,
                );
            }
            HologramRegime::FresnelAngularSpectrum { distance_z } => {
                self.propagator.propagate_into(
                    &self.replay_field,
                    -distance_z,
                    &mut self.back_field,
                );
            }
        }
    }

    #[verified_engine::verified]
    fn project_unit_modulus_slm_phase(&mut self) {
        for ((back, phi_out), slm_out) in self
            .back_field
            .iter()
            .zip(self.slm_phase.iter_mut())
            .zip(self.slm_field.iter_mut())
        {
            let phi = back.im.atan2(back.re);
            *phi_out = phi;
            *slm_out = Complex::new(phi.cos(), phi.sin());
        }
    }

    #[verified_engine::verified]
    fn compute_current_metrics(&self, iteration: usize) -> HologramMetrics {
        let mut dot_rt = 0.0_f64;
        let mut sum_r2 = 0.0_f64;
        let mut sum_t2 = 0.0_f64;
        let mut sig_power = 0.0_f64;
        let mut twin_power = 0.0_f64;

        for i in 0..self.replay_field.len() {
            let p = self.replay_field[i].norm_sqr();
            let r = p.sqrt();
            let t = self.target_unit[i];
            dot_rt += r * t;
            sum_r2 += p;
            sum_t2 += t * t;
            if self.signal_mask[i] != 0 {
                sig_power += p;
            } else if self.twin_mask[i] != 0 {
                twin_power += p;
            }
        }
        let alpha = if sum_r2 > 1.0e-24 {
            dot_rt / sum_r2
        } else {
            0.0
        };
        let mut err2 = 0.0_f64;
        for i in 0..self.replay_field.len() {
            let diff = alpha * self.replay_field[i].norm() - self.target_unit[i];
            err2 += diff * diff;
        }
        let nrmse = if sum_t2 > 1.0e-24 {
            (err2 / sum_t2).sqrt()
        } else {
            0.0
        };
        let inv_p = if sum_r2 > 1.0e-24 { 1.0 / sum_r2 } else { 0.0 };
        HologramMetrics {
            iteration,
            target_nrmse: nrmse,
            diffraction_efficiency: sig_power * inv_p,
            twin_leakage: twin_power * inv_p,
        }
    }

    /// Evaluates the real-scalar hologram baseline (`Im(U_slm) = 0`) to measure conjugate twin-image loss.
    #[verified_engine::verified]
    pub fn evaluate_real_scalar_baseline(&mut self) -> HologramMetrics {
        for (i, rep) in self.replay_field.iter_mut().enumerate() {
            *rep = Complex::new(self.target_unit[i], 0.0);
        }
        self.backward_replay_to_slm();
        for (back, slm) in self.back_field.iter().zip(self.slm_field.iter_mut()) {
            *slm = Complex::new(back.re, 0.0);
        }
        self.forward_slm_to_replay();
        self.compute_current_metrics(0)
    }

    /// Quantizes `slm_phase` $\in (-\pi, \pi]$ into calibrated 8-bit `[0..=255]` SLM grayscale levels.
    #[verified_engine::verified]
    pub fn quantize_phase_u8_into(&self, out_u8: &mut [u8]) {
        assert_eq!(
            out_u8.len(),
            self.slm_phase.len(),
            "8-bit phase buffer mismatch"
        );
        let inv_two_pi = 1.0 / (2.0 * PI);
        for (dst, &phi) in out_u8.iter_mut().zip(self.slm_phase.iter()) {
            let unit = ((phi + PI) * inv_two_pi).clamp(0.0, 1.0);
            *dst = (unit * 255.0).round() as u8;
        }
    }
}
