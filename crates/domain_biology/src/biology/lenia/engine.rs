//! Zero-allocation 2D FFT Lenia simulation engine on a periodic torus.

use super::kernel::{CompiledGrowthRule, CompiledKernelRule, GrowthParams, KernelShells};
use super::presets::{LeniaSpecies, seed_engine_species};
pub use super::telemetry_math::LeniaTelemetry;
use super::telemetry_math::{
    ACTIVE_CELL_THRESHOLD, StepMoments, advance_step_telemetry, build_trig_tables,
    measure_grid_moments, paint_periodic_disc, run_separable_fft_2d,
};
use super::{MAX_GRID_SIZE, MIN_GRID_SIZE};
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use oxidize_core::rng::OxidizeRng;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Spectral 2D Lenia simulation engine with pre-allocated `Complex<f64>` grids,
/// transpose workspace, and `rustfft` scratch buffers so [`LeniaEngine::step`] performs
/// zero heap allocations across both parametric and [`CompiledGrowthRule`] formulas.
pub struct LeniaEngine {
    width: usize,
    height: usize,
    shells: KernelShells,
    growth: GrowthParams,
    custom_growth: Option<CompiledGrowthRule>,
    custom_kernel: Option<CompiledKernelRule>,
    dt: f64,
    state: Vec<f64>,
    potential: Vec<f64>,
    freq_buf: Vec<Complex<f64>>,
    transpose_buf: Vec<Complex<f64>>,
    kernel_hat: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    row_fft_fwd: Arc<dyn Fft<f64>>,
    col_fft_fwd: Arc<dyn Fft<f64>>,
    row_fft_inv: Arc<dyn Fft<f64>>,
    col_fft_inv: Arc<dyn Fft<f64>>,
    cos_x: Vec<f64>,
    sin_x: Vec<f64>,
    cos_y: Vec<f64>,
    sin_y: Vec<f64>,
    has_prev_centroid: bool,
    telemetry: LeniaTelemetry,
}

impl LeniaEngine {
    /// Creates a pre-allocated 2D Lenia engine on a `width x height` periodic torus
    /// (dimensions clamped to `32..=256`) and precomputes the normalized spectrum $\hat{K}$.
    #[verified_engine::verified(alloc_ok = "constructor pre-allocates state")]
    pub fn new(
        width: usize,
        height: usize,
        shells: KernelShells,
        growth: GrowthParams,
        dt: f64,
    ) -> Self {
        let w = width.clamp(MIN_GRID_SIZE, MAX_GRID_SIZE);
        let h = height.clamp(MIN_GRID_SIZE, MAX_GRID_SIZE);
        let total = w * h;
        let mut planner = FftPlanner::new();
        let row_fft_fwd = planner.plan_fft_forward(w);
        let col_fft_fwd = planner.plan_fft_forward(h);
        let row_fft_inv = planner.plan_fft_inverse(w);
        let col_fft_inv = planner.plan_fft_inverse(h);
        let scratch_len = row_fft_fwd
            .get_inplace_scratch_len()
            .max(col_fft_fwd.get_inplace_scratch_len())
            .max(row_fft_inv.get_inplace_scratch_len())
            .max(col_fft_inv.get_inplace_scratch_len());
        let (cos_x, sin_x) = build_trig_tables(w);
        let (cos_y, sin_y) = build_trig_tables(h);

        let mut engine = Self {
            width: w,
            height: h,
            shells,
            growth,
            custom_growth: None,
            custom_kernel: None,
            dt: if dt.is_finite() { dt.max(1e-4) } else { 0.1 },
            state: vec![0.0; total],
            potential: vec![0.0; total],
            freq_buf: vec![Complex::new(0.0, 0.0); total],
            transpose_buf: vec![Complex::new(0.0, 0.0); total],
            kernel_hat: vec![Complex::new(0.0, 0.0); total],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            row_fft_fwd,
            col_fft_fwd,
            row_fft_inv,
            col_fft_inv,
            cos_x,
            sin_x,
            cos_y,
            sin_y,
            has_prev_centroid: false,
            telemetry: LeniaTelemetry::default(),
        };
        engine.recompute_kernel_spectrum();
        engine
    }

    /// Initializes a pre-allocated engine seeded with a deterministic [`LeniaSpecies`] soliton
    /// and bounded perturbation drawn from `rng`.
    #[verified_engine::verified]
    pub fn from_species(
        width: usize,
        height: usize,
        species: LeniaSpecies,
        perturbation: f64,
        rng: &mut OxidizeRng,
    ) -> Self {
        let genome = species.genome();
        let mut engine = Self::new(width, height, genome.shells, genome.growth, genome.dt);
        seed_engine_species(&mut engine, species, perturbation, rng);
        engine
    }

    /// Grid width $W$.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height $H$.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Active concentric-ring kernel specification.
    #[verified_engine::verified]
    pub fn shells(&self) -> &KernelShells {
        &self.shells
    }

    /// Active Gaussian growth parameters.
    #[verified_engine::verified]
    pub fn growth(&self) -> GrowthParams {
        self.growth
    }

    /// Optional custom compiled growth formula overriding `self.growth()`.
    #[must_use]
    #[verified_engine::verified]
    pub fn custom_growth(&self) -> Option<&CompiledGrowthRule> {
        self.custom_growth.as_ref()
    }

    /// Optional custom compiled radial kernel formula overriding `self.shells()` ring profile.
    #[must_use]
    #[verified_engine::verified]
    pub fn custom_kernel(&self) -> Option<&CompiledKernelRule> {
        self.custom_kernel.as_ref()
    }

    /// Installs or clears a custom compiled growth formula $G(u) \in [-1, 1]$.
    #[verified_engine::verified]
    pub fn set_custom_growth(&mut self, rule: Option<CompiledGrowthRule>) {
        self.custom_growth = rule;
    }

    /// Installs or clears a custom compiled radial kernel formula $K(r)$ and recomputes $\hat{K}$ if changed.
    #[verified_engine::verified]
    pub fn set_custom_kernel(&mut self, rule: Option<CompiledKernelRule>) {
        if self.custom_kernel != rule {
            self.custom_kernel = rule;
            self.recompute_kernel_spectrum();
            if self.telemetry.total_mass > 1e-9 {
                self.compute_potential_in_place();
            }
        }
    }

    /// Evaluates the active growth rule (`custom_growth` if set, otherwise Gaussian `growth`).
    #[verified_engine::verified]
    pub fn evaluate_growth(&self, u: f64) -> f64 {
        if let Some(rule) = &self.custom_growth {
            rule.evaluate(u)
        } else {
            self.growth.evaluate(u)
        }
    }

    /// Samples the active `[u, G(u)]` curve across $u \in [0, 1]$.
    #[verified_engine::verified]
    pub fn sample_growth_curve(&self, samples: usize) -> Vec<[f64; 2]> {
        if let Some(rule) = &self.custom_growth {
            rule.sample_curve(samples)
        } else {
            self.growth.sample_curve(samples)
        }
    }

    /// Samples the active `[r_norm, K(r_norm)]` radial kernel profile across $r_{\text{norm}} \in [0, 1]$.
    #[verified_engine::verified]
    pub fn sample_radial_kernel(&self, samples: usize) -> Vec<[f64; 2]> {
        if let Some(rule) = &self.custom_kernel {
            rule.sample_radial_profile(samples)
        } else {
            self.shells.sample_radial_profile(samples)
        }
    }

    /// Integration time step $\Delta t$.
    #[verified_engine::verified]
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Updates $\Delta t$.
    #[verified_engine::verified]
    pub fn set_dt(&mut self, dt: f64) {
        if dt.is_finite() && dt > 0.0 {
            self.dt = dt;
        }
    }

    /// Updates Gaussian growth parameters without recomputing $\hat{K}$.
    #[verified_engine::verified]
    pub fn set_growth(&mut self, growth: GrowthParams) {
        self.growth = growth;
    }

    /// Updates concentric-ring kernel shells and recomputes $\hat{K}$ only if changed.
    #[verified_engine::verified]
    pub fn set_shells(&mut self, shells: KernelShells) {
        if self.shells != shells {
            self.shells = shells;
            self.recompute_kernel_spectrum();
            if self.telemetry.total_mass > 1e-9 {
                self.compute_potential_in_place();
            }
        }
    }

    /// Batch-updates kernel shells, growth curve, and $\Delta t$, recomputing $\hat{K}$ only when `shells` changed.
    #[verified_engine::verified]
    pub fn update_parameters(&mut self, shells: KernelShells, growth: GrowthParams, dt: f64) {
        self.set_shells(shells);
        self.set_growth(growth);
        self.set_dt(dt);
    }

    /// Row-major cell activation field $A(x, y) \in [0, 1]$ of length `width * height`.
    #[verified_engine::verified]
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// Mutable row-major cell activation field $A(x, y)$ of length `width * height`.
    #[verified_engine::verified]
    pub fn state_mut(&mut self) -> &mut [f64] {
        &mut self.state
    }

    /// Most recently computed neighborhood potential $U = A * K$.
    #[verified_engine::verified]
    pub fn potential(&self) -> &[f64] {
        &self.potential
    }

    /// Maximum finite neighborhood potential $\max_{x, y} U(x, y)$ across the grid (`0.0` on vacuum).
    #[verified_engine::verified]
    pub fn max_potential(&self) -> f64 {
        let mut peak = 0.0_f64;
        for &u in &self.potential {
            if u.is_finite() && u > peak {
                peak = u;
            }
        }
        peak
    }

    /// Precomputed normalized spectrum $\hat{K} / (W \cdot H)$.
    #[verified_engine::verified]
    pub fn kernel_hat(&self) -> &[Complex<f64>] {
        &self.kernel_hat
    }

    /// Latest simulation telemetry snapshot.
    #[verified_engine::verified]
    pub fn telemetry(&self) -> LeniaTelemetry {
        self.telemetry
    }

    /// Resets all activations, potentials, and telemetry counters to zero.
    #[verified_engine::verified]
    pub fn clear(&mut self) {
        self.state.fill(0.0);
        self.potential.fill(0.0);
        self.has_prev_centroid = false;
        self.telemetry = LeniaTelemetry::default();
    }

    /// Seeds a [`LeniaSpecies`] genome and soliton onto the existing grid using `rng`.
    #[verified_engine::verified]
    pub fn seed_species(&mut self, species: LeniaSpecies, perturbation: f64, rng: &mut OxidizeRng) {
        seed_engine_species(self, species, perturbation, rng);
    }

    /// Paints or carves a smooth circular brush stroke centered at `(cx, cy)` with periodic wrapping.
    #[verified_engine::verified]
    pub fn paint_disc(&mut self, cx: usize, cy: usize, brush_radius: f64, delta: f64) {
        paint_periodic_disc(
            &mut self.state,
            self.width,
            self.height,
            cx,
            cy,
            brush_radius,
            delta,
        );
        self.refresh_telemetry();
    }

    /// Recomputes telemetry and neighborhood potential $U = A * K$ from `state` without advancing time.
    #[verified_engine::verified]
    pub fn refresh_telemetry(&mut self) {
        let (mass, active, cx, cy) = measure_grid_moments(
            &self.state,
            self.width,
            self.height,
            &self.cos_x,
            &self.sin_x,
            &self.cos_y,
            &self.sin_y,
        );
        self.telemetry.total_mass = mass;
        self.telemetry.active_area = active;
        self.telemetry.centroid_x = cx;
        self.telemetry.centroid_y = cy;
        self.telemetry.centroid_speed = 0.0;
        self.has_prev_centroid = mass > 1e-6;
        if mass > 1e-9 {
            self.compute_potential_in_place();
        } else {
            self.potential.fill(0.0);
        }
    }

    /// Resets simulation time and step counters after seeding a fresh initial condition.
    #[verified_engine::verified]
    pub(crate) fn reset_clock_and_measure(&mut self) {
        self.telemetry = LeniaTelemetry::default();
        self.has_prev_centroid = false;
        self.refresh_telemetry();
    }

    /// Recomputes `self.potential` ($U = \mathcal{F}^{-1}(\mathcal{F}(A) \odot \hat{K})$) in-place with zero allocations.
    #[verified_engine::verified]
    pub fn compute_potential_in_place(&mut self) {
        self.convolve_state_into_freq_buf();
        for (dst, src) in self.potential.iter_mut().zip(self.freq_buf.iter()) {
            *dst = src.re;
        }
    }

    /// Advances the continuous Lenia PDE by one step in $O(W H \log(W H))$ with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self) {
        self.convolve_state_into_freq_buf();
        self.apply_growth_and_update_telemetry();
    }

    #[verified_engine::verified]
    fn convolve_state_into_freq_buf(&mut self) {
        for (dst, &val) in self.freq_buf.iter_mut().zip(self.state.iter()) {
            *dst = Complex::new(val, 0.0);
        }
        self.fft_2d_forward_in_place();
        for (freq, &k_hat) in self.freq_buf.iter_mut().zip(self.kernel_hat.iter()) {
            *freq *= k_hat;
        }
        self.fft_2d_inverse_in_place();
    }

    #[verified_engine::verified]
    fn recompute_kernel_spectrum(&mut self) {
        if let Some(rule) = &self.custom_kernel {
            rule.write_wrapped_spatial_kernel(
                self.width,
                self.height,
                self.shells.radius(),
                &mut self.freq_buf,
            );
        } else {
            self.shells
                .write_wrapped_spatial_kernel(self.width, self.height, &mut self.freq_buf);
        }
        self.fft_2d_forward_in_place();
        let inv_area = 1.0 / ((self.width * self.height) as f64);
        for (dst, &src) in self.kernel_hat.iter_mut().zip(self.freq_buf.iter()) {
            *dst = src * inv_area;
        }
    }

    #[verified_engine::verified]
    fn fft_2d_forward_in_place(&mut self) {
        run_separable_fft_2d(
            self.row_fft_fwd.as_ref(),
            self.col_fft_fwd.as_ref(),
            self.width,
            self.height,
            &mut self.freq_buf,
            &mut self.transpose_buf,
            &mut self.scratch,
        );
    }

    #[verified_engine::verified]
    fn fft_2d_inverse_in_place(&mut self) {
        run_separable_fft_2d(
            self.row_fft_inv.as_ref(),
            self.col_fft_inv.as_ref(),
            self.width,
            self.height,
            &mut self.freq_buf,
            &mut self.transpose_buf,
            &mut self.scratch,
        );
    }

    #[verified_engine::verified]
    fn apply_growth_and_update_telemetry(&mut self) {
        let mut moments = StepMoments {
            mass: 0.0,
            active: 0.0,
            sum_cx: 0.0,
            sum_sx: 0.0,
            sum_cy: 0.0,
            sum_sy: 0.0,
        };

        for y in 0..self.height {
            let cy_cos = self.cos_y[y];
            let cy_sin = self.sin_y[y];
            for x in 0..self.width {
                let idx = flatten_2d_index(x, y, self.width);
                let u = self.freq_buf[idx].re;
                self.potential[idx] = u;
                let g = if let Some(rule) = &self.custom_growth {
                    rule.evaluate(u)
                } else {
                    self.growth.evaluate(u)
                };
                let next_a = (self.state[idx] + self.dt * g).clamp(0.0, 1.0);
                self.state[idx] = next_a;
                moments.mass += next_a;
                if next_a > ACTIVE_CELL_THRESHOLD {
                    moments.active += 1.0;
                }
                moments.sum_cx += next_a * self.cos_x[x];
                moments.sum_sx += next_a * self.sin_x[x];
                moments.sum_cy += next_a * cy_cos;
                moments.sum_sy += next_a * cy_sin;
            }
        }
        advance_step_telemetry(
            &mut self.telemetry,
            &mut self.has_prev_centroid,
            self.width,
            self.height,
            self.dt,
            &moments,
        );
    }
}
