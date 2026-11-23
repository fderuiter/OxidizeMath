//! Zero-allocation 2D FFT Lenia simulation engine on a periodic torus.

use super::kernel::{GrowthParams, KernelShells};
use super::presets::{LeniaSpecies, seed_engine_species};
use super::{MAX_GRID_SIZE, MIN_GRID_SIZE};
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use oxidize_core::rng::OxidizeRng;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::TAU;
use std::sync::Arc;

const ACTIVE_CELL_THRESHOLD: f64 = 0.01;

/// Live macro-observable telemetry for a [`LeniaEngine`] simulation state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LeniaTelemetry {
    /// Total integrated mass $\sum_{x, y} A(x, y)$.
    pub total_mass: f64,
    /// Count of active cells exceeding `0.01` activation.
    pub active_area: f64,
    /// Circular-mean torus centroid coordinate along $x \in [0, W)$.
    pub centroid_x: f64,
    /// Circular-mean torus centroid coordinate along $y \in [0, H)$.
    pub centroid_y: f64,
    /// Instantaneous torus-wrapped centroid translation speed (cells per unit time).
    pub centroid_speed: f64,
    /// Cumulative simulated biological time $t$.
    pub time: f64,
    /// Total discrete integration steps executed.
    pub step_count: u64,
}

/// Spectral 2D Lenia simulation engine with pre-allocated `Complex<f64>` grids,
/// transpose workspace, and `rustfft` scratch buffers so [`LeniaEngine::step`] performs
/// zero heap allocations.
pub struct LeniaEngine {
    width: usize,
    height: usize,
    shells: KernelShells,
    growth: GrowthParams,
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

#[verified_engine::verified]
fn transpose_complex(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    src_rows: usize,
    src_cols: usize,
) {
    for row in 0..src_rows {
        for col in 0..src_cols {
            let src_idx = flatten_2d_index(col, row, src_cols);
            let dst_idx = flatten_2d_index(row, col, src_rows);
            dst[dst_idx] = src[src_idx];
        }
    }
}

#[verified_engine::verified]
fn build_trig_tables(len: usize) -> (Vec<f64>, Vec<f64>) {
    let mut cosines = Vec::with_capacity(len);
    let mut sines = Vec::with_capacity(len);
    let scale = TAU / (len as f64);
    for idx in 0..len {
        let angle = (idx as f64) * scale;
        cosines.push(angle.cos());
        sines.push(angle.sin());
    }
    (cosines, sines)
}

#[verified_engine::verified]
fn circular_coord(cos_sum: f64, sin_sum: f64, period: usize) -> f64 {
    if cos_sum.hypot(sin_sum) <= 1e-12 {
        return 0.5 * (period as f64);
    }
    let angle = sin_sum.atan2(cos_sum).rem_euclid(TAU);
    angle * (period as f64) / TAU
}

#[verified_engine::verified]
fn torus_delta(next: f64, prev: f64, period: usize) -> f64 {
    let p = period as f64;
    let half = 0.5 * p;
    (next - prev + half).rem_euclid(p) - half
}

impl LeniaEngine {
    /// Creates a pre-allocated 2D Lenia engine on a `width x height` periodic torus
    /// (dimensions clamped to `32..=256`) and precomputes the normalized spectrum $\hat{K}$.
    #[verified_engine::verified]
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
    /// Positive `delta` deposits activation; negative `delta` slices/erases activation.
    #[verified_engine::verified]
    pub fn paint_disc(&mut self, cx: usize, cy: usize, brush_radius: f64, delta: f64) {
        let r = brush_radius.clamp(1.0, 32.0);
        let span = r.ceil() as isize;
        let w = self.width as isize;
        let h = self.height as isize;
        for dy in -span..=span {
            let y = (cy as isize + dy).rem_euclid(h) as usize;
            for dx in -span..=span {
                let dist = (dx as f64).hypot(dy as f64);
                if dist <= r {
                    let x = (cx as isize + dx).rem_euclid(w) as usize;
                    let falloff = 0.5 * (1.0 + (std::f64::consts::PI * dist / r).cos());
                    let idx = flatten_2d_index(x, y, self.width);
                    self.state[idx] = (self.state[idx] + delta * falloff).clamp(0.0, 1.0);
                }
            }
        }
        self.refresh_telemetry();
    }

    /// Recomputes telemetry directly from the current `state` without advancing time.
    #[verified_engine::verified]
    pub fn refresh_telemetry(&mut self) {
        let (mass, active, cx, cy) = self.measure_spatial_moments();
        self.telemetry.total_mass = mass;
        self.telemetry.active_area = active;
        self.telemetry.centroid_x = cx;
        self.telemetry.centroid_y = cy;
        self.telemetry.centroid_speed = 0.0;
        self.has_prev_centroid = mass > 1e-6;
    }

    /// Resets simulation time and step counters after seeding a fresh initial condition.
    #[verified_engine::verified]
    pub(crate) fn reset_clock_and_measure(&mut self) {
        self.telemetry = LeniaTelemetry::default();
        self.has_prev_centroid = false;
        self.refresh_telemetry();
    }

    /// Advances the continuous Lenia PDE by one step:
    /// $A_{t+\Delta t} = \operatorname{clamp}(A_t + \Delta t \, G(\operatorname{IFFT}(\operatorname{FFT}(A_t)\odot \hat{K})), 0, 1)$.
    ///
    /// Executes in $O(W H \log(W H))$ using exclusively pre-allocated buffers (zero heap allocations).
    #[verified_engine::verified]
    pub fn step(&mut self) {
        for (dst, &val) in self.freq_buf.iter_mut().zip(self.state.iter()) {
            *dst = Complex::new(val, 0.0);
        }
        self.fft_2d_forward_in_place();
        for (freq, &k_hat) in self.freq_buf.iter_mut().zip(self.kernel_hat.iter()) {
            *freq *= k_hat;
        }
        self.fft_2d_inverse_in_place();
        self.apply_growth_and_update_telemetry();
    }

    #[verified_engine::verified]
    fn recompute_kernel_spectrum(&mut self) {
        self.shells
            .write_wrapped_spatial_kernel(self.width, self.height, &mut self.freq_buf);
        self.fft_2d_forward_in_place();
        let inv_area = 1.0 / ((self.width * self.height) as f64);
        for (dst, &src) in self.kernel_hat.iter_mut().zip(self.freq_buf.iter()) {
            *dst = src * inv_area;
        }
    }

    #[verified_engine::verified]
    fn fft_2d_forward_in_place(&mut self) {
        self.row_fft_fwd
            .process_with_scratch(&mut self.freq_buf, &mut self.scratch);
        transpose_complex(
            &self.freq_buf,
            &mut self.transpose_buf,
            self.height,
            self.width,
        );
        self.col_fft_fwd
            .process_with_scratch(&mut self.transpose_buf, &mut self.scratch);
        transpose_complex(
            &self.transpose_buf,
            &mut self.freq_buf,
            self.width,
            self.height,
        );
    }

    #[verified_engine::verified]
    fn fft_2d_inverse_in_place(&mut self) {
        self.row_fft_inv
            .process_with_scratch(&mut self.freq_buf, &mut self.scratch);
        transpose_complex(
            &self.freq_buf,
            &mut self.transpose_buf,
            self.height,
            self.width,
        );
        self.col_fft_inv
            .process_with_scratch(&mut self.transpose_buf, &mut self.scratch);
        transpose_complex(
            &self.transpose_buf,
            &mut self.freq_buf,
            self.width,
            self.height,
        );
    }

    #[verified_engine::verified]
    fn apply_growth_and_update_telemetry(&mut self) {
        let mut mass = 0.0;
        let mut active = 0.0;
        let mut sum_cx = 0.0;
        let mut sum_sx = 0.0;
        let mut sum_cy = 0.0;
        let mut sum_sy = 0.0;

        for y in 0..self.height {
            let cy_cos = self.cos_y[y];
            let cy_sin = self.sin_y[y];
            for x in 0..self.width {
                let idx = flatten_2d_index(x, y, self.width);
                let u = self.freq_buf[idx].re;
                self.potential[idx] = u;
                let g = self.growth.evaluate(u);
                let next_a = (self.state[idx] + self.dt * g).clamp(0.0, 1.0);
                self.state[idx] = next_a;
                mass += next_a;
                if next_a > ACTIVE_CELL_THRESHOLD {
                    active += 1.0;
                }
                sum_cx += next_a * self.cos_x[x];
                sum_sx += next_a * self.sin_x[x];
                sum_cy += next_a * cy_cos;
                sum_sy += next_a * cy_sin;
            }
        }
        self.commit_step_telemetry(mass, active, sum_cx, sum_sx, sum_cy, sum_sy);
    }

    #[verified_engine::verified]
    fn commit_step_telemetry(
        &mut self,
        mass: f64,
        active: f64,
        sum_cx: f64,
        sum_sx: f64,
        sum_cy: f64,
        sum_sy: f64,
    ) {
        let next_cx = circular_coord(sum_cx, sum_sx, self.width);
        let next_cy = circular_coord(sum_cy, sum_sy, self.height);
        let speed = if self.has_prev_centroid && mass > 1e-6 && self.dt > 1e-9 {
            let dx = torus_delta(next_cx, self.telemetry.centroid_x, self.width);
            let dy = torus_delta(next_cy, self.telemetry.centroid_y, self.height);
            dx.hypot(dy) / self.dt
        } else {
            0.0
        };
        self.has_prev_centroid = mass > 1e-6;
        self.telemetry.total_mass = mass;
        self.telemetry.active_area = active;
        self.telemetry.centroid_x = next_cx;
        self.telemetry.centroid_y = next_cy;
        self.telemetry.centroid_speed = speed;
        self.telemetry.time += self.dt;
        self.telemetry.step_count += 1;
    }

    #[verified_engine::verified]
    fn measure_spatial_moments(&self) -> (f64, f64, f64, f64) {
        let mut mass = 0.0;
        let mut active = 0.0;
        let mut sum_cx = 0.0;
        let mut sum_sx = 0.0;
        let mut sum_cy = 0.0;
        let mut sum_sy = 0.0;
        for y in 0..self.height {
            let cy_cos = self.cos_y[y];
            let cy_sin = self.sin_y[y];
            for x in 0..self.width {
                let val = self.state[flatten_2d_index(x, y, self.width)];
                mass += val;
                if val > ACTIVE_CELL_THRESHOLD {
                    active += 1.0;
                }
                sum_cx += val * self.cos_x[x];
                sum_sx += val * self.sin_x[x];
                sum_cy += val * cy_cos;
                sum_sy += val * cy_sin;
            }
        }
        let cx = circular_coord(sum_cx, sum_sx, self.width);
        let cy = circular_coord(sum_cy, sum_sy, self.height);
        (mass, active, cx, cy)
    }
}
