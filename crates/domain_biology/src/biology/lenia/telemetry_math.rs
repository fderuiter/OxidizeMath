//! Private trigonometric, transpose, separable 2D FFT, spatial-moment, and periodic-brush
//! helpers for [`super::engine::LeniaEngine`].

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use rustfft::Fft;
use std::f64::consts::{PI, TAU};

pub(super) const ACTIVE_CELL_THRESHOLD: f64 = 0.01;

/// Live macro-observable telemetry for a [`super::engine::LeniaEngine`] simulation state.
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

/// Spatial moment sums accumulated during a single `LeniaEngine::step` pass.
pub(super) struct StepMoments {
    pub mass: f64,
    pub active: f64,
    pub sum_cx: f64,
    pub sum_sx: f64,
    pub sum_cy: f64,
    pub sum_sy: f64,
}

#[verified_engine::verified]
pub(super) fn advance_step_telemetry(
    telemetry: &mut LeniaTelemetry,
    has_prev_centroid: &mut bool,
    width: usize,
    height: usize,
    dt: f64,
    m: &StepMoments,
) {
    let next_cx = circular_coord(m.sum_cx, m.sum_sx, width);
    let next_cy = circular_coord(m.sum_cy, m.sum_sy, height);
    let speed = if *has_prev_centroid && m.mass > 1e-6 && dt > 1e-9 {
        let dx = torus_delta(next_cx, telemetry.centroid_x, width);
        let dy = torus_delta(next_cy, telemetry.centroid_y, height);
        dx.hypot(dy) / dt
    } else {
        0.0
    };
    *has_prev_centroid = m.mass > 1e-6;
    telemetry.total_mass = m.mass;
    telemetry.active_area = m.active;
    telemetry.centroid_x = next_cx;
    telemetry.centroid_y = next_cy;
    telemetry.centroid_speed = speed;
    telemetry.time += dt;
    telemetry.step_count += 1;
}

#[verified_engine::verified]
pub(super) fn transpose_complex(
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
pub(super) fn run_separable_fft_2d(
    row_fft: &dyn Fft<f64>,
    col_fft: &dyn Fft<f64>,
    width: usize,
    height: usize,
    freq_buf: &mut [Complex<f64>],
    transpose_buf: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    row_fft.process_with_scratch(freq_buf, scratch);
    transpose_complex(freq_buf, transpose_buf, height, width);
    col_fft.process_with_scratch(transpose_buf, scratch);
    transpose_complex(transpose_buf, freq_buf, width, height);
}

#[verified_engine::verified]
pub(super) fn build_trig_tables(len: usize) -> (Vec<f64>, Vec<f64>) {
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
pub(super) fn circular_coord(cos_sum: f64, sin_sum: f64, period: usize) -> f64 {
    if cos_sum.hypot(sin_sum) <= 1e-12 {
        return 0.5 * (period as f64);
    }
    let angle = sin_sum.atan2(cos_sum).rem_euclid(TAU);
    angle * (period as f64) / TAU
}

#[verified_engine::verified]
pub(super) fn torus_delta(next: f64, prev: f64, period: usize) -> f64 {
    let p = period as f64;
    let half = 0.5 * p;
    (next - prev + half).rem_euclid(p) - half
}

#[verified_engine::verified]
pub(super) fn paint_periodic_disc(
    state: &mut [f64],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    brush_radius: f64,
    delta: f64,
) {
    let r = brush_radius.clamp(1.0, 32.0);
    let span = r.ceil() as isize;
    let w = width as isize;
    let h = height as isize;
    for dy in -span..=span {
        let y = (cy as isize + dy).rem_euclid(h) as usize;
        for dx in -span..=span {
            let dist = (dx as f64).hypot(dy as f64);
            if dist <= r {
                let x = (cx as isize + dx).rem_euclid(w) as usize;
                let falloff = 0.5 * (1.0 + (PI * dist / r).cos());
                let idx = flatten_2d_index(x, y, width);
                state[idx] = (state[idx] + delta * falloff).clamp(0.0, 1.0);
            }
        }
    }
}

#[verified_engine::verified]
pub(super) fn measure_grid_moments(
    state: &[f64],
    width: usize,
    height: usize,
    cos_x: &[f64],
    sin_x: &[f64],
    cos_y: &[f64],
    sin_y: &[f64],
) -> (f64, f64, f64, f64) {
    let mut mass = 0.0;
    let mut active = 0.0;
    let mut sum_cx = 0.0;
    let mut sum_sx = 0.0;
    let mut sum_cy = 0.0;
    let mut sum_sy = 0.0;
    for y in 0..height {
        let cy_cos = cos_y[y];
        let cy_sin = sin_y[y];
        for x in 0..width {
            let val = state[flatten_2d_index(x, y, width)];
            mass += val;
            if val > ACTIVE_CELL_THRESHOLD {
                active += 1.0;
            }
            sum_cx += val * cos_x[x];
            sum_sx += val * sin_x[x];
            sum_cy += val * cy_cos;
            sum_sy += val * cy_sin;
        }
    }
    let cx = circular_coord(sum_cx, sum_sx, width);
    let cy = circular_coord(sum_cy, sum_sy, height);
    (mass, active, cx, cy)
}
