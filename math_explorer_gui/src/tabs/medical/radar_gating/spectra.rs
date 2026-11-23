//! Coarse FFT, `SpatialCztConfig` zoom, and `MusicEstimator` super-resolution range spectra.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::RadarGatingState;
use math_commons::constants::C;
use math_explorer::physics::medical::radar_gating::geometry::angle_resolution;
use math_explorer::physics::medical::radar_gating::{
    Beamformer, FmcwConfig, MusicEstimator, SpatialCztConfig,
};
use num_complex::Complex;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use std::f64::consts::PI;

/// Synchronized range-domain spectra and spatial beamformer diagnostics.
#[derive(Debug, Clone)]
pub struct SpectralSnapshot {
    /// Coarse un-zoomed FFT range profile (`[range_m, normalized_power]`).
    pub coarse_fft: Vec<[f64; 2]>,
    /// Fine-grid `SpatialCztConfig::process` zoom spectrum (`[range_m, normalized_power]`).
    pub czt_zoom: Vec<[f64; 2]>,
    /// `MusicEstimator::compute_spectrum` eigenspace pseudospectrum (`[range_m, normalized_power]`).
    pub music_pseudospectrum: Vec<[f64; 2]>,
    /// Rayleigh range-resolution limit $\Delta R = c / (2B)$ in meters.
    pub rayleigh_limit_m: f64,
    /// Array half-power angular resolution $\theta_{\text{res}}$ in degrees.
    pub angle_resolution_deg: f64,
    /// Normalized ULA beamformer gain `[0.0, 1.0]` toward the primary chest scatterer.
    pub primary_beam_gain: f64,
    /// Normalized ULA beamformer gain `[0.0, 1.0]` toward the secondary scatterer.
    pub secondary_beam_gain: f64,
    /// Number of prominent peaks resolved in the coarse FFT profile.
    pub coarse_peak_count: usize,
    /// Number of prominent peaks resolved in the MUSIC pseudospectrum.
    pub music_peak_count: usize,
}

impl Default for SpectralSnapshot {
    fn default() -> Self {
        Self {
            coarse_fft: Vec::new(),
            czt_zoom: Vec::new(),
            music_pseudospectrum: Vec::new(),
            rayleigh_limit_m: 0.0375,
            angle_resolution_deg: 14.3,
            primary_beam_gain: 1.0,
            secondary_beam_gain: 1.0,
            coarse_peak_count: 1,
            music_peak_count: 2,
        }
    }
}

/// Builds an `FmcwConfig` from user state (`60 GHz` carrier, `50 us` chirp duration).
pub fn build_fmcw_config(state: &RadarGatingState) -> FmcwConfig {
    FmcwConfig {
        bandwidth: state.bandwidth_ghz * 1.0e9,
        center_frequency: 60.0e9,
        chirp_time: 50.0e-6,
    }
}

/// Evaluates ULA directional gain toward a target angle using `Beamformer::steer`.
pub fn evaluate_ula_gain(
    beamformer: &Beamformer,
    num_antennas: usize,
    spacing_m: f64,
    wavelength_m: f64,
    target_angle_rad: f64,
    steering_rad: f64,
) -> f64 {
    let k_wave = 2.0 * PI / wavelength_m;
    let sin_target = target_angle_rad.sin();
    let mut rx_signals = [Complex::new(0.0, 0.0); 16];
    let count = num_antennas.min(rx_signals.len());
    for (idx, slot) in rx_signals.iter_mut().take(count).enumerate() {
        let phase = k_wave * (idx as f64 * spacing_m) * sin_target;
        *slot = Complex::new(phase.cos(), phase.sin());
    }
    let steered = beamformer.steer(&rx_signals[..count], steering_rad);
    (steered.norm() / count.max(1) as f64).clamp(0.0, 1.0)
}

/// Computes synchronized Coarse FFT, CZT zoom, and MUSIC super-resolution spectra.
pub fn compute_range_spectra(state: &RadarGatingState) -> SpectralSnapshot {
    let fmcw = build_fmcw_config(state);
    let wavelength = fmcw.wavelength();
    let spacing = 0.5 * wavelength;
    let steering_rad = state.steering_deg.to_radians();
    let beamformer = Beamformer::new_ula(state.num_antennas, spacing, wavelength);

    let res_rad = angle_resolution(wavelength, state.num_antennas, spacing, steering_rad);
    let angle_resolution_deg = if res_rad.is_finite() {
        res_rad.to_degrees().min(180.0)
    } else {
        180.0
    };

    let primary_gain = evaluate_ula_gain(
        &beamformer,
        state.num_antennas,
        spacing,
        wavelength,
        state.primary_target.angle_rad(),
        steering_rad,
    );
    let secondary_gain = evaluate_ula_gain(
        &beamformer,
        state.num_antennas,
        spacing,
        wavelength,
        state.secondary_target.angle_rad(),
        steering_rad,
    );

    let r1 = state.primary_target.range_m();
    let r2 = state.secondary_target.range_m();
    let center_r = 0.5 * (r1 + r2);
    let half_span = ((r2 - r1).abs() * 1.6 + 0.16).clamp(0.16, 0.65);
    let win_start = (center_r - half_span).max(0.08);
    let win_end = (center_r + half_span).min(2.80);

    let snapshots = synthesize_mimo_snapshots(state, &fmcw, primary_gain, secondary_gain);
    let first_chirp = &snapshots[0];
    let sample_rate = state.samples_per_chirp as f64 / fmcw.chirp_time;

    let coarse_fft = compute_coarse_fft_profile(first_chirp, &fmcw, win_start, win_end);
    let czt_zoom = compute_czt_profile(first_chirp, &fmcw, sample_rate, win_start, win_end, state);
    let music_pseudospectrum = compute_music_profile(&snapshots, &fmcw, win_start, win_end, state);

    let coarse_peak_count = count_prominent_peaks(&coarse_fft, 0.25, 0.08);
    let music_peak_count = count_prominent_peaks(&music_pseudospectrum, 0.22, 0.10);

    SpectralSnapshot {
        coarse_fft,
        czt_zoom,
        music_pseudospectrum,
        rayleigh_limit_m: fmcw.range_resolution(),
        angle_resolution_deg,
        primary_beam_gain: primary_gain,
        secondary_beam_gain: secondary_gain,
        coarse_peak_count,
        music_peak_count,
    }
}

fn synthesize_mimo_snapshots(
    state: &RadarGatingState,
    fmcw: &FmcwConfig,
    primary_gain: f64,
    secondary_gain: f64,
) -> Vec<Vec<Complex<f64>>> {
    let n = state.samples_per_chirp;
    let m_snaps = state.snapshots;
    let slope = fmcw.bandwidth / fmcw.chirp_time;
    let dt = fmcw.chirp_time / n as f64;
    let omega_1 = 2.0 * PI * FmcwConfig::beat_frequency(slope, state.primary_target.range_m());
    let omega_2 = 2.0 * PI * FmcwConfig::beat_frequency(slope, state.secondary_target.range_m());
    let amp_1 = (state.primary_target.rcs * primary_gain).max(0.05);
    let amp_2 = (state.secondary_target.rcs * secondary_gain).max(0.05);

    let mut rng = OxidizeRng::new(state.rng_seed ^ 0x5A17_6060);
    let mut snapshots = Vec::with_capacity(m_snaps);
    for m in 0..m_snaps {
        let phi_1 = 0.61 * (m as f64 + 1.0);
        let phi_2 = 1.73 * (m as f64 + 1.0) + 0.35 * (m as f64 * 0.9).sin();
        let mut chirp = Vec::with_capacity(n);
        for k in 0..n {
            let t_k = k as f64 * dt;
            let p1 = omega_1 * t_k + phi_1;
            let p2 = omega_2 * t_k + phi_2;
            let clean = Complex::new(p1.cos(), p1.sin()) * amp_1
                + Complex::new(p2.cos(), p2.sin()) * amp_2;
            let n_re = (unit_f64(&mut rng) - 0.5) * 2.0 * state.noise_sigma;
            let n_im = (unit_f64(&mut rng) - 0.5) * 2.0 * state.noise_sigma;
            chirp.push(clean + Complex::new(n_re, n_im));
        }
        snapshots.push(chirp);
    }
    snapshots
}

fn unit_f64(rng: &mut OxidizeRng) -> f64 {
    ((rng.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn compute_coarse_fft_profile(
    signal: &[Complex<f64>],
    fmcw: &FmcwConfig,
    win_start: f64,
    win_end: f64,
) -> Vec<[f64; 2]> {
    let n = signal.len();
    let df = 1.0 / fmcw.chirp_time;
    let mut raw = Vec::with_capacity(n);
    let mut max_power = 1.0e-12_f64;
    for bin in 0..n {
        let freq = bin as f64 * df;
        let range_m = fmcw.range_from_beat_frequency(freq);
        if range_m < win_start - fmcw.range_resolution()
            || range_m > win_end + fmcw.range_resolution()
        {
            continue;
        }
        let mut acc = Complex::new(0.0, 0.0);
        for (k, &sample) in signal.iter().enumerate() {
            let angle = -2.0 * PI * (bin * k) as f64 / n as f64;
            acc += sample * Complex::new(angle.cos(), angle.sin());
        }
        let power = acc.norm_sqr();
        max_power = max_power.max(power);
        raw.push([range_m, power]);
    }
    for pt in &mut raw {
        pt[1] /= max_power;
    }
    raw
}

fn compute_czt_profile(
    signal: &[Complex<f64>],
    fmcw: &FmcwConfig,
    sample_rate: f64,
    win_start: f64,
    win_end: f64,
    state: &RadarGatingState,
) -> Vec<[f64; 2]> {
    let bins = state.sweep_bins.max(2);
    let step_distance = (win_end - win_start) / (bins - 1) as f64;
    let cfg = SpatialCztConfig {
        start_distance: win_start,
        step_distance,
        output_bins: bins,
        bandwidth: fmcw.bandwidth,
        chirp_time: fmcw.chirp_time,
        c: C,
    };
    let czt_vals = cfg.process(signal, sample_rate);
    let max_power = czt_vals
        .iter()
        .map(|v| v.norm_sqr())
        .fold(1.0e-12_f64, f64::max);
    czt_vals
        .into_iter()
        .enumerate()
        .map(|(idx, val)| {
            let r = win_start + idx as f64 * step_distance;
            [r, val.norm_sqr() / max_power]
        })
        .collect()
}

fn compute_music_profile(
    snapshots: &[Vec<Complex<f64>>],
    fmcw: &FmcwConfig,
    win_start: f64,
    win_end: f64,
    state: &RadarGatingState,
) -> Vec<[f64; 2]> {
    let bins = state.sweep_bins.max(2);
    let step_range = (win_end - win_start) / (bins - 1) as f64;
    let Ok(mut estimator) = MusicEstimator::new(state.samples_per_chirp, snapshots.len(), 2) else {
        return Vec::new();
    };
    for snap in snapshots {
        if estimator.add_snapshot(snap).is_err() {
            return Vec::new();
        }
    }
    let Ok(spec) = estimator.compute_spectrum(win_start, win_end, step_range, fmcw.bandwidth, C)
    else {
        return Vec::new();
    };
    let log_vals: Vec<(f64, f64)> = spec
        .into_iter()
        .map(|(r, p)| (r, (1.0 + p.max(0.0)).ln()))
        .collect();
    let min_l = log_vals.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
    let max_l = log_vals.iter().map(|v| v.1).fold(1.0e-12_f64, f64::max);
    let denom = (max_l - min_l).max(1.0e-9);
    log_vals
        .into_iter()
        .map(|(r, l)| [r, ((l - min_l) / denom).clamp(0.0, 1.0)])
        .collect()
}

/// Counts local maxima exceeding `min_height` separated by valleys at least `min_prominence` deep.
pub fn count_prominent_peaks(curve: &[[f64; 2]], min_height: f64, min_prominence: f64) -> usize {
    if curve.len() < 3 {
        return 0;
    }
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    for i in 1..(curve.len() - 1) {
        let y = curve[i][1];
        if y >= min_height && y >= curve[i - 1][1] && y > curve[i + 1][1] {
            candidates.push((i, y));
        }
    }
    if candidates.len() <= 1 {
        return candidates.len();
    }
    let mut kept: Vec<(usize, f64)> = Vec::new();
    for (idx, height) in candidates {
        if let Some(&(prev_idx, prev_h)) = kept.last() {
            let valley = curve[prev_idx..=idx]
                .iter()
                .map(|pt| pt[1])
                .fold(f64::INFINITY, f64::min);
            let ref_peak = prev_h.min(height);
            if ref_peak - valley < min_prominence {
                if height > prev_h {
                    if let Some(last) = kept.last_mut() {
                        *last = (idx, height);
                    }
                }
                continue;
            }
        }
        kept.push((idx, height));
    }
    kept.len()
}

// [cite:mmwave_radiotherapy_setup]
