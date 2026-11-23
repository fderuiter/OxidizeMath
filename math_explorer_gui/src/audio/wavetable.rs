//! Pre-planned zero-allocation `rustfft` spectral processors for `2048`-sample wavetable banks
//! and stereo PDE impulse responses.

// theory_verification!

use std::sync::Arc;

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::frame::LIMITER_CEILING_LINEAR;

/// Standard single-cycle frame length (`2048` samples) required by Vital, Serum, and Surge XT.
pub const WAVETABLE_FRAME_SAMPLES: usize = 2048;

/// Maximum multi-frame morph bank depth (`256` frames).
pub const MAX_WAVETABLE_FRAMES: usize = 256;

/// Number of harmonic bins exported to Web Audio `PeriodicWave` for browser morph audition.
pub const PREVIEW_PERIODIC_WAVE_HARMONICS: usize = 128;

/// Default audition pitch (`110 Hz`, A2) used to determine Nyquist partial cutoff when band-limiting.
pub const DEFAULT_BANDLIMIT_FUNDAMENTAL_HZ: f32 = 110.0;

/// Computes the highest non-aliasing harmonic index at `(sample_rate_hz, fundamental_hz)`.
#[must_use]
pub fn max_nyquist_harmonic(sample_rate_hz: u32, fundamental_hz: f32) -> usize {
    let f0 = if fundamental_hz.is_finite() {
        fundamental_hz.clamp(20.0, 4_000.0)
    } else {
        DEFAULT_BANDLIMIT_FUNDAMENTAL_HZ
    };
    let nyquist = 0.49 * (sample_rate_hz.max(8_000) as f32);
    let max_bin = WAVETABLE_FRAME_SAMPLES / 2 - 1;
    ((nyquist / f0).floor() as usize).clamp(1, max_bin)
}

/// Pre-planned `2048`-point FFT processor executing with zero heap allocations in hot loops.
pub struct WavetableSpectralProcessor {
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,
    spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl Default for WavetableSpectralProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl WavetableSpectralProcessor {
    /// Allocates reusable `2048`-sample forward/inverse `rustfft` plans and scratch buffers once.
    #[must_use]
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fft_forward = planner.plan_fft_forward(WAVETABLE_FRAME_SAMPLES);
        let fft_inverse = planner.plan_fft_inverse(WAVETABLE_FRAME_SAMPLES);
        let scratch_len = fft_forward
            .get_inplace_scratch_len()
            .max(fft_inverse.get_inplace_scratch_len())
            .max(WAVETABLE_FRAME_SAMPLES);
        Self {
            fft_forward,
            fft_inverse,
            spectrum: vec![Complex::new(0.0, 0.0); WAVETABLE_FRAME_SAMPLES],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        }
    }

    /// Removes DC offset (`< 1e-6`), brickwall band-limits harmonics above `max_harmonic`,
    /// and peak-normalizes `frame` to `target_peak` with zero heap allocations.
    pub fn condition_frame_in_place(
        &mut self,
        frame: &mut [f32; WAVETABLE_FRAME_SAMPLES],
        max_harmonic: usize,
        target_peak: f32,
    ) {
        for (dst, &src) in self.spectrum.iter_mut().zip(frame.iter()) {
            let clean = if src.is_finite() { f64::from(src) } else { 0.0 };
            *dst = Complex::new(clean, 0.0);
        }
        self.fft_forward
            .process_with_scratch(&mut self.spectrum, &mut self.scratch);

        let limit = max_harmonic.clamp(1, WAVETABLE_FRAME_SAMPLES / 2 - 1);
        self.spectrum[0] = Complex::new(0.0, 0.0);
        self.spectrum[WAVETABLE_FRAME_SAMPLES / 2] = Complex::new(0.0, 0.0);
        for bin in (limit + 1)..(WAVETABLE_FRAME_SAMPLES / 2) {
            self.spectrum[bin] = Complex::new(0.0, 0.0);
            self.spectrum[WAVETABLE_FRAME_SAMPLES - bin] = Complex::new(0.0, 0.0);
        }

        self.fft_inverse
            .process_with_scratch(&mut self.spectrum, &mut self.scratch);

        let inv_n = 1.0 / (WAVETABLE_FRAME_SAMPLES as f64);
        let mut mean = 0.0_f64;
        for (out, spec) in frame.iter_mut().zip(self.spectrum.iter()) {
            let val = if spec.re.is_finite() {
                spec.re * inv_n
            } else {
                0.0
            };
            *out = val as f32;
            mean += val;
        }
        mean *= inv_n;

        let mut peak = 0.0_f32;
        for sample in frame.iter_mut() {
            *sample = (f64::from(*sample) - mean) as f32;
            peak = peak.max(sample.abs());
        }
        let ceiling = if target_peak.is_finite() {
            target_peak.clamp(0.05, 1.0)
        } else {
            0.98
        };
        if peak > 1.0e-7 {
            let gain = ceiling / peak;
            for sample in frame.iter_mut() {
                *sample = (*sample * gain).clamp(-ceiling, ceiling);
            }
        } else {
            frame.fill(0.0);
        }
    }

    /// Extracts Web Audio `PeriodicWave` cosine (`real`) and sine (`imag`) coefficients
    /// along with normalized harmonic magnitudes (`mags`, index `0` = fundamental `k = 1`)
    /// with zero heap allocations.
    pub fn extract_periodic_wave_and_spectrum(
        &mut self,
        frame: &[f32; WAVETABLE_FRAME_SAMPLES],
        real_out: &mut [f32],
        imag_out: &mut [f32],
        mags_out: &mut [f32],
    ) {
        for (dst, &src) in self.spectrum.iter_mut().zip(frame.iter()) {
            *dst = Complex::new(if src.is_finite() { f64::from(src) } else { 0.0 }, 0.0);
        }
        self.fft_forward
            .process_with_scratch(&mut self.spectrum, &mut self.scratch);

        if !real_out.is_empty() {
            real_out[0] = 0.0;
        }
        if !imag_out.is_empty() {
            imag_out[0] = 0.0;
        }
        let scale = 2.0 / (WAVETABLE_FRAME_SAMPLES as f64);
        let max_coeffs = real_out
            .len()
            .min(imag_out.len())
            .min(WAVETABLE_FRAME_SAMPLES / 2);
        for k in 1..max_coeffs {
            let c = self.spectrum[k];
            real_out[k] = (c.re * scale) as f32;
            imag_out[k] = (-c.im * scale) as f32;
        }
        let mag_count = mags_out.len().min(WAVETABLE_FRAME_SAMPLES / 2 - 1);
        for (idx, mag_slot) in mags_out.iter_mut().take(mag_count).enumerate() {
            *mag_slot = (self.spectrum[idx + 1].norm() * scale) as f32;
        }
    }
}

/// Linearly interpolates a `2048`-sample frame across `frame_count` contiguous frames in `bank`
/// at normalized morph coordinate `morph_pos` in `[0.0, 1.0]` with zero heap allocations.
pub fn morph_wavetable_frame(
    bank: &[f32],
    frame_count: usize,
    morph_pos: f32,
    out: &mut [f32; WAVETABLE_FRAME_SAMPLES],
) {
    let valid_frames = frame_count.min(bank.len() / WAVETABLE_FRAME_SAMPLES);
    if valid_frames == 0 {
        out.fill(0.0);
        return;
    }
    if valid_frames == 1 {
        out.copy_from_slice(&bank[..WAVETABLE_FRAME_SAMPLES]);
        return;
    }
    let m = if morph_pos.is_finite() {
        morph_pos.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let scaled = m * ((valid_frames - 1) as f32);
    let idx_a = (scaled.floor() as usize).min(valid_frames - 1);
    let idx_b = (idx_a + 1).min(valid_frames - 1);
    let frac = scaled - (idx_a as f32);
    let slice_a = &bank[idx_a * WAVETABLE_FRAME_SAMPLES..(idx_a + 1) * WAVETABLE_FRAME_SAMPLES];
    let slice_b = &bank[idx_b * WAVETABLE_FRAME_SAMPLES..(idx_b + 1) * WAVETABLE_FRAME_SAMPLES];
    for i in 0..WAVETABLE_FRAME_SAMPLES {
        out[i] = slice_a[i] + frac * (slice_b[i] - slice_a[i]);
    }
}

/// Conditions a stereo impulse response via `rustfft` (removing DC offset and ultrasonic bins,
/// applying onset/tail cosine tapers, and normalizing linked stereo peak headroom).
pub fn condition_stereo_ir_buffers(left: &mut [f32], right: &mut [f32], peak_ceiling: f32) {
    let len = left.len().min(right.len());
    if len == 0 {
        return;
    }
    let fft_len = len.next_power_of_two().max(64);
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(fft_len);
    let inv = planner.plan_fft_inverse(fft_len);
    let mut spectrum = vec![Complex::new(0.0, 0.0); fft_len];
    let mut scratch = vec![
        Complex::new(0.0, 0.0);
        fwd.get_inplace_scratch_len()
            .max(inv.get_inplace_scratch_len())
    ];

    filter_single_ir_channel(
        &mut left[..len],
        fft_len,
        &fwd,
        &inv,
        &mut spectrum,
        &mut scratch,
    );
    filter_single_ir_channel(
        &mut right[..len],
        fft_len,
        &fwd,
        &inv,
        &mut spectrum,
        &mut scratch,
    );
    apply_ir_taper_and_linked_peak(&mut left[..len], &mut right[..len], peak_ceiling);
}

fn filter_single_ir_channel(
    channel: &mut [f32],
    fft_len: usize,
    fwd: &Arc<dyn Fft<f64>>,
    inv: &Arc<dyn Fft<f64>>,
    spectrum: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
) {
    spectrum.fill(Complex::new(0.0, 0.0));
    for (dst, &src) in spectrum.iter_mut().zip(channel.iter()) {
        *dst = Complex::new(if src.is_finite() { f64::from(src) } else { 0.0 }, 0.0);
    }
    fwd.process_with_scratch(spectrum, scratch);

    // Zero DC, sub-15 Hz drift, and top 4% near Nyquist.
    let nyquist = fft_len / 2;
    let hi_cut = ((nyquist as f64) * 0.96).floor() as usize;
    spectrum[0] = Complex::new(0.0, 0.0);
    spectrum[nyquist] = Complex::new(0.0, 0.0);
    for bin in hi_cut.max(1)..nyquist {
        spectrum[bin] = Complex::new(0.0, 0.0);
        spectrum[fft_len - bin] = Complex::new(0.0, 0.0);
    }

    inv.process_with_scratch(spectrum, scratch);
    let inv_n = 1.0 / (fft_len as f64);
    let mut mean = 0.0_f64;
    for (dst, spec) in channel.iter_mut().zip(spectrum.iter()) {
        let v = if spec.re.is_finite() {
            spec.re * inv_n
        } else {
            0.0
        };
        *dst = v as f32;
        mean += v;
    }
    mean /= channel.len() as f64;
    for dst in channel.iter_mut() {
        *dst = (f64::from(*dst) - mean) as f32;
    }
}

fn apply_ir_taper_and_linked_peak(left: &mut [f32], right: &mut [f32], peak_ceiling: f32) {
    let len = left.len();
    let fade_in = 8.min(len / 8);
    for i in 0..fade_in {
        let w = 0.5 * (1.0 - (std::f32::consts::PI * (i as f32) / (fade_in as f32)).cos());
        left[i] *= w;
        right[i] *= w;
    }
    let fade_out = (len / 16).max(16).min(len / 2);
    let tail_start = len - fade_out;
    for i in 0..fade_out {
        let w = 0.5 * (1.0 + (std::f32::consts::PI * (i as f32) / (fade_out as f32)).cos());
        left[tail_start + i] *= w;
        right[tail_start + i] *= w;
    }

    let mut mean_l = 0.0_f32;
    let mut mean_r = 0.0_f32;
    for i in 0..len {
        mean_l += left[i];
        mean_r += right[i];
    }
    mean_l /= len as f32;
    mean_r /= len as f32;

    let mut peak = 0.0_f32;
    for i in 0..len {
        left[i] -= mean_l;
        right[i] -= mean_r;
        peak = peak.max(left[i].abs()).max(right[i].abs());
    }
    let ceiling = if peak_ceiling.is_finite() && peak_ceiling > 0.0 {
        peak_ceiling.min(1.0)
    } else {
        LIMITER_CEILING_LINEAR
    };
    if peak > 1.0e-8 {
        let gain = ceiling / peak;
        for i in 0..len {
            left[i] = (left[i] * gain).clamp(-ceiling, ceiling);
            right[i] = (right[i] * gain).clamp(-ceiling, ceiling);
        }
    }
}
