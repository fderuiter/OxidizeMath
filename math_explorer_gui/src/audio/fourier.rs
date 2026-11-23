//! Continuous Fourier series harmonic table generation for custom `PeriodicWave` timbres.

// theory_verification!

use super::frame::{HarmonicWeights, WaveformPreset, NUM_HARMONICS};

/// Precomputed bank of stack-resident [`HarmonicWeights`] indexed by [`WaveformPreset::index`].
#[derive(Clone, Copy, Debug)]
pub struct PresetHarmonicBank {
    tables: [HarmonicWeights; WaveformPreset::ALL.len()],
}

impl Default for PresetHarmonicBank {
    fn default() -> Self {
        Self::new()
    }
}

impl PresetHarmonicBank {
    /// Integrates continuous periodic waveshapes via `calculate_fourier_coefficients_real`
    /// during initialization (prior to `VerifiedAllocator::lock_allocations`).
    #[must_use]
    pub fn new() -> Self {
        let mut tables = [HarmonicWeights::pure_sine(); WaveformPreset::ALL.len()];
        for preset in WaveformPreset::ALL {
            tables[preset.index()] = compute_preset_harmonics(preset);
        }
        Self { tables }
    }

    /// Looks up the precomputed [`HarmonicWeights`] for `preset` in `O(1)` with zero allocations.
    #[inline]
    #[must_use]
    pub const fn get(&self, preset: WaveformPreset) -> HarmonicWeights {
        self.tables[preset.index()]
    }
}

/// Computes normalized real Fourier coefficients `(a_n, b_n)` for `preset`.
#[must_use]
pub fn compute_preset_harmonics(preset: WaveformPreset) -> HarmonicWeights {
    match preset {
        WaveformPreset::Sine => HarmonicWeights::pure_sine(),
        WaveformPreset::FourierWarm => integrate_waveform(wave_fourier_warm),
        WaveformPreset::PatchClamp => integrate_waveform(wave_patch_clamp),
        WaveformPreset::QuantumModal => integrate_waveform(wave_quantum_modal),
        WaveformPreset::SubharmonicSquare => integrate_waveform(wave_subharmonic_square),
        WaveformPreset::JustHarmonic => integrate_waveform(wave_just_harmonic),
    }
}

fn wave_fourier_warm(x: f64) -> f64 {
    x.sin() + 0.50 * (2.0 * x).sin() + 0.28 * (3.0 * x).sin() + 0.15 * (4.0 * x).cos()
}

fn wave_patch_clamp(x: f64) -> f64 {
    let depol = (-3.2 * (1.0 - x.cos())).exp();
    let hyperpol = -0.35 * (x - 0.8).sin();
    depol + hyperpol
}

fn wave_quantum_modal(x: f64) -> f64 {
    0.85 * x.sin() + 0.42 * (2.0 * x).cos() + 0.30 * (3.0 * x).sin() + 0.18 * (5.0 * x).cos()
}

fn wave_subharmonic_square(x: f64) -> f64 {
    (2.6 * x.sin()).tanh()
}

fn wave_just_harmonic(x: f64) -> f64 {
    x.sin()
        + 0.55 * (2.0 * x).sin()
        + 0.38 * (3.0 * x).sin()
        + 0.25 * (4.0 * x).sin()
        + 0.18 * (5.0 * x).sin()
        + 0.12 * (6.0 * x).sin()
}

#[cfg(feature = "pure_math")]
fn integrate_waveform(wave_fn: fn(f64) -> f64) -> HarmonicWeights {
    use math_explorer::pure_math::analysis::integration::ClenshawCurtis;
    use math_explorer::pure_math::analysis::transforms::fourier::continuous::calculate_fourier_coefficients_real;

    let integrator = ClenshawCurtis;
    let (_a0, an, bn) = calculate_fourier_coefficients_real(
        wave_fn,
        std::f64::consts::PI,
        NUM_HARMONICS,
        &integrator,
    );
    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    for i in 0..NUM_HARMONICS {
        cos_a[i] = an.get(i).copied().unwrap_or(0.0) as f32;
        sin_b[i] = bn.get(i).copied().unwrap_or(0.0) as f32;
    }
    HarmonicWeights { cos_a, sin_b }.normalized()
}

#[cfg(not(feature = "pure_math"))]
fn integrate_waveform(wave_fn: fn(f64) -> f64) -> HarmonicWeights {
    const SAMPLES: usize = 64;
    let inv_half = 2.0 / (SAMPLES as f64);
    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    for s in 0..SAMPLES {
        let theta = -std::f64::consts::PI
            + (2.0 * std::f64::consts::PI * ((s as f64) + 0.5)) / (SAMPLES as f64);
        let y = wave_fn(theta);
        for n in 0..NUM_HARMONICS {
            let k = (n + 1) as f64;
            cos_a[n] += (y * (k * theta).cos() * inv_half) as f32;
            sin_b[n] += (y * (k * theta).sin() * inv_half) as f32;
        }
    }
    HarmonicWeights { cos_a, sin_b }.normalized()
}
