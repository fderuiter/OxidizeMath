//! Zero-allocation spatial Fourier harmonic projection for [`crate::tabs::quantum::wave_sim::WaveRunner`]
//! and [`crate::tabs::quantum::wave_sim::WaveData`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::{WaveData, WaveRunner};
use crate::audio::{SonificationFrame, MAX_HARMONIC_SLOTS};
use core::f64::consts::PI;

struct WaveSpectralMoments {
    total_prob: f64,
    mean_x: f64,
    mean_x2: f64,
    mean_v: f64,
    real_acc: [f64; MAX_HARMONIC_SLOTS],
    imag_acc: [f64; MAX_HARMONIC_SLOTS],
}

fn normalize_periodic_harmonics(
    real_acc: &[f64; MAX_HARMONIC_SLOTS],
    imag_acc: &[f64; MAX_HARMONIC_SLOTS],
) -> ([f32; MAX_HARMONIC_SLOTS], [f32; MAX_HARMONIC_SLOTS]) {
    let mut peak_mag = 1e-6_f64;
    for k in 1..MAX_HARMONIC_SLOTS {
        peak_mag = peak_mag.max(real_acc[k].hypot(imag_acc[k]));
    }
    let mut harmonics_real = [0.0_f32; MAX_HARMONIC_SLOTS];
    let mut harmonics_imag = [0.0_f32; MAX_HARMONIC_SLOTS];
    for k in 1..MAX_HARMONIC_SLOTS {
        harmonics_real[k] = ((real_acc[k] / peak_mag) as f32).clamp(-1.0, 1.0);
        harmonics_imag[k] = ((imag_acc[k] / peak_mag) as f32).clamp(-1.0, 1.0);
    }
    if harmonics_imag[1].abs() < 0.25 {
        harmonics_imag[1] = 0.25;
    }
    (harmonics_real, harmonics_imag)
}

fn moments_to_frame(m: WaveSpectralMoments, span: f64) -> SonificationFrame {
    let inv_prob = 1.0 / m.total_prob;
    let exp_x = m.mean_x * inv_prob;
    let spread = (m.mean_x2 * inv_prob - exp_x * exp_x).max(0.0).sqrt() as f32;
    let exp_v = (m.mean_v * inv_prob).clamp(0.0, 6.0) as f32;
    let (harmonics_real, harmonics_imag) = normalize_periodic_harmonics(&m.real_acc, &m.imag_acc);

    SonificationFrame {
        active: true,
        carrier_hz: (165.0 + 48.0 * exp_v).clamp(110.0, 880.0),
        amplitude: (0.42 + 0.35 / (0.6 + spread)).clamp(0.25, 0.88),
        pan: (exp_x / (span * 0.5).max(0.5)).clamp(-1.0, 1.0) as f32,
        filter_cutoff_hz: (850.0 + 3_200.0 / (0.35 + spread)).clamp(450.0, 6_500.0),
        filter_q: 1.5,
        harmonics_real,
        harmonics_imag,
        use_periodic_wave: true,
        ..SonificationFrame::silent()
    }
}

/// Projects `|ψ(x, t)|²` in a [`WaveData`] snapshot onto `PeriodicWave` harmonics with zero heap allocations.
#[must_use]
pub(super) fn sonification_from_wave_data(data: &WaveData) -> SonificationFrame {
    let probs = data.prob_density.as_slice();
    let xs = data.x_axis.as_slice();
    if probs.len() < 2 || xs.len() != probs.len() {
        return SonificationFrame::silent();
    }
    let x_min = xs[0];
    let span = xs[xs.len() - 1] - x_min;
    if !span.is_finite() || span <= 0.0 {
        return SonificationFrame::silent();
    }
    let dx = span / (probs.len() as f64 - 1.0);
    let mut m = WaveSpectralMoments {
        total_prob: 0.0,
        mean_x: 0.0,
        mean_x2: 0.0,
        mean_v: 0.0,
        real_acc: [0.0; MAX_HARMONIC_SLOTS],
        imag_acc: [0.0; MAX_HARMONIC_SLOTS],
    };
    for (i, &p) in probs.iter().enumerate() {
        let x = xs[i];
        if !p.is_finite() || !x.is_finite() {
            return SonificationFrame::silent();
        }
        let w = p * dx;
        m.total_prob += w;
        m.mean_x += x * w;
        m.mean_x2 += x * x * w;
        if i < data.potential.len() && data.potential[i].is_finite() {
            m.mean_v += data.potential[i] * w;
        }
        let u = ((x - x_min) / span).clamp(0.0, 1.0);
        for k in 1..MAX_HARMONIC_SLOTS {
            let angle = (k as f64) * PI * u;
            m.real_acc[k] += w * (2.0 * angle).cos();
            m.imag_acc[k] += w * angle.sin();
        }
    }
    if !m.total_prob.is_finite() || m.total_prob <= 1e-9 {
        return SonificationFrame::silent();
    }
    moments_to_frame(m, span)
}

impl WaveRunner {
    fn accumulate_spectral_moments(&self, span: f64) -> Option<WaveSpectralMoments> {
        let psi = self.psi.vector.as_slice();
        let xs = self.x_axis.as_slice();
        let dx = span / (psi.len() as f64 - 1.0);
        let mut moments = WaveSpectralMoments {
            total_prob: 0.0,
            mean_x: 0.0,
            mean_x2: 0.0,
            mean_v: 0.0,
            real_acc: [0.0; MAX_HARMONIC_SLOTS],
            imag_acc: [0.0; MAX_HARMONIC_SLOTS],
        };

        for (i, c) in psi.iter().enumerate() {
            if !c.re.is_finite() || !c.im.is_finite() {
                return None;
            }
            let p = c.norm_sqr();
            let x = xs[i];
            if !p.is_finite() || !x.is_finite() {
                return None;
            }
            let w = p * dx;
            moments.total_prob += w;
            moments.mean_x += x * w;
            moments.mean_x2 += x * x * w;
            if i < self.potential.len() && self.potential[i].is_finite() {
                moments.mean_v += self.potential[i] * w;
            }
            let u = ((x - self.x_min) / span).clamp(0.0, 1.0);
            for k in 1..MAX_HARMONIC_SLOTS {
                let angle = (k as f64) * PI * u;
                moments.real_acc[k] += w * (2.0 * angle).cos();
                moments.imag_acc[k] += w * angle.sin();
            }
        }
        (moments.total_prob.is_finite() && moments.total_prob > 1e-9).then_some(moments)
    }

    /// Projects `|ψ(x, t)|²` onto `PeriodicWave` spatial standing-wave harmonics with zero heap allocations.
    #[must_use]
    pub fn compute_sonification_frame(&self) -> SonificationFrame {
        let psi = self.psi.vector.as_slice();
        let span = self.x_max - self.x_min;
        if self.diverged
            || psi.len() < 2
            || self.x_axis.len() != psi.len()
            || !span.is_finite()
            || span <= 0.0
        {
            return SonificationFrame::silent();
        }
        let Some(m) = self.accumulate_spectral_moments(span) else {
            return SonificationFrame::silent();
        };
        moments_to_frame(m, span)
    }
}
