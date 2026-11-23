//! Acoustic luthiery and 2D wave-tank modal sonification mapping.
//!
//! Maps top-8 2D FFT spatial-frequency peaks ([`SpectralPeak2D`]), Kirchhoff-Love thin-plate
//! eigenfrequency dispersion ([`analytic_clamped_square_omega`]), and signed point-probe
//! phase velocity ([`WaveProbeSample`]) into a stack-resident 8-voice [`SonificationFrame`]
//! with zero heap allocations.

// theory_verification!

use std::f64::consts::TAU;

use math_explorer::pure_math::analysis::pde::wave_2d::{
    analytic_clamped_square_omega, SpectralPeak2D, Wave2DMedium, WaveProbeSample,
};

use crate::audio::frame::{
    clamp_freq_hz, HarmonicWeights, SonificationFrame, VoiceSpec, WaveformPreset, MAX_VOICES,
    NUM_HARMONICS,
};

/// Luthiery concert-pitch scale factor mapping `D = 4e-4` clamped `(1, 1)` plates near `A3 = 220 Hz`.
const PLATE_CONCERT_SCALE_HZ: f64 = 1_920.0;

/// Synthesizes an 8-voice [`SonificationFrame`] from 2D FFT spatial peaks and a signed pickup probe.
///
/// # Example
///
/// ```rust,ignore
/// use math_explorer::pure_math::analysis::pde::wave_2d::{
///     SpectralPeak2D, Wave2DMedium, WaveProbeSample,
/// };
/// use crate::audio::mappings::chladni_plate_frame;
///
/// let peaks = [
///     SpectralPeak2D { kx: 2, ky: 1, k_radial: 2.236, magnitude: 0.42 },
///     SpectralPeak2D { kx: 3, ky: 2, k_radial: 3.606, magnitude: 0.21 },
/// ];
/// let probe = WaveProbeSample { displacement: 0.18, velocity: -0.65 };
/// let frame = chladni_plate_frame(
///     &peaks,
///     probe,
///     Wave2DMedium::BiharmonicPlate,
///     1.2,
///     0.0,
///     4.0e-4,
///     0.65,
/// );
/// assert_eq!(frame.active_voices, 2);
/// assert!(frame.is_audible());
/// ```
#[must_use]
pub fn chladni_plate_frame(
    peaks: &[SpectralPeak2D],
    probe: WaveProbeSample,
    medium: Wave2DMedium,
    driver_freq_hz: f64,
    wave_speed: f64,
    plate_stiffness: f64,
    pickup_x_norm: f64,
) -> SonificationFrame {
    let disp = finite_or(probe.displacement, 0.0).clamp(-10.0, 10.0);
    let vel = finite_or(probe.velocity, 0.0).clamp(-80.0, 80.0);
    let pickup_pan = (2.0 * finite_or(pickup_x_norm, 0.5).clamp(0.0, 1.0) - 1.0) as f32;

    let mut active = 0_usize;
    let mut total_mag = 0.0_f64;
    for peak in peaks.iter().take(MAX_VOICES) {
        if peak.magnitude.is_finite() && peak.magnitude > 1.0e-7 && peak.k_radial.is_finite() {
            total_mag += peak.magnitude;
            active += 1;
        } else {
            break;
        }
    }

    let probe_energy = disp.abs() + 0.08 * vel.abs();
    if active == 0 && probe_energy <= 1.0e-6 {
        return SonificationFrame::silent();
    }

    let root_hz = compute_audible_root_hz(medium, driver_freq_hz, wave_speed, plate_stiffness);
    let phase_angle = vel.atan2(disp);
    let phase_detune_hz = ((vel / (1.0 + vel.abs())) * 14.0) as f32;
    let harmonics = build_probe_harmonics(&peaks[..active], total_mag, phase_angle);

    let mut frame = SonificationFrame::silent();
    frame.waveform = match medium {
        Wave2DMedium::BiharmonicPlate => WaveformPreset::QuantumModal,
        Wave2DMedium::AcousticWave => WaveformPreset::FourierWarm,
    };
    frame.custom_harmonics = Some(harmonics);
    frame.master_gain = (0.58 + 0.24 * (probe_energy.min(1.5) as f32)).clamp(0.55, 0.85);

    if active == 0 {
        frame.active_voices = 1;
        frame.voices[0] = impulse_pickup_voice(root_hz, probe_energy, phase_detune_hz, pickup_pan);
        return frame.sanitize();
    }

    let ref_k = peaks[0].k_radial.max(1.0);
    let d_eff = finite_or(plate_stiffness, 4.0e-4).clamp(1.0e-9, 1.0);
    let omega_11 = analytic_clamped_square_omega(1, 1, 1.0, d_eff).max(1.0e-6);

    for (slot, &peak) in peaks.iter().take(active).enumerate() {
        let ratio = modal_frequency_ratio(peak, medium, ref_k, d_eff, omega_11);
        let weight = (peak.magnitude / total_mag.max(1.0e-9)).clamp(0.04, 1.0) as f32;
        frame.voices[slot] = build_modal_voice(
            slot,
            peak,
            root_hz * ratio,
            weight,
            pickup_pan,
            phase_detune_hz,
            vel,
        );
    }
    frame.active_voices = active as u8;
    frame.sanitize()
}

fn build_modal_voice(
    slot: usize,
    peak: SpectralPeak2D,
    base_hz: f64,
    weight: f32,
    pickup_pan: f32,
    phase_detune_hz: f32,
    vel: f64,
) -> VoiceSpec {
    let mode_pan = (peak.kx as f64 / peak.k_radial.max(1.0)).clamp(-1.0, 1.0) as f32;
    let pan = (0.55 * pickup_pan + 0.45 * mode_pan).clamp(-1.0, 1.0);
    let detune = phase_detune_hz / ((slot + 1) as f32).sqrt();
    let freq_hz = clamp_freq_hz(base_hz as f32 + detune);
    let vel_norm = (vel.abs() / (vel.abs() + 4.0)) as f32;
    VoiceSpec {
        freq_hz,
        gain: (0.12 + 0.76 * weight).clamp(0.08, 0.86),
        pan,
        fm_mod_hz: clamp_freq_hz(freq_hz * 0.5 + 6.0 * vel_norm),
        fm_index: (0.18 + 1.45 * vel_norm * weight).clamp(0.10, 2.8),
        filter_cutoff_hz: clamp_freq_hz(950.0 + 3800.0 * (0.35 * weight + 0.65 * vel_norm)),
        transient_impulse: if slot == 0 && vel.abs() > 6.0 {
            0.35
        } else {
            0.0
        },
    }
}

#[inline]
fn finite_or(val: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val
    } else {
        fallback
    }
}

fn compute_audible_root_hz(
    medium: Wave2DMedium,
    driver_freq_hz: f64,
    wave_speed: f64,
    plate_stiffness: f64,
) -> f64 {
    let drv = finite_or(driver_freq_hz, 1.0).clamp(0.02, 40.0);
    match medium {
        Wave2DMedium::BiharmonicPlate => {
            let d_eff = finite_or(plate_stiffness, 4.0e-4).clamp(1.0e-9, 0.05);
            let f11 = analytic_clamped_square_omega(1, 1, 1.0, d_eff) / TAU;
            (f11 * PLATE_CONCERT_SCALE_HZ + 5.5 * drv).clamp(65.0, 1_320.0)
        }
        Wave2DMedium::AcousticWave => {
            let c = finite_or(wave_speed, 1.0).clamp(0.05, 5.0);
            (110.0 + 58.0 * c + 9.0 * drv).clamp(65.0, 1_320.0)
        }
    }
}

fn modal_frequency_ratio(
    peak: SpectralPeak2D,
    medium: Wave2DMedium,
    ref_k: f64,
    d_eff: f64,
    omega_11: f64,
) -> f64 {
    match medium {
        Wave2DMedium::BiharmonicPlate => {
            let m = peak.kx.unsigned_abs().clamp(1, 12);
            let n = peak.ky.unsigned_abs().clamp(1, 12);
            let omega_mn = analytic_clamped_square_omega(m, n, 1.0, d_eff);
            (omega_mn / omega_11).sqrt().clamp(0.5, 12.0)
        }
        Wave2DMedium::AcousticWave => (peak.k_radial / ref_k).clamp(0.5, 10.0),
    }
}

fn build_probe_harmonics(
    peaks: &[SpectralPeak2D],
    total_mag: f64,
    phase_angle: f64,
) -> HarmonicWeights {
    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    let denom = total_mag.max(1.0e-9);
    for idx in 0..NUM_HARMONICS {
        let modal_w = peaks.get(idx).map_or(0.12 / ((idx + 1) as f64), |p| {
            (p.magnitude / denom).max(0.05)
        });
        let harmonic_phase = ((idx + 1) as f64) * phase_angle;
        cos_a[idx] = (modal_w * harmonic_phase.cos()) as f32;
        sin_b[idx] = (modal_w * harmonic_phase.sin()) as f32;
    }
    HarmonicWeights { cos_a, sin_b }.normalized()
}

fn impulse_pickup_voice(root_hz: f64, probe_energy: f64, detune_hz: f32, pan: f32) -> VoiceSpec {
    let amp = (probe_energy * 2.2).clamp(0.10, 0.80) as f32;
    VoiceSpec {
        freq_hz: clamp_freq_hz(root_hz as f32 + detune_hz),
        gain: amp,
        pan,
        fm_mod_hz: clamp_freq_hz((root_hz * 0.5) as f32),
        fm_index: 0.35 + 1.2 * amp,
        filter_cutoff_hz: clamp_freq_hz(1200.0 + 2600.0 * amp),
        transient_impulse: (amp * 0.5).clamp(0.0, 0.6),
    }
}
