//! Kirchhoff-Love biharmonic plate dispersion and 2D wave-tank interference sonification mapping.

// theory_verification!

use math_explorer::pure_math::analysis::pde::wave_2d::{
    analytic_clamped_square_omega, SpectralMode2D, Wave2DMedium, Wave2DParams,
};

use crate::audio::frame::{
    clamp_freq_hz, HarmonicWeights, SonificationFrame, VoiceSpec, WaveformPreset, NUM_HARMONICS,
};

/// Maximum polyphonic spatial Fourier modes mapped per [`chladni_wave_tank_frame`] call.
pub const MAX_CHLADNI_MODES: usize = 6;

const PLATE_OMEGA_TO_HZ_SCALE: f64 = 150.0;
const ACOUSTIC_WAVE_TO_HZ_SCALE: f64 = 55.0;
const QUIESCENT_MAGNITUDE_FLOOR: f64 = 1.0e-6;

/// Maps up to six 2D Fourier [`SpectralMode2D`] peaks, Kirchhoff-Love / acoustic dispersion,
/// Young double-slit detector fringes, and mallet strike impulses into a [`SonificationFrame`].
#[must_use]
pub fn chladni_wave_tank_frame(
    modes: &[SpectralMode2D],
    params: &Wave2DParams,
    domain_length: f64,
    detector_buf: &[f64],
    mallet_impulse: f32,
) -> SonificationFrame {
    if !inputs_are_finite(modes, params, domain_length, detector_buf) {
        return SonificationFrame::silent();
    }
    let safe_len = domain_length.max(1.0e-3);
    let impulse = if mallet_impulse.is_finite() {
        mallet_impulse.clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut active_modes = [SpectralMode2D::default(); MAX_CHLADNI_MODES];
    let mut count = 0_usize;
    let mut total_mag = 0.0_f64;
    let mut max_mag = 0.0_f64;
    for &m in modes.iter().take(MAX_CHLADNI_MODES) {
        if m.magnitude > QUIESCENT_MAGNITUDE_FLOOR && m.k_radial > 0.0 {
            active_modes[count] = m;
            total_mag += m.magnitude;
            max_mag = max_mag.max(m.magnitude);
            count += 1;
        }
    }
    if count == 0 || max_mag <= QUIESCENT_MAGNITUDE_FLOOR {
        return SonificationFrame::silent();
    }

    let mut frame = SonificationFrame::silent();
    frame.waveform = match params.medium {
        Wave2DMedium::BiharmonicPlate => WaveformPreset::QuantumModal,
        Wave2DMedium::AcousticWave => WaveformPreset::FourierWarm,
    };
    frame.custom_harmonics = Some(build_modal_harmonics(
        &active_modes[..count],
        total_mag,
        params.medium,
    ));
    frame.active_voices = count as u8;
    frame.master_gain = 0.76;

    let fm = compute_frame_fm_modulation(&active_modes[..count], params, safe_len, detector_buf);
    populate_chladni_voices(
        &mut frame.voices[..count],
        &active_modes[..count],
        params,
        safe_len,
        total_mag,
        fm,
        impulse,
    );
    frame.sanitize()
}

fn compute_frame_fm_modulation(
    modes: &[SpectralMode2D],
    params: &Wave2DParams,
    safe_len: f64,
    detector_buf: &[f64],
) -> (f32, f32) {
    let (fringe_hz, fringe_index) = extract_detector_fringe_modulation(detector_buf);
    if fringe_hz > 0.0 {
        return (fringe_hz, fringe_index);
    }
    if modes.len() >= 2 {
        let f0 = modal_dispersion_hz(&modes[0], params, safe_len);
        let f1 = modal_dispersion_hz(&modes[1], params, safe_len);
        ((f0 - f1).abs().clamp(0.0, 16.0), 0.16)
    } else {
        (0.0, 0.06)
    }
}

fn populate_chladni_voices(
    out_voices: &mut [VoiceSpec],
    modes: &[SpectralMode2D],
    params: &Wave2DParams,
    safe_len: f64,
    total_mag: f64,
    fm: (f32, f32),
    impulse: f32,
) {
    let count = modes.len();
    let env = (total_mag * 24.0).clamp(0.18, 1.0) as f32;
    for (slot, (voice, mode)) in out_voices.iter_mut().zip(modes.iter()).enumerate() {
        let freq_hz = modal_dispersion_hz(mode, params, safe_len);
        let weight = (mode.magnitude / total_mag) as f32;
        let dir = (mode.kx as f64 / mode.k_radial.max(1.0e-6)).clamp(-1.0, 1.0) as f32;
        let spread = if count <= 1 {
            0.0
        } else {
            -0.55 + 1.10 * ((slot as f32) / ((count - 1) as f32))
        };
        let pan = (0.60 * dir + 0.40 * spread).clamp(-0.92, 0.92);
        let strike = (impulse * (1.0 - 0.14 * (slot as f32))).clamp(0.0, 1.0);
        *voice = VoiceSpec {
            freq_hz,
            gain: ((0.12 + 0.78 * weight) * env).clamp(0.05, 0.88),
            pan,
            fm_mod_hz: fm.0,
            fm_index: fm.1,
            filter_cutoff_hz: clamp_freq_hz(freq_hz * (2.6 + 2.4 * weight) + 550.0),
            transient_impulse: strike,
        };
    }
}

fn inputs_are_finite(
    modes: &[SpectralMode2D],
    params: &Wave2DParams,
    domain_length: f64,
    detector_buf: &[f64],
) -> bool {
    params.wave_speed.is_finite()
        && params.plate_stiffness.is_finite()
        && params.damping.is_finite()
        && domain_length.is_finite()
        && domain_length > 0.0
        && modes
            .iter()
            .all(|m| m.magnitude.is_finite() && m.k_radial.is_finite() && m.phase.is_finite())
        && detector_buf.iter().all(|v| v.is_finite())
}

fn modal_dispersion_hz(mode: &SpectralMode2D, params: &Wave2DParams, safe_len: f64) -> f32 {
    let acoustic_hz =
        ACOUSTIC_WAVE_TO_HZ_SCALE * params.wave_speed.abs() * mode.k_radial / safe_len;
    let raw_hz = match params.medium {
        Wave2DMedium::AcousticWave => acoustic_hz,
        Wave2DMedium::BiharmonicPlate => {
            let stiffness = params.plate_stiffness.max(0.0);
            let mode_m = mode.kx.unsigned_abs().max(1);
            let mode_n = mode.ky.unsigned_abs().max(1);
            let denom = (mode_m * mode_m + mode_n * mode_n) as f64;
            let axial_scale = (mode.k_radial * mode.k_radial) / denom.max(1.0);
            let omega = analytic_clamped_square_omega(mode_m, mode_n, safe_len, stiffness);
            let plate_hz = PLATE_OMEGA_TO_HZ_SCALE * omega * axial_scale;
            (plate_hz * plate_hz + acoustic_hz * acoustic_hz).sqrt()
        }
    };
    clamp_freq_hz(raw_hz as f32)
}

fn build_modal_harmonics(
    modes: &[SpectralMode2D],
    total_mag: f64,
    medium: Wave2DMedium,
) -> HarmonicWeights {
    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    for (idx, mode) in modes.iter().take(NUM_HARMONICS).enumerate() {
        let w = (mode.magnitude / total_mag.max(1.0e-12)) as f32;
        cos_a[idx] = w * (mode.phase.cos() as f32);
        sin_b[idx] = w * (mode.phase.sin() as f32);
    }
    if medium == Wave2DMedium::BiharmonicPlate && !modes.is_empty() {
        let p0 = modes[0].phase;
        cos_a[6] += 0.14 * ((2.0 * p0).cos() as f32);
        sin_b[6] += 0.14 * ((2.0 * p0).sin() as f32);
        cos_a[7] += 0.09 * ((3.0 * p0).cos() as f32);
        sin_b[7] += 0.09 * ((3.0 * p0).sin() as f32);
    }
    HarmonicWeights { cos_a, sin_b }.normalized()
}

fn extract_detector_fringe_modulation(detector_buf: &[f64]) -> (f32, f32) {
    if detector_buf.len() < 5 {
        return (0.0, 0.0);
    }
    let max_i = detector_buf.iter().copied().fold(0.0_f64, f64::max);
    if max_i <= 1.0e-9 {
        return (0.0, 0.0);
    }
    let threshold = 0.18 * max_i;
    let mut peak_count = 0_usize;
    let mut first_peak = 0_usize;
    let mut last_peak = 0_usize;
    let mut min_between = max_i;

    for y in 1..(detector_buf.len() - 1) {
        let v = detector_buf[y];
        if peak_count > 0 && v < min_between {
            min_between = v;
        }
        let is_peak = v >= threshold && v >= detector_buf[y - 1] && v > detector_buf[y + 1];
        if is_peak && (peak_count == 0 || y.saturating_sub(last_peak) >= 3) {
            if peak_count == 0 {
                first_peak = y;
            }
            last_peak = y;
            peak_count += 1;
        }
    }
    if peak_count < 2 || last_peak <= first_peak {
        return (0.0, 0.0);
    }
    let avg_spacing = ((last_peak - first_peak) as f64) / ((peak_count - 1) as f64);
    let fringe_density = (detector_buf.len() as f64) / avg_spacing.max(2.0);
    let contrast = ((max_i - min_between) / (max_i + min_between.max(0.0))).clamp(0.0, 1.0) as f32;
    if contrast < 0.08 {
        return (0.0, 0.0);
    }
    let beat_hz = (2.2 * fringe_density).clamp(1.5, 80.0) as f32;
    let fm_index = (0.35 + 1.65 * contrast).clamp(0.35, 2.2);
    (beat_hz, fm_index)
}
