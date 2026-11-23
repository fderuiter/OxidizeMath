//! Surface morphogenesis sonification mapping morphogen spatial variance and mean-curvature buckling shifts.

// theory_verification!

use crate::audio::frame::{SonificationFrame, VoiceSpec, WaveformPreset, clamp_freq_hz};

/// Maps surface morphogen spatial variance, activator mean, mean-curvature shift, and buckling RMS
/// into a zero-allocation stereo polyphonic [`SonificationFrame`].
#[must_use]
pub fn surface_morphogenesis_frame(
    morphogen_variance: f64,
    activator_mean: f64,
    mean_curvature_shift: f64,
    buckling_rms: f64,
) -> SonificationFrame {
    if !morphogen_variance.is_finite()
        || !activator_mean.is_finite()
        || !mean_curvature_shift.is_finite()
        || !buckling_rms.is_finite()
    {
        return SonificationFrame::silent();
    }

    let var_norm = (morphogen_variance.max(0.0).sqrt() * 3.2).clamp(0.0, 1.0) as f32;
    let act_norm = (activator_mean.abs() * 0.55).clamp(0.05, 1.0) as f32;
    let curv_norm = (mean_curvature_shift.abs() * 1.8).clamp(0.0, 1.0) as f32;
    let buckle_norm = (buckling_rms.abs() * 4.0).clamp(0.0, 1.0) as f32;

    let base_hz = 165.0 + 275.0 * var_norm + 110.0 * curv_norm;
    let fifth_hz = base_hz * (1.5 + 0.08 * buckle_norm);
    let modal_hz = base_hz * (2.0 + 0.25 * curv_norm);

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::QuantumModal;
    frame.active_voices = 3;
    frame.master_gain = (0.45 + 0.35 * (var_norm + buckle_norm).min(1.0)).clamp(0.25, 0.85);

    frame.voices[0] = VoiceSpec {
        freq_hz: clamp_freq_hz(base_hz),
        gain: (0.28 + 0.45 * act_norm + 0.15 * var_norm).clamp(0.15, 0.85),
        pan: -0.22,
        fm_mod_hz: 36.0 + 140.0 * var_norm,
        fm_index: 0.25 + 2.4 * var_norm,
        filter_cutoff_hz: clamp_freq_hz(620.0 + 3400.0 * (0.2 + 0.8 * var_norm)),
        transient_impulse: (0.35 * curv_norm * buckle_norm).clamp(0.0, 0.6),
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(fifth_hz),
        gain: (0.16 + 0.42 * curv_norm).clamp(0.08, 0.65),
        pan: 0.22,
        fm_mod_hz: 54.0 + 180.0 * curv_norm,
        fm_index: 0.2 + 2.1 * curv_norm,
        filter_cutoff_hz: clamp_freq_hz(800.0 + 3800.0 * curv_norm),
        transient_impulse: 0.0,
    };
    frame.voices[2] = VoiceSpec {
        freq_hz: clamp_freq_hz(modal_hz),
        gain: (0.10 + 0.38 * buckle_norm).clamp(0.05, 0.55),
        pan: 0.0,
        fm_mod_hz: 82.0 + 120.0 * buckle_norm,
        fm_index: 0.15 + 1.6 * buckle_norm,
        filter_cutoff_hz: clamp_freq_hz(950.0 + 3200.0 * buckle_norm),
        transient_impulse: 0.0,
    };
    frame.sanitize()
}
