//! Neuroscience sonification mappings: Hodgkin-Huxley, Spike Analysis, and 5-Neuron Ring.

// theory_verification!

use std::f64::consts::TAU;

use crate::audio::frame::{clamp_freq_hz, SonificationFrame, VoiceSpec, WaveformPreset};

/// Five-note just-intonation major pentatonic frequencies (`1:1, 9:8, 5:4, 3:2, 5:3` over `220 Hz`).
pub const RING_PENTATONIC_HZ: [f32; 5] = [220.0, 247.5, 275.0, 330.0, 366.666_7];

/// Maps Hodgkin-Huxley membrane voltage (`mV`) and gating variables (`m, h, n`) to FM pitch and spike clicks.
#[must_use]
pub fn hodgkin_huxley_frame(
    v_mv: f64,
    prev_v_mv: f64,
    m: f64,
    h: f64,
    n: f64,
    i_ext: f64,
) -> SonificationFrame {
    if !v_mv.is_finite() {
        return SonificationFrame::silent();
    }
    let norm_v = ((v_mv + 80.0) / 120.0).clamp(0.0, 1.0) as f32;
    let spiked = prev_v_mv.is_finite() && prev_v_mv <= 0.0 && v_mv > 0.0;
    let na_open = (m.clamp(0.0, 1.0).powi(3) * h.clamp(0.0, 1.0)) as f32;
    let k_open = n.clamp(0.0, 1.0).powi(4) as f32;
    let drive = (i_ext.abs() / 30.0).clamp(0.05, 1.0) as f32;

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::PatchClamp;
    frame.active_voices = 2;
    frame.master_gain = 0.75;
    frame.voices[0] = VoiceSpec {
        freq_hz: clamp_freq_hz(130.0 + norm_v * 520.0),
        gain: (0.20 + 0.65 * norm_v + 0.15 * drive).clamp(0.0, 0.95),
        pan: -0.15,
        fm_mod_hz: 55.0 + 210.0 * na_open,
        fm_index: 0.35 + 3.8 * norm_v,
        filter_cutoff_hz: clamp_freq_hz(550.0 + 4800.0 * (0.25 + 0.75 * norm_v)),
        transient_impulse: if spiked { 1.0 } else { 0.0 },
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz((130.0 + norm_v * 520.0) * 1.5),
        gain: (0.45 * k_open).clamp(0.0, 0.55),
        pan: 0.25,
        fm_mod_hz: 90.0 + 120.0 * k_open,
        fm_index: 0.2 + 1.5 * k_open,
        filter_cutoff_hz: clamp_freq_hz(700.0 + 3000.0 * k_open),
        transient_impulse: if spiked { 0.5 } else { 0.0 },
    };
    frame.sanitize()
}

/// Maps Spike Analysis membrane voltage, spike threshold crossings, and instantaneous `Hz` rate.
#[must_use]
pub fn spike_analysis_frame(
    v_mv: f64,
    prev_v_mv: f64,
    threshold_mv: f64,
    firing_rate_hz: f64,
) -> SonificationFrame {
    if !v_mv.is_finite() {
        return SonificationFrame::silent();
    }
    let norm_v = ((v_mv + 80.0) / 120.0).clamp(0.0, 1.0) as f32;
    let spiked = prev_v_mv.is_finite() && prev_v_mv <= threshold_mv && v_mv > threshold_mv;
    let rate = if firing_rate_hz.is_finite() {
        firing_rate_hz.clamp(0.0, 250.0) as f32
    } else {
        0.0
    };

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::PatchClamp;
    frame.active_voices = 2;
    frame.master_gain = 0.75;
    let base_hz = clamp_freq_hz(140.0 + norm_v * 500.0);
    frame.voices[0] = VoiceSpec {
        freq_hz: base_hz,
        gain: (0.22 + 0.65 * norm_v).clamp(0.0, 0.92),
        pan: 0.0,
        fm_mod_hz: (rate * 2.5).clamp(8.0, 600.0),
        fm_index: 0.30 + 3.2 * norm_v + (rate / 120.0).min(1.5),
        filter_cutoff_hz: clamp_freq_hz(650.0 + 4200.0 * norm_v + rate * 8.0),
        transient_impulse: if spiked { 1.0 } else { 0.0 },
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(220.0 + rate * 3.0),
        gain: (rate / 160.0).clamp(0.0, 0.45),
        pan: 0.2,
        fm_mod_hz: rate.clamp(4.0, 250.0),
        fm_index: 0.8,
        filter_cutoff_hz: clamp_freq_hz(1200.0 + rate * 12.0),
        transient_impulse: 0.0,
    };
    frame.sanitize()
}

/// Maps 5 coupled ring neurons to 5 pentatonic voices panned across stereo by ring angle.
#[must_use]
pub fn neural_ring_frame(
    voltages: &[f64; 5],
    prev_voltages: &[f64; 5],
    threshold_mv: f64,
) -> SonificationFrame {
    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::FourierWarm;
    frame.active_voices = 5;
    frame.master_gain = 0.72;

    for i in 0..5 {
        let v = voltages[i];
        if !v.is_finite() {
            continue;
        }
        let norm_v = ((v + 80.0) / 110.0).clamp(0.0, 1.0) as f32;
        let spiked =
            prev_voltages[i].is_finite() && prev_voltages[i] <= threshold_mv && v > threshold_mv;
        let angle = (i as f64) * TAU / 5.0;
        let pan = (angle.cos() as f32).clamp(-1.0, 1.0);
        let pitch_bend = 1.0 + 0.04 * (norm_v - 0.15);

        frame.voices[i] = VoiceSpec {
            freq_hz: clamp_freq_hz(RING_PENTATONIC_HZ[i] * pitch_bend),
            gain: (0.10 + 0.48 * norm_v + if spiked { 0.25 } else { 0.0 }).clamp(0.0, 0.85),
            pan,
            fm_mod_hz: RING_PENTATONIC_HZ[i] * 0.5,
            fm_index: 0.15 + 1.8 * norm_v,
            filter_cutoff_hz: clamp_freq_hz(900.0 + 3600.0 * norm_v),
            transient_impulse: if spiked { 0.95 } else { 0.0 },
        };
    }
    frame.sanitize()
}
