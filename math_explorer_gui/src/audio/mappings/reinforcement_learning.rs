//! Reinforcement learning temporal-difference surprise, Bellman value, and wind-drift sonification.

// theory_verification!

use crate::audio::frame::{SonificationFrame, VoiceSpec, WaveformPreset, clamp_freq_hz};

/// Synthesizes a zero-allocation [`SonificationFrame`] from reinforcement learning step telemetry.
///
/// - `state_value`: Current Bellman or empirical state value $V(s_t)$, mapped monotonically to
///   fundamental carrier pitch (`140..=660 Hz`, clamped inside `20..=8000 Hz`).
/// - `td_error`: Temporal-difference error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, whose
///   magnitude $|\delta_t|$ modulates FM roughness (`fm_index` and inharmonic modulator ratio) and
///   low-pass filter cutoff brightness.
/// - `reward`: Immediate transition reward $r_t$, triggering consonant harmonic chimes on positive
///   goal capture (`+10`) and dissonant transient clusters on trap hazards (`-10`).
/// - `wind_pan`: Stochastic crosswind drift stereo pan in `[-1.0, 1.0]`.
#[must_use]
pub fn rl_td_surprise_frame(
    state_value: f64,
    td_error: f64,
    reward: f64,
    wind_pan: f32,
) -> SonificationFrame {
    if !state_value.is_finite() || !td_error.is_finite() || !reward.is_finite() {
        return SonificationFrame::silent();
    }

    let v_norm = ((state_value + 10.0) / 20.0).clamp(0.0, 1.0) as f32;
    let surprise = (td_error.abs() / 10.0).clamp(0.0, 1.0) as f32;
    let pan = if wind_pan.is_finite() {
        wind_pan.clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let carrier_hz = clamp_freq_hz(140.0 + v_norm * 520.0);
    let fm_ratio = 1.0 + std::f32::consts::SQRT_2 * surprise;
    let fm_mod_hz = clamp_freq_hz(carrier_hz * fm_ratio);
    let fm_index = (0.15 + 4.85 * surprise).clamp(0.15, 5.0);
    let cutoff_hz = clamp_freq_hz(700.0 + 4800.0 * surprise + 900.0 * v_norm);

    let mut frame = SonificationFrame::silent();
    frame.master_gain = 0.72;

    if reward >= 5.0 {
        populate_goal_voices(&mut frame, carrier_hz, pan, surprise, v_norm);
    } else if reward <= -5.0 {
        populate_trap_voices(&mut frame, carrier_hz, pan, surprise, fm_mod_hz, cutoff_hz);
    } else {
        populate_step_voices(
            &mut frame, carrier_hz, pan, surprise, v_norm, fm_mod_hz, fm_index, cutoff_hz,
        );
    }

    frame.sanitize()
}

fn populate_goal_voices(
    frame: &mut SonificationFrame,
    carrier_hz: f32,
    pan: f32,
    surprise: f32,
    v_norm: f32,
) {
    frame.waveform = WaveformPreset::JustHarmonic;
    frame.active_voices = 3;
    frame.voices[0] = VoiceSpec {
        freq_hz: carrier_hz,
        gain: 0.62,
        pan,
        fm_mod_hz: clamp_freq_hz(carrier_hz * 2.0),
        fm_index: (0.25 + 0.8 * surprise).clamp(0.2, 1.8),
        filter_cutoff_hz: clamp_freq_hz(3600.0 + 2400.0 * v_norm),
        transient_impulse: 0.55,
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * 1.5),
        gain: 0.44,
        pan: (pan - 0.25).clamp(-1.0, 1.0),
        fm_mod_hz: carrier_hz,
        fm_index: 0.2,
        filter_cutoff_hz: 4400.0,
        transient_impulse: 0.25,
    };
    frame.voices[2] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * 2.0),
        gain: 0.34,
        pan: (pan + 0.25).clamp(-1.0, 1.0),
        fm_mod_hz: carrier_hz,
        fm_index: 0.15,
        filter_cutoff_hz: 5200.0,
        transient_impulse: 0.15,
    };
}

fn populate_trap_voices(
    frame: &mut SonificationFrame,
    carrier_hz: f32,
    pan: f32,
    surprise: f32,
    fm_mod_hz: f32,
    cutoff_hz: f32,
) {
    let sqrt2 = std::f32::consts::SQRT_2;
    frame.waveform = WaveformPreset::SubharmonicSquare;
    frame.active_voices = 3;
    frame.noise_gain = 0.24;
    frame.noise_cutoff_hz = clamp_freq_hz(1800.0 + 2400.0 * surprise);
    frame.voices[0] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * 0.75),
        gain: 0.64,
        pan,
        fm_mod_hz,
        fm_index: (2.2 + 3.2 * surprise).clamp(2.0, 6.0),
        filter_cutoff_hz: cutoff_hz,
        transient_impulse: 0.85,
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * 1.059_463_1),
        gain: 0.42,
        pan: (pan - 0.35).clamp(-1.0, 1.0),
        fm_mod_hz: clamp_freq_hz(carrier_hz * sqrt2),
        fm_index: 3.2,
        filter_cutoff_hz: cutoff_hz,
        transient_impulse: 0.45,
    };
    frame.voices[2] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * sqrt2),
        gain: 0.34,
        pan: (pan + 0.35).clamp(-1.0, 1.0),
        fm_mod_hz,
        fm_index: 2.8,
        filter_cutoff_hz: cutoff_hz,
        transient_impulse: 0.30,
    };
}

#[allow(clippy::too_many_arguments)]
fn populate_step_voices(
    frame: &mut SonificationFrame,
    carrier_hz: f32,
    pan: f32,
    surprise: f32,
    v_norm: f32,
    fm_mod_hz: f32,
    fm_index: f32,
    cutoff_hz: f32,
) {
    frame.waveform = WaveformPreset::FourierWarm;
    frame.active_voices = 2;
    frame.voices[0] = VoiceSpec {
        freq_hz: carrier_hz,
        gain: (0.38 + 0.36 * surprise).clamp(0.25, 0.78),
        pan,
        fm_mod_hz,
        fm_index,
        filter_cutoff_hz: cutoff_hz,
        transient_impulse: (0.08 + 0.35 * surprise).clamp(0.08, 0.45),
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * 1.5),
        gain: (0.14 + 0.24 * v_norm).clamp(0.10, 0.42),
        pan: (-0.5 * pan).clamp(-1.0, 1.0),
        fm_mod_hz: clamp_freq_hz(carrier_hz * 0.5),
        fm_index: (0.1 + 1.4 * surprise).clamp(0.1, 2.2),
        filter_cutoff_hz: clamp_freq_hz(900.0 + 3000.0 * v_norm),
        transient_impulse: 0.0,
    };
}
