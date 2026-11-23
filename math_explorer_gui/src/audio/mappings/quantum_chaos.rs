//! Quantum wavepacket and chaotic dynamical system sonification mappings.

// theory_verification!

use std::f64::consts::{PI, TAU};

use num_complex::Complex;

use crate::audio::frame::{
    clamp_freq_hz, HarmonicWeights, SonificationFrame, VoiceSpec, WaveformPreset, NUM_HARMONICS,
};

/// Projects complex spatial-mode amplitudes and phase velocities into `PeriodicWave` harmonics
/// and detuned beat-frequency oscillator voices prior to `norm_sqr()` collapse.
#[must_use]
pub fn quantum_wavepacket_frame(
    modes: &[Complex<f64>; NUM_HARMONICS],
    prev_phases: &mut [f64; NUM_HARMONICS],
    dt: f64,
) -> SonificationFrame {
    let safe_dt = dt.abs().max(1e-4);
    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    let mut mags = [0.0_f64; NUM_HARMONICS];
    let mut phase_vel_hz = [0.0_f64; NUM_HARMONICS];

    for idx in 0..NUM_HARMONICS {
        let c = modes[idx];
        if !c.re.is_finite() || !c.im.is_finite() {
            continue;
        }
        let mag = c.norm();
        mags[idx] = mag;
        cos_a[idx] = c.re as f32;
        sin_b[idx] = c.im as f32;
        if mag > 1e-6 {
            let phase = c.im.atan2(c.re);
            let delta = wrap_angle_pi(phase - prev_phases[idx]);
            prev_phases[idx] = phase;
            phase_vel_hz[idx] = (delta / (TAU * safe_dt)).clamp(-40.0, 40.0);
        }
    }

    let total_mag: f64 = mags.iter().sum();
    if total_mag <= 1e-6 {
        return SonificationFrame::silent();
    }

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::QuantumModal;
    frame.custom_harmonics = Some(HarmonicWeights { cos_a, sin_b }.normalized());
    frame.master_gain = 0.74;

    let top_indices = top_four_mode_indices(&mags);
    let beat_hz = (phase_vel_hz[top_indices[0]] - phase_vel_hz[top_indices[1]])
        .abs()
        .clamp(0.5, 24.0) as f32;
    let base_hz = 165.0_f32;

    for (slot, &mode_idx) in top_indices.iter().enumerate() {
        let weight = (mags[mode_idx] / total_mag) as f32;
        let detune = (phase_vel_hz[mode_idx] * 0.45).clamp(-18.0, 18.0) as f32;
        let partial = (mode_idx + 1) as f32;
        let pan = -0.6 + (slot as f32) * 0.4;
        frame.voices[slot] = VoiceSpec {
            freq_hz: clamp_freq_hz(base_hz * partial + detune),
            gain: (weight * 1.15).clamp(0.04, 0.75),
            pan,
            fm_mod_hz: beat_hz,
            fm_index: (0.25 + 0.08 * beat_hz).clamp(0.2, 2.4),
            filter_cutoff_hz: clamp_freq_hz(1100.0 + 700.0 * partial),
            transient_impulse: 0.0,
        };
    }
    frame.voices[4] = VoiceSpec {
        freq_hz: clamp_freq_hz(base_hz * ((top_indices[0] + 1) as f32) + beat_hz),
        gain: 0.32,
        pan: 0.15,
        fm_mod_hz: beat_hz * 0.5,
        fm_index: 0.35,
        filter_cutoff_hz: 2600.0,
        transient_impulse: 0.0,
    };
    frame.active_voices = 5;
    frame.sanitize()
}

fn top_four_mode_indices(mags: &[f64; NUM_HARMONICS]) -> [usize; 4] {
    let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7];
    for i in 0..4 {
        let mut best = i;
        for j in (i + 1)..NUM_HARMONICS {
            if mags[order[j]] > mags[order[best]] {
                best = j;
            }
        }
        order.swap(i, best);
    }
    [order[0], order[1], order[2], order[3]]
}

#[inline]
fn wrap_angle_pi(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

/// Maps 3D chaotic attractor position `(x, y, z)` and velocity `(dx, dy, dz)` to stereo pan and FM pitch.
#[must_use]
pub fn attractor_frame(pos: [f64; 3], deriv: [f64; 3], diverged: bool) -> SonificationFrame {
    if diverged || !pos.iter().all(|v| v.is_finite()) || !deriv.iter().all(|v| v.is_finite()) {
        return SonificationFrame::silent();
    }
    let speed = (deriv[0] * deriv[0] + deriv[1] * deriv[1] + deriv[2] * deriv[2]).sqrt();
    let pan = (pos[0] / 20.0).clamp(-1.0, 1.0) as f32;
    let z_norm = ((pos[2] + 5.0) / 55.0).clamp(0.0, 1.0) as f32;
    let speed_norm = (speed / 75.0).clamp(0.0, 1.0) as f32;
    let carrier_hz = clamp_freq_hz(110.0 + z_norm * 550.0);

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::FourierWarm;
    frame.active_voices = 2;
    frame.master_gain = 0.74;
    frame.voices[0] = VoiceSpec {
        freq_hz: carrier_hz,
        gain: (0.35 + 0.45 * speed_norm).clamp(0.15, 0.85),
        pan,
        fm_mod_hz: (35.0 + speed as f32 * 2.2).clamp(20.0, 1200.0),
        fm_index: (0.25 + 4.2 * speed_norm).clamp(0.2, 5.0),
        filter_cutoff_hz: clamp_freq_hz(500.0 + 4500.0 * speed_norm),
        transient_impulse: 0.0,
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * 1.5),
        gain: (0.18 + 0.28 * z_norm).clamp(0.10, 0.48),
        pan: (-0.5 * pan).clamp(-1.0, 1.0),
        fm_mod_hz: (25.0 + speed as f32 * 1.1).clamp(20.0, 800.0),
        fm_index: (0.15 + 1.8 * speed_norm).clamp(0.1, 2.5),
        filter_cutoff_hz: clamp_freq_hz(800.0 + 3200.0 * z_norm),
        transient_impulse: 0.0,
    };
    frame.sanitize()
}

/// Maps logistic growth parameter `r` to period-doubling octave subharmonics and chaotic filtered noise.
#[must_use]
pub fn bifurcation_frame(r: f64) -> SonificationFrame {
    if !r.is_finite() || r <= 0.01 {
        return SonificationFrame::silent();
    }
    let r_clamped = r.clamp(0.5, 4.0);
    let (orbit, lyapunov) = sample_logistic_orbit(r_clamped);
    let mean_x = (orbit.iter().sum::<f64>() / (orbit.len() as f64)).clamp(0.0, 1.0) as f32;
    let base_hz = clamp_freq_hz(240.0 + ((r_clamped - 2.5).max(0.0) as f32) * 160.0);

    let diff_p2 = (orbit[15] - orbit[14]).abs() as f32;
    let diff_p4 = (orbit[15] - orbit[13]).abs() as f32;
    let diff_p8 = (orbit[15] - orbit[11]).abs() as f32;

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::SubharmonicSquare;
    frame.active_voices = 4;
    frame.master_gain = 0.72;
    frame.voices[0] = VoiceSpec {
        freq_hz: base_hz,
        gain: (0.28 + 0.45 * mean_x).clamp(0.15, 0.75),
        pan: 0.0,
        fm_mod_hz: base_hz * 0.5,
        fm_index: 0.25 + 0.6 * diff_p2,
        filter_cutoff_hz: clamp_freq_hz(1400.0 + 2200.0 * mean_x),
        transient_impulse: 0.0,
    };
    frame.voices[1] = subharmonic_voice(base_hz * 0.5, diff_p2, 0.012, 1.35, -0.35);
    frame.voices[2] = subharmonic_voice(base_hz * 0.25, diff_p4, 0.008, 1.65, 0.35);
    frame.voices[3] = subharmonic_voice(base_hz * 0.125, diff_p8, 0.004, 1.90, -0.55);

    if lyapunov > 0.0 {
        let chaos = (lyapunov / 0.70).clamp(0.0, 1.0) as f32;
        frame.noise_gain = (0.12 + 0.68 * chaos).clamp(0.0, 0.80);
        frame.noise_cutoff_hz = clamp_freq_hz(700.0 + 4800.0 * chaos);
    }
    frame.sanitize()
}

fn subharmonic_voice(
    freq_hz: f32,
    orbit_split: f32,
    threshold: f32,
    scale: f32,
    pan: f32,
) -> VoiceSpec {
    let gain = if orbit_split > threshold {
        ((orbit_split - threshold) * scale + 0.18).clamp(0.0, 0.62)
    } else {
        0.0
    };
    VoiceSpec {
        freq_hz: clamp_freq_hz(freq_hz),
        gain,
        pan,
        fm_mod_hz: clamp_freq_hz(freq_hz * 0.5),
        fm_index: 0.2,
        filter_cutoff_hz: clamp_freq_hz(freq_hz * 5.0),
        transient_impulse: 0.0,
    }
}

fn sample_logistic_orbit(r: f64) -> ([f64; 16], f64) {
    let mut x = 0.5_f64;
    for _ in 0..160 {
        x = (r * x * (1.0 - x)).clamp(0.0, 1.0);
    }
    let mut lyap_sum = 0.0_f64;
    let mut orbit = [x; 16];
    for step in 0..64 {
        x = (r * x * (1.0 - x)).clamp(0.0, 1.0);
        let deriv = (r * (1.0 - 2.0 * x)).abs().max(1e-9);
        lyap_sum += deriv.ln();
        if step >= 48 {
            orbit[step - 48] = x;
        }
    }
    (orbit, lyap_sum / 64.0)
}
