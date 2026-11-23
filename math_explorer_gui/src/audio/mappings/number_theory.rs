//! Number-theory prime-power chord synthesis in pure just intonation.

use crate::audio::frame::{
    clamp_freq_hz, SonificationFrame, VoiceSpec, WaveformPreset, MAX_VOICES,
};

/// Root reference pitch (`A3 = 220.0 Hz`) for prime-power harmonic chords.
pub const PRIME_CHORD_ROOT_HZ: f32 = 220.0;

/// Computes the octave-normalized just-intonation harmonic ratio for prime `p`.
///
/// Maps `p = 2 -> 2.0` (`2:1` octave), `p = 3 -> 1.5` (`3:2` fifth),
/// `p = 5 -> 1.25` (`5:4` major third), `p = 7 -> 1.75` (`7:4` harmonic seventh),
/// and general odd prime `p -> p / 2^floor(log2(p))` in `(1.0, 2.0)`.
#[must_use]
pub fn just_intonation_ratio(prime: u64) -> f32 {
    if prime <= 1 {
        return 1.0;
    }
    if prime == 2 {
        return 2.0;
    }
    let log2_floor = 63_u32.saturating_sub(prime.leading_zeros());
    let denom = (1_u64 << log2_floor.min(62)) as f64;
    ((prime as f64) / denom).clamp(1.0, 2.0) as f32
}

/// Maps prime-power pairs `(p, e)` to up to 8 just-intonation harmonic voices weighted by `e`.
#[must_use]
pub fn prime_chord_frame(factors: &[(u64, u32)], waveform: WaveformPreset) -> SonificationFrame {
    if factors.is_empty() {
        return SonificationFrame::silent();
    }
    let count = factors.len().min(MAX_VOICES);
    let total_exp: u32 = factors[..count].iter().map(|&(_, e)| e.max(1)).sum();
    let inv_total = 1.0 / (total_exp.max(1) as f32);

    let mut frame = SonificationFrame::silent();
    frame.waveform = waveform;
    frame.active_voices = count as u8;
    frame.master_gain = 0.76;

    for (idx, &(prime, exp)) in factors[..count].iter().enumerate() {
        let e = exp.max(1) as f32;
        let ratio = just_intonation_ratio(prime);
        let octave_lift = if idx >= 4 { 2.0 } else { 1.0 };
        let freq_hz = clamp_freq_hz(PRIME_CHORD_ROOT_HZ * ratio * octave_lift);
        let weight = (e * inv_total).clamp(0.08, 1.0);
        let pan = if count <= 1 {
            0.0
        } else {
            -0.70 + 1.40 * ((idx as f32) / ((count - 1) as f32))
        };

        frame.voices[idx] = VoiceSpec {
            freq_hz,
            gain: (0.14 + 0.72 * weight).clamp(0.10, 0.88),
            pan,
            fm_mod_hz: freq_hz,
            fm_index: (0.08 * (e - 1.0)).clamp(0.0, 0.8),
            filter_cutoff_hz: clamp_freq_hz(1600.0 + 750.0 * e),
            transient_impulse: 0.0,
        };
    }
    frame.sanitize()
}
