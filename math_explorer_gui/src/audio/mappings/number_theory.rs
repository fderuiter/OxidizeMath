//! Number-theory prime-power chords, Ulam quadratic polyrhythms, FRACTRAN register
//! Kolmogorov modulation, and finite-field elliptic curve torsion sonification.

// theory_verification!

use crate::audio::frame::{
    clamp_freq_hz, SonificationFrame, VoiceSpec, WaveformPreset, MAX_VOICES,
};

/// Root reference pitch (`A3 = 220.0 Hz`) for prime-power harmonic chords.
pub const PRIME_CHORD_ROOT_HZ: f32 = 220.0;

/// Stack-allocated sample for one Ulam spiral cursor or parallel Euler quadratic ray ($4n^2 + bn + c$).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UlamPolySample {
    /// Evaluated positive lattice integer $m \ge 1$ at the playhead step.
    pub value: u64,
    /// Whether `value` is prime.
    pub is_prime: bool,
    /// Step distance (`>= 1`) since the preceding prime along this ray or spiral trajectory.
    pub prime_gap: u32,
}

/// Parameters describing a finite-field elliptic curve torsion step `[k]P` on $E(\mathbb{F}_p)$.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EllipticTorsionSample {
    /// Prime field modulus $p \ge 5$.
    pub modulus_p: i64,
    /// Whether the Weierstrass curve is singular ($4a^3 + 27b^2 \equiv 0 \pmod p$).
    pub is_singular: bool,
    /// Affine coordinates `Some((x, y))` of `[k]P`, or `None` when `[k]P = \mathcal{O}`.
    pub kp_coords: Option<(i64, i64)>,
    /// Cyclic subgroup order $\operatorname{ord}(P) \ge 1$.
    pub p_order: usize,
    /// Active scalar multiplier $k$.
    pub scalar_k: i64,
    /// Hasse-Weil Frobenius trace $a_p = p + 1 - \#E(\mathbb{F}_p)$.
    pub frobenius_trace: i64,
    /// Number of torus boundary segments traversed by the active secant or tangent ray.
    pub torus_wraps: usize,
}

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
        let pan = stereo_spread(idx, count, 0.70);

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

/// Synthesizes prime-gap polyrhythmic just-intonation pulses across up to 8 Ulam rays/cursors.
#[must_use]
pub fn ulam_spiral_frame(
    samples: &[UlamPolySample],
    tempo_bpm: f32,
    pulse_gate: bool,
) -> SonificationFrame {
    if samples.is_empty() || !tempo_bpm.is_finite() || tempo_bpm <= 0.0 {
        return SonificationFrame::silent();
    }
    let count = samples.len().min(MAX_VOICES);
    let prime_hits = samples[..count].iter().filter(|s| s.is_prime).count();
    if prime_hits == 0 {
        return composite_ulam_fallback_frame(samples[0]);
    }

    let tempo_hz = (tempo_bpm / 60.0).clamp(0.5, 24.0);
    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::JustHarmonic;
    frame.master_gain = (0.58 + 0.035 * (prime_hits as f32)).clamp(0.58, 0.84);

    let mut slot = 0usize;
    for (ray_idx, &sample) in samples[..count].iter().enumerate() {
        if !sample.is_prime || slot >= MAX_VOICES {
            continue;
        }
        frame.voices[slot] = build_ulam_prime_voice(
            sample, ray_idx, slot, count, prime_hits, tempo_hz, pulse_gate,
        );
        slot += 1;
    }
    frame.active_voices = slot as u8;
    frame.sanitize()
}

fn composite_ulam_fallback_frame(primary: UlamPolySample) -> SonificationFrame {
    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::Sine;
    frame.active_voices = 1;
    frame.master_gain = 0.22;
    let ratio = just_intonation_ratio(primary.value.max(2));
    frame.voices[0] = VoiceSpec {
        freq_hz: clamp_freq_hz(PRIME_CHORD_ROOT_HZ * 0.5 * ratio),
        gain: 0.16,
        pan: 0.0,
        fm_mod_hz: 0.0,
        fm_index: 0.0,
        filter_cutoff_hz: 360.0,
        transient_impulse: 0.0,
    };
    frame.sanitize()
}

fn build_ulam_prime_voice(
    sample: UlamPolySample,
    ray_idx: usize,
    slot: usize,
    count: usize,
    prime_hits: usize,
    tempo_hz: f32,
    pulse_gate: bool,
) -> VoiceSpec {
    let ratio = just_intonation_ratio(sample.value.max(2));
    let octave_lift = if ray_idx >= 4 { 2.0 } else { 1.0 };
    let freq_hz = clamp_freq_hz(PRIME_CHORD_ROOT_HZ * ratio * octave_lift);
    let gap = sample.prime_gap.max(1) as f32;
    let poly_hz = (tempo_hz * ((ray_idx + 1) as f32) / gap).clamp(0.5, 240.0);
    let base_gain = if ray_idx == 0 {
        0.74
    } else {
        0.48 / (prime_hits.max(1) as f32).sqrt()
    };
    let gate_scale = if pulse_gate { 1.0 } else { 0.72 };
    let impulse = if pulse_gate {
        (0.95 / (1.0 + 0.22 * (slot as f32))).clamp(0.30, 0.95)
    } else {
        0.0
    };
    VoiceSpec {
        freq_hz,
        gain: (base_gain * gate_scale).clamp(0.12, 0.86),
        pan: stereo_spread(ray_idx, count, 0.75),
        fm_mod_hz: poly_hz,
        fm_index: (0.16 + 0.95 / gap).clamp(0.12, 1.45),
        filter_cutoff_hz: clamp_freq_hz((5600.0 - 160.0 * (gap - 1.0)).clamp(1800.0, 7000.0)),
        transient_impulse: impulse,
    }
}

/// Synthesizes an up-to-8-voice just-intonation chord over active FRACTRAN prime registers,
/// modulating filter cutoff from `bdm_bits` and FM depth from `lz76_bits`, chiming on $2^k$,
/// and silencing cleanly on halts or Floyd cycles.
#[must_use]
pub fn fractran_register_frame(
    primes: &[u64],
    exponents: &[u32],
    pow2_hit: Option<u32>,
    bdm_bits: f64,
    lz76_bits: f64,
    halted_or_cycled: bool,
) -> SonificationFrame {
    if halted_or_cycled {
        return SonificationFrame::silent();
    }
    let (selected, count) = collect_top_prime_registers(primes, exponents);
    if count == 0 {
        return SonificationFrame::silent();
    }
    let bdm_norm = norm_metric(bdm_bits, 160.0);
    let lz_norm = norm_metric(lz76_bits, 96.0);
    let total_exp: u64 = selected[..count].iter().map(|&(_, e)| u64::from(e)).sum();
    let inv_total = 1.0 / (total_exp.max(1) as f32);

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::JustHarmonic;
    frame.active_voices = count as u8;
    frame.master_gain = 0.76;

    for (idx, &(prime, exp)) in selected[..count].iter().enumerate() {
        let e = exp.max(1) as f32;
        let ratio = just_intonation_ratio(prime);
        let freq_hz = clamp_freq_hz(PRIME_CHORD_ROOT_HZ * ratio * if idx >= 4 { 2.0 } else { 1.0 });
        let weight = (e * inv_total).clamp(0.08, 1.0);
        frame.voices[idx] = VoiceSpec {
            freq_hz,
            gain: (0.16 + 0.66 * weight).clamp(0.10, 0.86),
            pan: stereo_spread(idx, count, 0.72),
            fm_mod_hz: clamp_freq_hz(freq_hz * (1.0 + 0.50 * bdm_norm)),
            fm_index: (0.05 * (e - 1.0) + 2.40 * lz_norm).clamp(0.0, 4.5),
            filter_cutoff_hz: clamp_freq_hz(900.0 + 4900.0 * bdm_norm + 180.0 * e.min(8.0)),
            transient_impulse: 0.0,
        };
    }
    if let Some(k) = pow2_hit.filter(|&k| k >= 2) {
        apply_pow2_chime(&mut frame, k, lz_norm);
    }
    frame.sanitize()
}

fn collect_top_prime_registers(
    primes: &[u64],
    exponents: &[u32],
) -> ([(u64, u32); MAX_VOICES], usize) {
    let mut buf = [(0_u64, 0_u32); MAX_VOICES];
    let mut len = 0usize;
    for (&prime, &exp) in primes.iter().zip(exponents.iter()) {
        if exp == 0 || prime < 2 {
            continue;
        }
        if len < MAX_VOICES {
            buf[len] = (prime, exp);
            len += 1;
        } else if let Some((min_idx, &(_, min_e))) =
            buf.iter().enumerate().min_by_key(|&(_, &(_, e))| e)
        {
            if exp > min_e {
                buf[min_idx] = (prime, exp);
            }
        }
    }
    for i in 1..len {
        let mut j = i;
        while j > 0 && buf[j - 1].0 > buf[j].0 {
            buf.swap(j - 1, j);
            j -= 1;
        }
    }
    (buf, len)
}

fn apply_pow2_chime(frame: &mut SonificationFrame, exponent_k: u32, lz_norm: f32) {
    frame.voices[0].gain = frame.voices[0].gain.max(0.80);
    frame.voices[0].filter_cutoff_hz = 7200.0;
    frame.voices[0].transient_impulse = 1.0;

    let chime_idx = if frame.active_voices <= 1 {
        1
    } else {
        (frame.active_voices as usize).min(MAX_VOICES - 1)
    };
    let chime_freq =
        clamp_freq_hz(PRIME_CHORD_ROOT_HZ * 2.0 * just_intonation_ratio(u64::from(exponent_k)));
    frame.voices[chime_idx] = VoiceSpec {
        freq_hz: chime_freq,
        gain: 0.64,
        pan: 0.32,
        fm_mod_hz: clamp_freq_hz(chime_freq * 2.0),
        fm_index: (0.40 + 1.10 * lz_norm).clamp(0.30, 2.2),
        filter_cutoff_hz: 7400.0,
        transient_impulse: 0.92,
    };
    frame.active_voices = frame.active_voices.max((chime_idx + 1) as u8);
}

/// Maps `[k]P` torsion coordinates, cyclic subgroup order `p_order`, Frobenius trace, and
/// torus secant wraps into harmonic intervals repeating modulo `p_order`.
#[must_use]
pub fn elliptic_torsion_frame(sample: EllipticTorsionSample) -> SonificationFrame {
    if sample.is_singular || sample.modulus_p < 5 || sample.p_order == 0 {
        return SonificationFrame::silent();
    }
    let p_u64 = sample.modulus_p as u64;
    let p_f = sample.modulus_p as f32;
    let order = sample.p_order as u64;
    let residue_k = sample.scalar_k.rem_euclid(sample.p_order as i64) as u64;
    let hasse_limit = (2.0 * p_f.sqrt()).max(1.0);
    let hasse_norm = ((sample.frobenius_trace as f32) / hasse_limit).clamp(-1.0, 1.0);
    let root_hz = clamp_freq_hz(PRIME_CHORD_ROOT_HZ * 0.5 * just_intonation_ratio(p_u64));
    let order_hz = clamp_freq_hz(root_hz * just_intonation_ratio(order.max(2)));
    let trace_ratio = 1.5 * (1.0 + 0.08 * hasse_norm);

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::JustHarmonic;
    frame.master_gain = 0.74;
    frame.voices[0] = VoiceSpec {
        freq_hz: root_hz,
        gain: 0.34,
        pan: -0.35 * hasse_norm,
        fm_mod_hz: root_hz,
        fm_index: 0.12,
        filter_cutoff_hz: clamp_freq_hz(1800.0 + 900.0 * (hasse_norm + 1.0)),
        transient_impulse: 0.0,
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: order_hz,
        gain: 0.30,
        pan: -0.20,
        fm_mod_hz: order_hz * 0.5,
        fm_index: 0.15,
        filter_cutoff_hz: 2600.0,
        transient_impulse: 0.0,
    };
    frame.voices[2] = VoiceSpec {
        freq_hz: clamp_freq_hz(root_hz * trace_ratio),
        gain: 0.26,
        pan: 0.20,
        fm_mod_hz: clamp_freq_hz(root_hz * 0.5),
        fm_index: 0.20 * hasse_norm.abs(),
        filter_cutoff_hz: clamp_freq_hz(2200.0 + 1200.0 * hasse_norm.abs()),
        transient_impulse: 0.0,
    };
    populate_torsion_orbit_voices(&mut frame, sample, root_hz, p_f, residue_k);
    frame.active_voices = 5;
    frame.sanitize()
}

fn populate_torsion_orbit_voices(
    frame: &mut SonificationFrame,
    sample: EllipticTorsionSample,
    root_hz: f32,
    p_f: f32,
    residue_k: u64,
) {
    let wraps = sample.torus_wraps.saturating_sub(1) as f32;
    if let Some((raw_x, raw_y)) = sample.kp_coords {
        let x_mod = raw_x.rem_euclid(sample.modulus_p);
        let y_mod = raw_y.rem_euclid(sample.modulus_p);
        let x_hz = clamp_freq_hz(root_hz * 2.0 * just_intonation_ratio((x_mod as u64) + 2));
        let y_hz = clamp_freq_hz(
            root_hz * 2.0 * just_intonation_ratio((y_mod as u64) + 2 + (residue_k % 5)),
        );
        let x_pan = ((2.0 * (x_mod as f32) / p_f) - 1.0).clamp(-0.85, 0.85);
        let wrap_impulse = (0.28 * wraps).clamp(0.0, 0.90);
        frame.voices[3] = VoiceSpec {
            freq_hz: x_hz,
            gain: 0.56,
            pan: x_pan,
            fm_mod_hz: clamp_freq_hz(x_hz * 0.5),
            fm_index: (0.18 + 0.24 * wraps).clamp(0.15, 2.4),
            filter_cutoff_hz: clamp_freq_hz(2800.0 + 2600.0 * ((y_mod as f32) / p_f)),
            transient_impulse: wrap_impulse,
        };
        frame.voices[4] = VoiceSpec {
            freq_hz: y_hz,
            gain: 0.44,
            pan: (-0.6 * x_pan).clamp(-0.85, 0.85),
            fm_mod_hz: clamp_freq_hz(root_hz * ((residue_k % 7 + 1) as f32)),
            fm_index: (0.14 + 0.18 * wraps).clamp(0.10, 2.0),
            filter_cutoff_hz: clamp_freq_hz(3200.0 + 320.0 * wraps),
            transient_impulse: (0.5 * wrap_impulse).clamp(0.0, 0.60),
        };
    } else {
        frame.voices[3] = VoiceSpec {
            freq_hz: clamp_freq_hz(root_hz * 2.0),
            gain: 0.62,
            pan: 0.0,
            fm_mod_hz: root_hz,
            fm_index: 0.04,
            filter_cutoff_hz: 5200.0,
            transient_impulse: 0.78,
        };
        frame.voices[4] = VoiceSpec {
            freq_hz: clamp_freq_hz(root_hz * 4.0),
            gain: 0.46,
            pan: 0.0,
            fm_mod_hz: clamp_freq_hz(root_hz * 2.0),
            fm_index: 0.04,
            filter_cutoff_hz: 6000.0,
            transient_impulse: 0.60,
        };
    }
}

#[inline]
fn stereo_spread(idx: usize, count: usize, span: f32) -> f32 {
    if count <= 1 {
        0.0
    } else {
        -span + (2.0 * span) * ((idx as f32) / ((count - 1) as f32))
    }
}

#[inline]
fn norm_metric(value: f64, scale: f64) -> f32 {
    if value.is_finite() && scale > 0.0 {
        (value / scale).clamp(0.0, 1.0) as f32
    } else {
        0.25
    }
}
