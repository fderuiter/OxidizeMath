//! Zero-allocation Larmor spin-resonance sonification mappings across 0D FID,
//! 8-voice Hahn spin echoes, 2D k-space voxel ensembles, and quantum Zeeman Bloch states.

// theory_verification!

use std::f64::consts::TAU;

use nalgebra::Vector3;

use crate::audio::frame::{
    HarmonicWeights, MAX_VOICES, NUM_HARMONICS, SonificationFrame, VoiceSpec, WaveformPreset,
    clamp_freq_hz,
};

/// Hydrogen proton gyromagnetic ratio $\gamma$ in $\text{rad}\cdot\text{s}^{-1}\cdot\text{T}^{-1}$ (`2.675e8`).
pub const PROTON_GAMMA_RAD_PER_S_T: f64 = 2.675e8;

/// Number of static-inhomogeneity spins simulated by `MriPulseWorkspace::evaluate_hahn_echo`.
pub const HAHN_ECHO_SPIN_COUNT: usize = 9;

/// Maps static magnetic field strength $|B_0|$ (Tesla) monotonically onto an audible heterodyne Larmor carrier (`Hz`).
#[inline]
#[must_use]
pub fn larmor_carrier_hz(b0_tesla: f64) -> f32 {
    if !b0_tesla.is_finite() {
        return 220.0;
    }
    let b_abs = b0_tesla.abs().min(35.0);
    clamp_freq_hz((220.0 + 180.0 * b_abs) as f32)
}

/// Extracts a zero-allocation quadrature NMR Free Induction Decay (`FID`) frame from
/// macroscopic magnetization $\vec{M}$, magnetic field $\vec{B}$, and equilibrium $M_0$.
#[must_use]
pub fn bloch_fid_frame(
    magnetization: Vector3<f64>,
    b_field: Vector3<f64>,
    m0: f64,
) -> SonificationFrame {
    if !is_finite_vec3(&magnetization) || !is_finite_vec3(&b_field) || !m0.is_finite() {
        return SonificationFrame::silent();
    }
    let scale = m0.abs().max(1e-6);
    let mx = (magnetization.x / scale).clamp(-1.5, 1.5);
    let my = (magnetization.y / scale).clamp(-1.5, 1.5);
    let mz = (magnetization.z / scale).clamp(-1.5, 1.5);
    let m_perp = mx.hypot(my).clamp(0.0, 1.25);
    if m_perp <= 1e-4 {
        return SonificationFrame::silent();
    }

    let carrier_hz = larmor_carrier_hz(b_field.z);
    let b1_tesla = b_field.x.hypot(b_field.y);
    let rabi_hz = ((PROTON_GAMMA_RAD_PER_S_T * b1_tesla / TAU) * 0.02).clamp(0.0, 240.0) as f32;
    let pan = ((mx / m_perp.max(1e-6)) * 0.75).clamp(-1.0, 1.0) as f32;
    let env = (m_perp as f32).clamp(0.0, 1.0);
    let inv_mix = (0.5 * (1.0 - mz)).clamp(0.0, 1.0) as f32;

    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    cos_a[0] = (mx as f32).clamp(-1.0, 1.0);
    sin_b[0] = (my as f32).clamp(-1.0, 1.0);
    cos_a[1] = 0.35 * inv_mix;
    sin_b[2] = 0.18 * env;

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::FourierWarm;
    frame.custom_harmonics = Some(HarmonicWeights { cos_a, sin_b }.normalized());
    frame.active_voices = 2;
    frame.master_gain = (0.18 + 0.66 * env).clamp(0.0, 0.85);
    frame.voices[0] = VoiceSpec {
        freq_hz: carrier_hz,
        gain: (0.20 + 0.72 * env).clamp(0.05, 0.92),
        pan,
        fm_mod_hz: (4.0 + rabi_hz).clamp(0.0, 400.0),
        fm_index: (0.12 + 0.015 * rabi_hz).clamp(0.0, 2.5),
        filter_cutoff_hz: clamp_freq_hz(900.0 + 3400.0 * env + 900.0 * inv_mix),
        transient_impulse: 0.0,
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * 1.5),
        gain: (0.10 + 0.42 * env * (0.35 + 0.65 * inv_mix)).clamp(0.0, 0.55),
        pan: (-0.75 * pan).clamp(-1.0, 1.0),
        fm_mod_hz: (2.0 + 0.5 * rabi_hz).clamp(0.0, 200.0),
        fm_index: 0.15,
        filter_cutoff_hz: clamp_freq_hz(1200.0 + 2600.0 * env),
        transient_impulse: 0.0,
    };
    frame.sanitize()
}

/// Folds 9 static-inhomogeneity Hahn spin isochromats symmetrically into `MAX_VOICES = 8` voices,
/// rendering destructive $T_2^*$ dephasing beats mid-interval and coherent chord recovery at $t = TE$.
#[must_use]
pub fn hahn_echo_isochromat_frame(
    spins: &[Vector3<f64>; HAHN_ECHO_SPIN_COUNT],
    offsets_hz: &[f64; HAHN_ECHO_SPIN_COUNT],
    b0_tesla: f64,
    m0: f64,
) -> SonificationFrame {
    if !m0.is_finite()
        || !b0_tesla.is_finite()
        || !spins.iter().all(is_finite_vec3)
        || !offsets_hz.iter().all(|df| df.is_finite())
    {
        return SonificationFrame::silent();
    }
    let scale = m0.abs().max(1e-6);
    let (folded_spins, folded_offsets) = fold_nine_to_eight(spins, offsets_hz);
    ensemble_eight_voice_frame(
        &folded_spins,
        &folded_offsets,
        larmor_carrier_hz(b0_tesla),
        0.0,
        scale,
    )
}

/// Extracts an 8-voice sub-voxel ensemble frame at `selected_voxel` in `MriKSpaceStudio`,
/// shifting pitch and pan with gradient phase/k-space encoding and modeling $\Delta B_0$ $T_2^*$ echoes.
#[must_use]
pub fn kspace_voxel_ensemble_frame(
    probe_spins: &[Vector3<f64>; MAX_VOICES],
    delta_b0_tesla: f64,
    grad_phase_rad: f64,
    k_coord_cycles_per_m: [f64; 2],
    m0: f64,
) -> SonificationFrame {
    if !m0.is_finite()
        || m0 <= 1e-4
        || !delta_b0_tesla.is_finite()
        || !grad_phase_rad.is_finite()
        || !k_coord_cycles_per_m.iter().all(|k| k.is_finite())
        || !probe_spins.iter().all(is_finite_vec3)
    {
        return SonificationFrame::silent();
    }
    let kx_shift = (k_coord_cycles_per_m[0] * 1.8).clamp(-140.0, 140.0);
    let ky_shift = (k_coord_cycles_per_m[1] * 1.2).clamp(-100.0, 100.0);
    let phase_shift = grad_phase_rad.sin() * 18.0;
    let base_hz = clamp_freq_hz((330.0 + kx_shift + ky_shift + phase_shift) as f32);
    let pan_bias = (grad_phase_rad.sin() * 0.22).clamp(-0.30, 0.30) as f32;

    // Physical sub-voxel frequency spread scaled into distinct audible beating offsets
    let larmor_pocket_hz = PROTON_GAMMA_RAD_PER_S_T * delta_b0_tesla.abs() / TAU;
    let step_hz = (larmor_pocket_hz * 1.15).clamp(0.0, 16.0);
    let mut offsets_hz = [0.0_f64; MAX_VOICES];
    for (i, slot) in offsets_hz.iter_mut().enumerate() {
        *slot = ((i as f64) - 3.5) * step_hz;
    }
    ensemble_eight_voice_frame(probe_spins, &offsets_hz, base_hz, pan_bias, m0.max(1e-6))
}

/// Maps quantum spin-1/2 Bloch expectation $\langle\vec{\sigma}\rangle$ and magnetic field $\vec{B}$
/// to Zeeman Larmor pitch, stereo pan, eigenstate interval balance, and precession torque harmonics.
#[must_use]
pub fn zeeman_bloch_frame(expectation_xyz: [f64; 3], b_field: [f64; 3]) -> SonificationFrame {
    if !expectation_xyz.iter().all(|v| v.is_finite()) || !b_field.iter().all(|v| v.is_finite()) {
        return SonificationFrame::silent();
    }
    let [ex, ey, _] = expectation_xyz;
    let Some((b_mag, align, torque)) = zeeman_alignment_and_torque(expectation_xyz, b_field) else {
        return SonificationFrame::silent();
    };

    let excited = (0.5 * (1.0 - align)) as f32;
    let ground = (1.0 - excited).clamp(0.0, 1.0);
    let torque_f = torque as f32;
    let larmor_hz = clamp_freq_hz((165.0 + 110.0 * b_mag.min(12.0)) as f32 + 24.0 * excited);
    let beat_hz = (2.0 + 14.0 * (b_mag.min(8.0) * torque)) as f32;
    let pan = (ex as f32).clamp(-1.0, 1.0);

    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    cos_a[0] = (0.55 + 0.45 * ground).clamp(0.1, 1.0);
    sin_b[0] = (ey as f32 * 0.65).clamp(-1.0, 1.0);
    cos_a[1] = 0.50 * torque_f;
    sin_b[2] = 0.40 * excited;
    cos_a[3] = 0.25 * torque_f * excited;

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::QuantumModal;
    frame.custom_harmonics = Some(HarmonicWeights { cos_a, sin_b }.normalized());
    frame.active_voices = 3;
    frame.master_gain = (0.52 + 0.24 * torque_f).clamp(0.25, 0.80);
    frame.voices[0] = VoiceSpec {
        freq_hz: larmor_hz,
        gain: (0.28 + 0.46 * ground).clamp(0.15, 0.78),
        pan,
        fm_mod_hz: beat_hz,
        fm_index: (0.15 + 1.65 * torque_f).clamp(0.1, 2.5),
        filter_cutoff_hz: clamp_freq_hz(950.0 + 2800.0 * torque_f + 900.0 * excited),
        transient_impulse: 0.0,
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(larmor_hz * 1.5),
        gain: (0.12 + 0.52 * excited).clamp(0.08, 0.68),
        pan: (-0.65 * pan).clamp(-1.0, 1.0),
        fm_mod_hz: beat_hz * 0.5,
        fm_index: (0.10 + 1.10 * excited).clamp(0.05, 1.8),
        filter_cutoff_hz: clamp_freq_hz(1200.0 + 2600.0 * excited),
        transient_impulse: 0.0,
    };
    frame.voices[2] = VoiceSpec {
        freq_hz: clamp_freq_hz(larmor_hz + beat_hz),
        gain: (0.50 * torque_f).clamp(0.0, 0.52),
        pan: (ey as f32 * 0.75).clamp(-1.0, 1.0),
        fm_mod_hz: beat_hz,
        fm_index: 0.35 + 0.90 * torque_f,
        filter_cutoff_hz: clamp_freq_hz(1400.0 + 2800.0 * torque_f),
        transient_impulse: 0.0,
    };
    frame.sanitize()
}

fn fold_nine_to_eight(
    spins: &[Vector3<f64>; HAHN_ECHO_SPIN_COUNT],
    offsets_hz: &[f64; HAHN_ECHO_SPIN_COUNT],
) -> ([Vector3<f64>; MAX_VOICES], [f64; MAX_VOICES]) {
    let center_spin = spins[4] * 0.5;
    let center_off = offsets_hz[4] * 0.5;
    let inv_inner = 1.0 / 1.5;
    let folded_spins = [
        spins[0],
        spins[1],
        spins[2],
        (spins[3] + center_spin) * inv_inner,
        (spins[5] + center_spin) * inv_inner,
        spins[6],
        spins[7],
        spins[8],
    ];
    let folded_offsets = [
        offsets_hz[0],
        offsets_hz[1],
        offsets_hz[2],
        (offsets_hz[3] + center_off) * inv_inner,
        (offsets_hz[5] + center_off) * inv_inner,
        offsets_hz[6],
        offsets_hz[7],
        offsets_hz[8],
    ];
    (folded_spins, folded_offsets)
}

fn ensemble_eight_voice_frame(
    spins: &[Vector3<f64>; MAX_VOICES],
    offsets_hz: &[f64; MAX_VOICES],
    base_hz: f32,
    pan_bias: f32,
    scale: f64,
) -> SonificationFrame {
    let mut net_x = 0.0_f64;
    let mut net_y = 0.0_f64;
    let mut sum_mag = 0.0_f64;
    let mut amps = [0.0_f64; MAX_VOICES];
    let mut phases = [0.0_f64; MAX_VOICES];

    for i in 0..MAX_VOICES {
        let mx = spins[i].x / scale;
        let my = spins[i].y / scale;
        let mag = mx.hypot(my);
        net_x += mx;
        net_y += my;
        sum_mag += mag;
        amps[i] = mag.clamp(0.0, 1.25);
        phases[i] = my.atan2(mx);
    }

    let inv_n = 1.0 / (MAX_VOICES as f64);
    let mean_amp = sum_mag * inv_n;
    if mean_amp <= 1e-4 {
        return SonificationFrame::silent();
    }
    let net_amp = (net_x.hypot(net_y) * inv_n).clamp(0.0, 1.25);
    let coherence = (net_amp / mean_amp).clamp(0.0, 1.0);
    let net_phase = net_y.atan2(net_x);
    let dispersion = 1.0 - 0.92 * coherence;

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::JustHarmonic;
    frame.active_voices = MAX_VOICES as u8;
    frame.master_gain = (0.10 + 0.74 * (net_amp as f32).clamp(0.0, 1.0)).clamp(0.06, 0.84);

    for i in 0..MAX_VOICES {
        let align = 0.5 * (1.0 + (phases[i] - net_phase).cos());
        let voice_gain = (amps[i] * (0.06 + 0.76 * coherence * align)).clamp(0.0, 0.85) as f32;
        let detune = (offsets_hz[i] * dispersion) as f32;
        let pan = (-0.84 + 0.24 * (i as f32) + pan_bias).clamp(-1.0, 1.0);
        frame.voices[i] = VoiceSpec {
            freq_hz: clamp_freq_hz(base_hz + detune),
            gain: voice_gain,
            pan,
            fm_mod_hz: clamp_freq_hz(offsets_hz[i].abs() as f32 * 0.5 + 20.0),
            fm_index: (0.45 * (1.0 - coherence as f32)).clamp(0.0, 0.6),
            filter_cutoff_hz: clamp_freq_hz(900.0 + 3800.0 * (coherence as f32)),
            transient_impulse: 0.0,
        };
    }
    frame.sanitize()
}

#[inline]
fn zeeman_alignment_and_torque(
    expectation_xyz: [f64; 3],
    b_field: [f64; 3],
) -> Option<(f64, f64, f64)> {
    let [ex, ey, ez] = expectation_xyz;
    let [bx, by, bz] = b_field;
    let s_norm = (ex * ex + ey * ey + ez * ez).sqrt().clamp(0.0, 1.0);
    if s_norm <= 1e-4 {
        return None;
    }
    let b_mag = (bx * bx + by * by + bz * bz).sqrt();
    if b_mag > 1e-6 {
        let dot = ((ex * bx + ey * by + ez * bz) / b_mag).clamp(-1.0, 1.0);
        let cx = ey * bz - ez * by;
        let cy = ez * bx - ex * bz;
        let cz = ex * by - ey * bx;
        let cross = ((cx * cx + cy * cy + cz * cz).sqrt() / b_mag).clamp(0.0, 1.0);
        Some((b_mag, dot, cross))
    } else {
        Some((b_mag, ez.clamp(-1.0, 1.0), ex.hypot(ey).clamp(0.0, 1.0)))
    }
}

#[inline]
fn is_finite_vec3(v: &Vector3<f64>) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}
