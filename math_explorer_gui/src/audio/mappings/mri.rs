//! Zero-allocation MRI Bloch, Hahn spin-echo, and 2D k-space Turing phantom sonification mappings.

// theory_verification!

use std::f64::consts::TAU;

use math_explorer::physics::mri::proton;
use nalgebra::Vector3;

#[cfg(feature = "biology")]
use math_explorer::biology::morphogenesis::{ReactionKinetics, SchnakenbergKinetics};

use crate::audio::frame::{
    HarmonicWeights, NUM_HARMONICS, SonificationFrame, VoiceSpec, WaveformPreset, clamp_freq_hz,
};
use crate::tabs::mri::kspace_types::ISOCHROMAT_COUNT;
use crate::tabs::mri::pulse::{MriPulseMode, RfPulseConfig};

/// Stack-only input bundle for 2D MRI k-space and Turing phantom sonification.
#[derive(Clone, Copy, Debug)]
pub struct MriKSpaceAudioInput {
    /// Three-isochromat spin ensemble at the inspected or active pocket voxel.
    pub spins: [Vector3<f64>; ISOCHROMAT_COUNT],
    /// Local proton density $\rho \in [0, 1]$.
    pub rho: f64,
    /// Local transverse relaxation time $T_2$ in seconds.
    pub t2: f64,
    /// Local static field offset $\Delta B_0$ in Tesla.
    pub delta_b0: f64,
    /// Activator-inhibitor concentrations `[u, v]` on Turing phantoms.
    pub turing_uv: [f64; 2],
    /// Elapsed simulation time since excitation in seconds.
    pub current_time: f64,
    /// Recorded $180^\circ$ refocusing pulse time $\tau$ ($T_E = 2\tau$), if active.
    pub echo_tau: Option<f64>,
    /// Active 2D k-space coordinate $(k_x, k_y)$ in cycles/meter.
    pub k_coord_cycles_per_m: [f64; 2],
    /// Accumulated spatial readout/phase-encode gradient phase in radians.
    pub selected_grad_phase_rad: f64,
    /// Whether an off-resonance `KSpaceRfSpike` stroke fired since the previous frame.
    pub rf_spike_triggered: bool,
}

/// Maps rotating-frame Bloch magnetization $(M_x, M_y, M_z)$, $B$-field, and `detuning_hz` to audio.
#[must_use]
pub fn mri_bloch_frame(
    magnetization: Vector3<f64>,
    m0: f64,
    b_field: Vector3<f64>,
    detuning_hz: f64,
) -> SonificationFrame {
    if !magnetization.x.is_finite()
        || !magnetization.y.is_finite()
        || !magnetization.z.is_finite()
        || !m0.is_finite()
        || m0.abs() <= 1e-9
        || !b_field.x.is_finite()
        || !b_field.y.is_finite()
        || !b_field.z.is_finite()
        || !detuning_hz.is_finite()
    {
        return SonificationFrame::silent();
    }
    let inv_m0 = 1.0 / m0.abs();
    let m_xy = (magnetization.x.hypot(magnetization.y) * inv_m0).clamp(0.0, 1.25) as f32;
    let mz_norm = (magnetization.z * inv_m0).clamp(-1.0, 1.0) as f32;
    let carrier_hz = rotating_carrier_hz(b_field.z, detuning_hz);
    let b1_amp = b_field.x.hypot(b_field.y);
    let nutation_hz = ((proton::GYROMAGNETIC_RATIO * b1_amp) / TAU).clamp(0.0, 1200.0) as f32;
    let pan = (magnetization.y.atan2(magnetization.x).sin() as f32 * 0.70).clamp(-0.85, 0.85);
    let lead_gain = (0.20 + 0.70 * m_xy.min(1.0)).clamp(0.20, 0.92);

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::FourierWarm;
    frame.active_voices = 2;
    frame.master_gain = (0.34 + 0.50 * m_xy.min(1.0)).clamp(0.34, 0.86);
    frame.voices[0] = VoiceSpec {
        freq_hz: carrier_hz,
        gain: lead_gain,
        pan,
        fm_mod_hz: nutation_hz.max(12.0),
        fm_index: (0.15 + nutation_hz / 320.0).clamp(0.15, 2.8),
        filter_cutoff_hz: clamp_freq_hz(950.0 + 4200.0 * m_xy.min(1.0)),
        transient_impulse: 0.0,
    };
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * 0.5),
        gain: (0.06 + 0.22 * (0.5 * (1.0 - mz_norm))).min(lead_gain * 0.70),
        pan: -0.5 * pan,
        fm_mod_hz: 18.0,
        fm_index: 0.20,
        filter_cutoff_hz: clamp_freq_hz(750.0 + 1800.0 * (0.5 * (1.0 + mz_norm))),
        transient_impulse: 0.0,
    };
    frame.sanitize()
}

/// Maps a sample on the 201-point `MriPulseWorkspace` (`InversionPulse` / `HahnSpinEcho`) to audio.
#[must_use]
pub fn mri_pulse_playhead_frame(
    mode: MriPulseMode,
    sample_time_s: f64,
    magnetization: Vector3<f64>,
    dispersion: f64,
    m0: f64,
    b0: f64,
    cfg: &RfPulseConfig,
) -> SonificationFrame {
    if mode.is_continuous() {
        return mri_bloch_frame(
            magnetization,
            m0,
            Vector3::new(0.0, 0.0, b0),
            cfg.detuning_hz,
        );
    }
    if !sample_time_s.is_finite()
        || !magnetization.x.is_finite()
        || !magnetization.y.is_finite()
        || !magnetization.z.is_finite()
        || !m0.is_finite()
        || m0.abs() <= 1e-9
    {
        return SonificationFrame::silent();
    }
    let inv_m0 = 1.0 / m0.abs();
    let m_xy = (magnetization.x.hypot(magnetization.y) * inv_m0).clamp(0.0, 1.2) as f32;
    let mz_norm = (magnetization.z * inv_m0).clamp(-1.0, 1.0) as f32;
    let disp = if dispersion.is_finite() {
        dispersion.clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let carrier_hz = rotating_carrier_hz(b0, cfg.detuning_hz);
    let pulse_on = is_rf_pulse_window(mode, sample_time_s, cfg);
    let lead_gain = (0.14 + 0.78 * m_xy.min(1.0)).clamp(0.14, 0.92);

    let mut frame = SonificationFrame::silent();
    frame.waveform = match mode {
        MriPulseMode::HahnSpinEcho => WaveformPreset::QuantumModal,
        _ => WaveformPreset::FourierWarm,
    };
    frame.active_voices = 4;
    frame.master_gain = (0.18 + 0.72 * m_xy.min(1.0)).clamp(0.18, 0.90);
    frame.voices[0] = VoiceSpec {
        freq_hz: carrier_hz,
        gain: lead_gain,
        pan: 0.0,
        fm_mod_hz: 24.0 + 96.0 * disp,
        fm_index: (0.08 + 1.35 * disp + if pulse_on { 0.9 } else { 0.0 }).clamp(0.08, 3.2),
        filter_cutoff_hz: clamp_freq_hz(1100.0 + 4600.0 * m_xy.min(1.0)),
        transient_impulse: if pulse_on { 0.85 } else { 0.0 },
    };
    let side_gain = (0.20 * disp).min(lead_gain * 0.65);
    let spread_hz = 14.0 + 42.0 * disp;
    frame.voices[1] = pulse_side_voice(carrier_hz - spread_hz, side_gain, -0.65, disp);
    frame.voices[2] = pulse_side_voice(carrier_hz + spread_hz, side_gain, 0.65, disp);
    frame.voices[3] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * (1.25 + 0.25 * mz_norm)),
        gain: (0.06 + 0.16 * (0.5 * (1.0 - mz_norm))).min(lead_gain * 0.60),
        pan: 0.20,
        fm_mod_hz: 16.0,
        fm_index: 0.18,
        filter_cutoff_hz: 2200.0,
        transient_impulse: 0.0,
    };
    frame.sanitize()
}

/// Maps 2D MRI k-space 3-isochromat ensembles, Turing `(u, v)` tissues, gradients, and RF spikes.
#[must_use]
pub fn mri_kspace_frame(input: &MriKSpaceAudioInput) -> SonificationFrame {
    if !input.rho.is_finite()
        || input.rho <= 1e-5
        || !input.t2.is_finite()
        || !input.delta_b0.is_finite()
        || !input.current_time.is_finite()
        || !input
            .spins
            .iter()
            .all(|s| s.x.is_finite() && s.y.is_finite() && s.z.is_finite())
    {
        return SonificationFrame::silent();
    }
    let (net_xy, coherence, dephasing) = ensemble_coherence(&input.spins);
    let echo_boost = echo_refocus_boost(input.current_time, input.echo_tau);
    let (harmonics, morph_bright) =
        turing_tissue_harmonics(input.turing_uv, input.rho, input.t2, coherence);

    let larmor_df_hz = proton::larmor_frequency(input.delta_b0) / TAU;
    let pocket_shift = (6.0 * larmor_df_hz).clamp(-600.0, 600.0) as f32;
    let spread_hz = (5.1 * larmor_df_hz.abs() as f32 + 28.0 * dephasing).clamp(0.0, 420.0);
    let base_hz = clamp_freq_hz(330.0 + 110.0 * (input.rho.clamp(0.0, 1.0) as f32) + pocket_shift);
    let k_mag = input.k_coord_cycles_per_m[0].hypot(input.k_coord_cycles_per_m[1]);
    let (grad_fm_hz, grad_fm_idx) = gradient_chirp_fm(k_mag, input.selected_grad_phase_rad);
    let env = ((net_xy / input.rho.max(0.1)).clamp(0.0, 1.2) as f32 * (0.35 + 0.65 * coherence)
        + 0.18 * echo_boost)
        .clamp(0.06, 0.95);

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::FourierWarm;
    frame.custom_harmonics = Some(harmonics);
    frame.master_gain = (0.24 + 0.64 * env).clamp(0.24, 0.90);
    for s in 0..ISOCHROMAT_COUNT {
        let offset = (s as f32) - 1.0;
        let iso_xy =
            (input.spins[s].x.hypot(input.spins[s].y) / input.rho.max(0.1)).clamp(0.0, 1.1) as f32;
        frame.voices[s] = VoiceSpec {
            freq_hz: clamp_freq_hz(base_hz + offset * spread_hz),
            gain: ((0.14 + 0.56 * iso_xy * (0.30 + 0.70 * coherence)) + 0.12 * echo_boost)
                .clamp(0.06, 0.88),
            pan: (offset * 0.55).clamp(-0.85, 0.85),
            fm_mod_hz: grad_fm_hz + 14.0 * dephasing,
            fm_index: (grad_fm_idx + 1.1 * dephasing).clamp(0.05, 5.5),
            filter_cutoff_hz: clamp_freq_hz(900.0 + 3600.0 * morph_bright * coherence.max(0.25)),
            transient_impulse: if input.rf_spike_triggered && s == 1 {
                1.0
            } else if input.rf_spike_triggered {
                0.55
            } else {
                0.0
            },
        };
    }
    frame.voices[3] = kspace_gradient_voice(base_hz, k_mag, input, grad_fm_hz, grad_fm_idx);
    frame.active_voices = if k_mag > 1e-3 { 4 } else { 3 };
    frame.sanitize()
}

#[inline]
fn ensemble_coherence(spins: &[Vector3<f64>; ISOCHROMAT_COUNT]) -> (f64, f32, f32) {
    let net_m = (spins[0] + spins[1] + spins[2]) / (ISOCHROMAT_COUNT as f64);
    let net_xy = net_m.x.hypot(net_m.y);
    let sum_iso_xy =
        spins[0].x.hypot(spins[0].y) + spins[1].x.hypot(spins[1].y) + spins[2].x.hypot(spins[2].y);
    let mean_iso_xy = sum_iso_xy / (ISOCHROMAT_COUNT as f64);
    let coherence = if mean_iso_xy > 1e-6 {
        (net_xy / mean_iso_xy).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    (net_xy, coherence, (1.0 - coherence).clamp(0.0, 1.0))
}

#[inline]
fn rotating_carrier_hz(bz_tesla: f64, detuning_hz: f64) -> f32 {
    let b0_shift = 40.0 * ((bz_tesla.abs().clamp(0.0, 10.0) - 1.0) as f32);
    let detune_shift = 0.25 * (detuning_hz.clamp(-5000.0, 5000.0) as f32);
    clamp_freq_hz(660.0 + b0_shift + detune_shift)
}

#[inline]
fn is_rf_pulse_window(mode: MriPulseMode, t_s: f64, cfg: &RfPulseConfig) -> bool {
    match mode {
        MriPulseMode::HahnSpinEcho => {
            let half_te = 0.5 * cfg.echo_time_te.max(0.010);
            let half_tau = 0.5 * cfg.duration_tau.max(0.0005);
            t_s <= half_tau || (t_s - half_te).abs() <= half_tau
        }
        MriPulseMode::InversionPulse => t_s <= cfg.duration_tau.max(0.0005),
        MriPulseMode::ContinuousField => false,
    }
}

#[inline]
fn pulse_side_voice(freq_hz: f32, gain: f32, pan: f32, disp: f32) -> VoiceSpec {
    VoiceSpec {
        freq_hz: clamp_freq_hz(freq_hz),
        gain: gain.clamp(0.0, 0.55),
        pan,
        fm_mod_hz: 18.0 + 40.0 * disp,
        fm_index: 0.25 + 1.2 * disp,
        filter_cutoff_hz: clamp_freq_hz(800.0 + 2400.0 * (1.0 - 0.5 * disp)),
        transient_impulse: 0.0,
    }
}

#[inline]
fn echo_refocus_boost(current_time: f64, echo_tau: Option<f64>) -> f32 {
    let Some(tau) = echo_tau else {
        return 0.0;
    };
    if !tau.is_finite() || tau <= 0.0 {
        return 0.0;
    }
    let dt = (current_time - 2.0 * tau) / 0.012;
    (-(dt * dt)).exp().clamp(0.0, 1.0) as f32
}

#[inline]
fn gradient_chirp_fm(k_mag: f64, grad_phase_rad: f64) -> (f32, f32) {
    if !k_mag.is_finite() || k_mag <= 1e-3 {
        return (10.0, 0.10);
    }
    let phase_mod = if grad_phase_rad.is_finite() {
        grad_phase_rad.sin().abs() as f32
    } else {
        0.0
    };
    let fm_hz = (28.0 + 0.75 * (k_mag as f32)).clamp(28.0, 2800.0);
    let fm_idx =
        (0.65 + 2.6 * ((k_mag / 1400.0).clamp(0.0, 1.0) as f32) + 0.45 * phase_mod).clamp(0.4, 5.5);
    (fm_hz, fm_idx)
}

fn kspace_gradient_voice(
    base_hz: f32,
    k_mag: f64,
    input: &MriKSpaceAudioInput,
    grad_fm_hz: f32,
    grad_fm_idx: f32,
) -> VoiceSpec {
    if !k_mag.is_finite() || k_mag <= 1e-3 {
        return VoiceSpec::silent();
    }
    let k_norm = (k_mag / 1800.0).clamp(0.0, 1.0) as f32;
    let pan = (input.k_coord_cycles_per_m[0] / k_mag.max(1.0)).clamp(-0.9, 0.9) as f32;
    let phase_bend = if input.selected_grad_phase_rad.is_finite() {
        24.0 * (input.selected_grad_phase_rad.cos() as f32)
    } else {
        0.0
    };
    VoiceSpec {
        freq_hz: clamp_freq_hz(base_hz * 1.5 + 360.0 * k_norm + phase_bend),
        gain: (0.10 + 0.34 * k_norm).clamp(0.08, 0.46),
        pan,
        fm_mod_hz: grad_fm_hz,
        fm_index: grad_fm_idx,
        filter_cutoff_hz: clamp_freq_hz(1500.0 + 4200.0 * k_norm),
        transient_impulse: if input.rf_spike_triggered { 0.8 } else { 0.0 },
    }
}

#[cfg(feature = "biology")]
fn turing_tissue_harmonics(
    turing_uv: [f64; 2],
    rho: f64,
    t2: f64,
    coherence: f32,
) -> (HarmonicWeights, f32) {
    let u = if turing_uv[0].is_finite() {
        turing_uv[0].clamp(0.02, 4.0)
    } else {
        rho.clamp(0.1, 1.0)
    };
    let v = if turing_uv[1].is_finite() {
        turing_uv[1].clamp(0.02, 4.0)
    } else {
        (1.0 - 0.5 * rho).clamp(0.1, 1.0)
    };
    let kinetics = SchnakenbergKinetics { a: 0.10, b: 0.90 };
    let rates = <SchnakenbergKinetics as ReactionKinetics<2>>::reaction(&kinetics, [u, v]);
    let flux = (rates[0].hypot(rates[1]) * 0.6).clamp(0.0, 1.0) as f32;
    let t2_norm = (t2.clamp(0.02, 0.35) / 0.35) as f32;
    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    sin_b[0] = 1.0;
    sin_b[1] = 0.42 * coherence + 0.25 * flux;
    sin_b[2] = 0.26 * t2_norm + 0.18 * (u as f32).min(1.5);
    cos_a[3] = 0.16 * flux;
    sin_b[4] = 0.12 * (1.0 - t2_norm);
    (
        HarmonicWeights { cos_a, sin_b }.normalized(),
        (0.45 + 0.35 * t2_norm + 0.20 * flux).clamp(0.25, 1.0),
    )
}

#[cfg(not(feature = "biology"))]
fn turing_tissue_harmonics(
    _turing_uv: [f64; 2],
    rho: f64,
    t2: f64,
    coherence: f32,
) -> (HarmonicWeights, f32) {
    let rho_f = rho.clamp(0.1, 1.0) as f32;
    let t2_norm = (t2.clamp(0.02, 0.35) / 0.35) as f32;
    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    sin_b[0] = 1.0;
    sin_b[1] = 0.40 * coherence;
    sin_b[2] = 0.24 * rho_f;
    cos_a[3] = 0.12 * t2_norm;
    (
        HarmonicWeights { cos_a, sin_b }.normalized(),
        (0.45 + 0.45 * t2_norm).clamp(0.25, 1.0),
    )
}
