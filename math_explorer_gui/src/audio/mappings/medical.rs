//! Medical physics sonification mappings: 60 GHz radar respiratory Doppler gating
//! and 4D coplanar IMRT LINAC pulse delivery with OAR overdose roughness.

use crate::audio::frame::{clamp_freq_hz, SonificationFrame, VoiceSpec, WaveformPreset};

/// Feature-independent LINAC gating phase mirrored from `domain_physics` `BeamState`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LinacGatePhase {
    /// Controller dormant; beam emission disabled.
    #[default]
    Standby,
    /// Controller armed and tracking thoracic phase.
    Armed,
    /// LINAC actively delivering radiation pulses inside gating window.
    BeamActive,
    /// Delivery held while respiratory motion exceeds gating threshold.
    BeamHold,
    /// Emergency interlock latched on non-finite sensor fault.
    SafetyInterlock,
}

#[cfg(feature = "physics")]
impl From<math_explorer::physics::medical::radar_gating::BeamState> for LinacGatePhase {
    fn from(state: math_explorer::physics::medical::radar_gating::BeamState) -> Self {
        use math_explorer::physics::medical::radar_gating::BeamState;
        match state {
            BeamState::Standby => Self::Standby,
            BeamState::Armed => Self::Armed,
            BeamState::BeamActive => Self::BeamActive,
            BeamState::BeamHold => Self::BeamHold,
            BeamState::SafetyInterlock => Self::SafetyInterlock,
        }
    }
}

#[inline]
const fn voice(m: [f32; 3], s: [f32; 4]) -> VoiceSpec {
    VoiceSpec {
        freq_hz: m[0],
        gain: m[1],
        pan: m[2],
        fm_mod_hz: s[0],
        fm_index: s[1],
        filter_cutoff_hz: s[2],
        transient_impulse: s[3],
    }
}

/// Synthesizes a polyphonic [`SonificationFrame`] for `RadarGatingTool` mapping thoracic
/// respiratory Doppler displacement/velocity and LINAC macro-pulse cadence during `BeamActive`.
#[must_use]
pub fn radar_linac_frame(
    gate: impl Into<LinacGatePhase>,
    disp_mm: f64,
    vel_mm_s: f64,
    duty_cycle_pct: f64,
    sim_time_s: f64,
) -> SonificationFrame {
    let phase = gate.into();
    let inactive = matches!(
        phase,
        LinacGatePhase::Standby | LinacGatePhase::SafetyInterlock
    );
    if inactive || !disp_mm.is_finite() || !vel_mm_s.is_finite() {
        return SonificationFrame::silent();
    }
    let dn = (disp_mm.clamp(-15.0, 15.0) / 15.0) as f32;
    let vn = (vel_mm_s.clamp(-40.0, 40.0) / 40.0) as f32;
    let duty = if duty_cycle_pct.is_finite() {
        duty_cycle_pct.clamp(0.0, 100.0) as f32
    } else {
        0.0
    };
    let pulse = sim_time_s.is_finite() && (sim_time_s.max(0.0) * 10.0).fract() < 0.30;

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::PatchClamp;
    frame.active_voices = 3;
    frame.master_gain = 0.74;
    let (f0, g0) = (
        clamp_freq_hz(210.0 + dn * 65.0 + vn * 55.0),
        0.28 + 0.16 * vn.abs(),
    );
    let (m0, i0) = (6.0 + 18.0 * vn.abs(), 0.30 + 0.90 * dn.abs());
    let c0 = clamp_freq_hz(950.0 + 1800.0 * (0.5 + 0.5 * dn));
    frame.voices[0] = voice([f0, g0, -0.28], [m0, i0, c0, 0.0]);
    if phase == LinacGatePhase::BeamActive {
        let (f1, c1) = (
            clamp_freq_hz(440.0 + 1.4 * duty),
            clamp_freq_hz(2400.0 + 16.0 * duty),
        );
        let (g1, p1) = if pulse { (0.62, 0.85) } else { (0.26, 0.0) };
        frame.voices[1] = voice([f1, g1, 0.22], [14.0 + 0.30 * duty, 1.15, c1, p1]);
    } else if phase == LinacGatePhase::BeamHold {
        let f2 = clamp_freq_hz(277.18 + 30.0 * dn.max(0.0));
        frame.voices[2] = voice([f2, 0.20, 0.35], [5.0, 0.45, 1100.0, 0.0]);
    }
    frame.sanitize()
}

/// Synthesizes a polyphonic [`SonificationFrame`] for `DoseCalculationTool` mapping
/// `beam_loading_energy` LINAC pulse delivery, PTV coverage harmony, and OAR overdose roughness.
#[must_use]
pub fn imrt_delivery_frame(
    gate: impl Into<LinacGatePhase>,
    disp_mm: f64,
    ptv_mean_gy: f64,
    prescription_gy: f64,
    oar_max_gy: f64,
    oar_limit_gy: f64,
    beam_energy_mev: f64,
) -> SonificationFrame {
    let phase = gate.into();
    let inactive = matches!(
        phase,
        LinacGatePhase::Standby | LinacGatePhase::SafetyInterlock
    );
    let inputs_ok = disp_mm.is_finite()
        && ptv_mean_gy.is_finite()
        && oar_max_gy.is_finite()
        && beam_energy_mev.is_finite();
    if inactive || !inputs_ok {
        return SonificationFrame::silent();
    }
    let rx = (ptv_mean_gy / prescription_gy.max(1.0)).clamp(0.0, 1.0) as f32;
    let od = ((oar_max_gy - oar_limit_gy) / oar_limit_gy.max(1.0)).clamp(0.0, 2.0) as f32;
    let es = ((beam_energy_mev - 5.925).clamp(-2.5, 0.5) * 85.0) as f32;
    let dn = (disp_mm.clamp(-15.0, 15.0) / 15.0) as f32;

    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::FourierWarm;
    frame.active_voices = 4;
    frame.master_gain = 0.74;
    if phase == LinacGatePhase::BeamActive {
        let f0 = clamp_freq_hz(330.0 + es);
        let (g0, c0) = (
            (0.34 + 0.28 * rx).clamp(0.15, 0.72),
            clamp_freq_hz(2100.0 + 900.0 * rx),
        );
        let (f1, g1) = (
            clamp_freq_hz(f0 * 1.5),
            (0.16 + 0.24 * rx).clamp(0.08, 0.44),
        );
        frame.voices[0] = voice([f0, g0, -0.20], [16.0, 0.75, c0, 0.60]);
        frame.voices[1] = voice([f1, g1, 0.20], [8.0, 0.25, 2600.0, 0.0]);
    }
    frame.voices[2] = voice(
        [clamp_freq_hz(196.0 + 48.0 * dn), 0.22, -0.35],
        [4.5, 0.30, 1200.0, 0.0],
    );
    if od > 0.0 {
        apply_overdose_roughness(&mut frame, od);
    }
    frame.sanitize()
}

fn apply_overdose_roughness(frame: &mut SonificationFrame, od: f32) {
    let (f3, g3) = (
        clamp_freq_hz(466.16 + 95.0 * od),
        (0.18 + 0.34 * od).clamp(0.0, 0.68),
    );
    let c3 = clamp_freq_hz(1800.0 + 2200.0 * od);
    frame.voices[3] = voice([f3, g3, 0.40], [39.0 + 52.0 * od, 2.2 + 2.8 * od, c3, 0.0]);
    frame.noise_gain = (0.14 * od).clamp(0.0, 0.32);
    frame.noise_cutoff_hz = clamp_freq_hz(1400.0 + 1800.0 * od);
}
// [cite:mmwave_radiotherapy_setup]
