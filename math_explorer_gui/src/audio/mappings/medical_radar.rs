//! Zero-allocation 60 GHz FMCW Doppler, sub-Rayleigh MUSIC, and LINAC `BeamState` sonification.

// theory_verification!

use math_explorer::physics::medical::radar_gating::{BeamState, FmcwConfig, C};

use crate::audio::frame::{clamp_freq_hz, SonificationFrame, VoiceSpec, WaveformPreset};

/// Center carrier frequency (`60 GHz`) of the clinical MIMO respiratory radar.
pub const RADAR_CARRIER_60GHZ_HZ: f64 = 60.0e9;

/// Stack-only telemetry snapshot mapped into a [`SonificationFrame`] by [`medical_radar_frame`].
#[derive(Clone, Copy, Debug)]
pub struct RadarDopplerAudioInput {
    /// `FastCvKalman` radial thoracic velocity estimate `state[1]` in `m/s`.
    pub velocity_m_s: f64,
    /// `FastCvKalman` radial thoracic displacement estimate `state[0]` in `m`.
    pub amplitude_m: f64,
    /// FMCW sweep bandwidth $B$ in `GHz`.
    pub bandwidth_ghz: f64,
    /// Prominent range peaks resolved by the Rayleigh-limited coarse FFT.
    pub coarse_peak_count: usize,
    /// Prominent range peaks resolved by the low-rank `MUSIC` pseudospectrum (ADR 0025).
    pub music_peak_count: usize,
    /// ULA beamformer gain `[0.0, 1.0]` toward the primary thoracic target.
    pub primary_beam_gain: f64,
    /// ULA beamformer gain `[0.0, 1.0]` toward the secondary gown/clutter scatterer.
    pub secondary_beam_gain: f64,
    /// Remaining duration of an injected cough transient in seconds (`0.0..=0.45`).
    pub cough_timer_s: f64,
    /// Formal LINAC gating state machine status (ADR 0016).
    pub beam_state: BeamState,
}

/// Computes the monostatic 60 GHz Doppler shift $f_D = 2 f_c v / c$ in `Hz`.
#[inline]
#[must_use]
pub fn doppler_shift_60ghz_hz(velocity_m_s: f64) -> f64 {
    if !velocity_m_s.is_finite() {
        return 0.0;
    }
    (2.0 * RADAR_CARRIER_60GHZ_HZ * velocity_m_s) / C
}

/// Maps 60 GHz Kalman velocity, sub-Rayleigh `MUSIC` scatterers, cough spikes, and `BeamState`
/// to a stack-allocated [`SonificationFrame`] with zero heap allocations.
#[must_use]
pub fn medical_radar_frame(input: &RadarDopplerAudioInput) -> SonificationFrame {
    if input.beam_state == BeamState::SafetyInterlock
        || !input.velocity_m_s.is_finite()
        || !input.amplitude_m.is_finite()
        || !input.bandwidth_ghz.is_finite()
    {
        return SonificationFrame::silent();
    }

    let fmcw = FmcwConfig {
        bandwidth: input.bandwidth_ghz.clamp(0.25, 10.0) * 1.0e9,
        center_frequency: RADAR_CARRIER_60GHZ_HZ,
        chirp_time: 50.0e-6,
    };
    let rayleigh_cm = (fmcw.range_resolution() * 100.0).clamp(1.5, 60.0) as f32;
    let doppler_hz = (2.0 * fmcw.center_frequency * input.velocity_m_s) / C;
    let cough_norm = if input.cough_timer_s.is_finite() && input.cough_timer_s > 0.0 {
        (input.cough_timer_s / 0.45).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let carrier_hz =
        clamp_freq_hz(330.0 + (doppler_hz * 9.0).clamp(-210.0, 1400.0) as f32 + 280.0 * cough_norm);
    let (waveform, interval_ratio, master_gain, alert_fm) = beam_state_voicing(input.beam_state);

    let pri_gain = sanitize_unit(input.primary_beam_gain, 0.85);
    let sec_gain = sanitize_unit(input.secondary_beam_gain, 0.75);
    let lead_gain = (0.28 + 0.52 * pri_gain + 0.14 * cough_norm).clamp(0.24, 0.92);
    let cough_click = if cough_norm > 0.0 {
        (0.45 + 0.55 * cough_norm).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut frame = SonificationFrame::silent();
    frame.waveform = waveform;
    frame.master_gain = master_gain;
    frame.voices[0] = VoiceSpec {
        freq_hz: carrier_hz,
        gain: lead_gain,
        pan: -0.20,
        fm_mod_hz: (16.0 + doppler_hz.abs() as f32 * 2.4 + 90.0 * cough_norm).clamp(12.0, 900.0),
        fm_index: (alert_fm + 1.8 * cough_norm).clamp(0.10, 4.5),
        filter_cutoff_hz: clamp_freq_hz(1100.0 + 3600.0 * pri_gain + 1800.0 * cough_norm),
        transient_impulse: cough_click,
    };
    let coarse_weight = (input.coarse_peak_count.min(3) as f32) * 0.11;
    frame.voices[1] = VoiceSpec {
        freq_hz: clamp_freq_hz(carrier_hz * interval_ratio),
        gain: (0.10 + coarse_weight).min(lead_gain * 0.65),
        pan: 0.15,
        fm_mod_hz: (32.0 + 180.0 / rayleigh_cm).clamp(20.0, 420.0),
        fm_index: alert_fm,
        filter_cutoff_hz: clamp_freq_hz(1400.0 + 1800.0 * pri_gain),
        transient_impulse: if cough_norm > 0.5 { 0.5 } else { 0.0 },
    };

    populate_music_scatterers(&mut frame, input, carrier_hz, lead_gain, sec_gain);
    frame.sanitize()
}

fn populate_music_scatterers(
    frame: &mut SonificationFrame,
    input: &RadarDopplerAudioInput,
    carrier_hz: f32,
    lead_gain: f32,
    sec_gain: f32,
) {
    if input.music_peak_count >= 2 {
        let sub_rayleigh = input.music_peak_count > input.coarse_peak_count;
        let beat_hz =
            if sub_rayleigh { 11.5 } else { 5.5 } + (input.music_peak_count.min(4) as f32) * 2.0;
        let scatter_gain = (0.12 + 0.28 * sec_gain).min(lead_gain * 0.62);
        frame.voices[2] = scatterer_beat_voice(carrier_hz * 1.25 - beat_hz, scatter_gain, -0.72);
        frame.voices[3] = scatterer_beat_voice(carrier_hz * 1.25 + beat_hz, scatter_gain, 0.72);
        frame.active_voices = 4;
    } else {
        frame.active_voices = 2;
    }
}

#[inline]
const fn beam_state_voicing(state: BeamState) -> (WaveformPreset, f32, f32, f32) {
    match state {
        BeamState::BeamActive => (WaveformPreset::JustHarmonic, 1.5, 0.78, 0.20),
        BeamState::BeamHold => (
            WaveformPreset::SubharmonicSquare,
            std::f32::consts::SQRT_2,
            0.74,
            1.85,
        ),
        BeamState::Armed => (WaveformPreset::FourierWarm, 2.0, 0.58, 0.35),
        BeamState::Standby => (WaveformPreset::Sine, 1.0, 0.34, 0.10),
        BeamState::SafetyInterlock => (WaveformPreset::Sine, 1.0, 0.0, 0.0),
    }
}

#[inline]
fn scatterer_beat_voice(freq_hz: f32, gain: f32, pan: f32) -> VoiceSpec {
    VoiceSpec {
        freq_hz: clamp_freq_hz(freq_hz),
        gain,
        pan,
        fm_mod_hz: 14.0,
        fm_index: 0.30,
        filter_cutoff_hz: 2800.0,
        transient_impulse: 0.0,
    }
}

#[inline]
fn sanitize_unit(val: f64, fallback: f32) -> f32 {
    if val.is_finite() {
        val.clamp(0.0, 1.0) as f32
    } else {
        fallback
    }
}
