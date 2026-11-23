#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Stack-resident seismic crackle and modal resonance sonification for Abelian sandpile avalanches.

use crate::audio::frame::{
    SonificationFrame, VoiceSpec, WaveformPreset, LIMITER_CEILING_LINEAR, MAX_VOICES,
};

/// Synthesizes a zero-allocation [`SonificationFrame`] mapping sandpile wavefront topples,
/// normalized avalanche area, cascade duration, and power-law slope into seismic crackle
/// and harmonic lattice modes strictly bounded within `-3.0 dBFS` ([`LIMITER_CEILING_LINEAR`]).
#[must_use]
pub fn sandpile_sonification_frame(
    wavefront_sites: usize,
    last_topples: u64,
    last_area: usize,
    grid_cells: usize,
    power_law_slope: f64,
) -> SonificationFrame {
    let cells_f = (grid_cells.max(1)) as f32;
    let wave_intensity = ((wavefront_sites as f32) / cells_f.sqrt().max(1.0)).clamp(0.0, 1.0);
    let area_ratio = ((last_area as f32) / cells_f).clamp(0.0, 1.0);
    let log_size = ((last_topples.max(1) as f32).ln() / 8.0).clamp(0.0, 1.0);
    let activity = (0.65 * wave_intensity + 0.35 * log_size).clamp(0.0, 1.0);

    if wavefront_sites == 0 && last_topples == 0 {
        return SonificationFrame::silent();
    }

    let slope_mag = if power_law_slope.is_finite() {
        (power_law_slope.abs() as f32).clamp(0.05, 2.0)
    } else {
        0.30
    };

    // Map larger avalanches to deeper seismic fundamentals (65 Hz .. 520 Hz) for 1/f texture.
    let base_hz = (440.0 * (1.0 - 0.72 * area_ratio.sqrt()) + 65.0).clamp(65.0, 620.0);
    let cutoff_hz = (520.0 + 3_600.0 * wave_intensity + 700.0 * slope_mag).clamp(180.0, 6_400.0);
    let impulse = (0.25 + 0.75 * wave_intensity).clamp(0.0, 1.0);

    let mut voices = [VoiceSpec::silent(); MAX_VOICES];
    voices[0] = VoiceSpec {
        freq_hz: base_hz,
        gain: 0.24 * (0.25 + 0.75 * activity),
        pan: -0.25,
        fm_mod_hz: base_hz * 1.414,
        fm_index: 0.8 + 2.4 * wave_intensity,
        filter_cutoff_hz: cutoff_hz,
        transient_impulse: if wavefront_sites > 0 { impulse } else { 0.0 },
    };
    voices[1] = VoiceSpec {
        freq_hz: (base_hz * 1.5).clamp(90.0, 4_000.0),
        gain: 0.14 * activity,
        pan: 0.25,
        fm_mod_hz: base_hz * 0.5,
        fm_index: 0.5 + 1.5 * area_ratio,
        filter_cutoff_hz: cutoff_hz * 0.85,
        transient_impulse: 0.0,
    };
    voices[2] = VoiceSpec {
        freq_hz: (base_hz * 2.0 + 90.0 * slope_mag).clamp(120.0, 5_000.0),
        gain: 0.08 * wave_intensity,
        pan: 0.0,
        fm_mod_hz: 0.0,
        fm_index: 0.0,
        filter_cutoff_hz: cutoff_hz,
        transient_impulse: 0.0,
    };

    let noise_gain = (0.06 + 0.22 * wave_intensity + 0.08 * log_size).clamp(0.0, 0.32);
    let master_gain = ((0.22 + 0.46 * activity).min(LIMITER_CEILING_LINEAR)).clamp(0.0, LIMITER_CEILING_LINEAR);

    SonificationFrame {
        voices,
        active_voices: 3,
        waveform: WaveformPreset::PatchClamp,
        custom_harmonics: None,
        noise_gain,
        noise_cutoff_hz: cutoff_hz,
        master_gain,
    }
    .sanitize()
}
