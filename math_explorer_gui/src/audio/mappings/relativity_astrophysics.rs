//! Relativistic redshift, `FourVector` Doppler shift, and synchrotron beaming sonification mappings.

use math_explorer::physics::astrophysics::{OrbitParticleKind, OrbitalState};
use math_explorer::physics::high_energy::{
    inverse_compton_spectral_index, synchrotron_power, FourVector, C, SIGMA_T,
};

use crate::audio::frame::{
    clamp_freq_hz, SonificationFrame, VoiceSpec, WaveformPreset, MAX_VOICES,
};

/// Stack-resident kinematic and metric telemetry for a single geodesic ray or relativistic probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicVoiceSample {
    /// Proper-to-coordinate clock rate $d\tau / dt$ or gravitational frequency factor.
    pub clock_rate: f64,
    /// Local three-velocity $\boldsymbol{\beta} = \mathbf{v} / c$ (`|beta| < 1`).
    pub source_beta: [f64; 3],
    /// Unit direction vector from emitter toward the distant observer.
    pub line_of_sight: [f64; 3],
    /// Dimensionless local magnetic energy density scale $U_B \ge 0$.
    pub magnetic_u_b: f64,
    /// Stereo pan in `[-1.0, 1.0]` derived from projected viewport coordinate.
    pub pan: f32,
    /// `true` once the geodesic halts at a horizon or coordinate singularity guard.
    pub halted: bool,
}

/// Maps up to 16 curved-spacetime geodesics into an 8-voice polyphonic [`SonificationFrame`]
/// using gravitational clock dilation, [`FourVector::relativistic_doppler_factor`], and
/// [`synchrotron_power`]. Returns [`SonificationFrame::silent`] when all rays halt at the horizon guard.
#[must_use]
pub fn geodesic_bundle_frame(
    rays: &[GeodesicVoiceSample],
    waveform: WaveformPreset,
) -> SonificationFrame {
    let mut active_indices = [0usize; MAX_VOICES];
    let active_count = select_active_geodesics(rays, &mut active_indices);
    if active_count == 0 {
        return SonificationFrame::silent();
    }

    let mut frame = SonificationFrame::silent();
    frame.waveform = waveform;
    frame.active_voices = active_count as u8;
    frame.master_gain = 0.76;

    let voice_scale = 1.0 / (active_count as f32).sqrt();
    for slot in 0..active_count {
        let sample = &rays[active_indices[slot]];
        frame.voices[slot] = build_geodesic_voice(sample, slot, active_count, voice_scale);
    }
    frame.sanitize()
}

fn select_active_geodesics(
    rays: &[GeodesicVoiceSample],
    out_indices: &mut [usize; MAX_VOICES],
) -> usize {
    let mut valid = [0usize; 16];
    let mut n_valid = 0;
    for (idx, r) in rays.iter().take(16).enumerate() {
        if !r.halted && r.clock_rate.is_finite() && r.clock_rate > 1e-4 {
            valid[n_valid] = idx;
            n_valid += 1;
        }
    }
    if n_valid == 0 {
        return 0;
    }
    let count = n_valid.min(MAX_VOICES);
    for (slot, dst) in out_indices.iter_mut().take(count).enumerate() {
        let pick = (slot * n_valid) / count;
        *dst = valid[pick];
    }
    count
}

fn build_geodesic_voice(
    sample: &GeodesicVoiceSample,
    slot: usize,
    total_slots: usize,
    voice_scale: f32,
) -> VoiceSpec {
    let beta = clamp_subluminal_beta(sample.source_beta, 0.94);
    let los = normalize_los(sample.line_of_sight);
    let doppler = FourVector::new(1.0, los[0], los[1], los[2])
        .relativistic_doppler_factor(beta)
        .unwrap_or(1.0)
        .clamp(0.08, 12.0);
    let beta_sq = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
    let gamma = (1.0 / (1.0 - beta_sq).max(1e-4).sqrt()).max(1.0);
    let u_b = if sample.magnetic_u_b.is_finite() {
        sample.magnetic_u_b.max(0.0)
    } else {
        0.0
    };
    let raw_sync = synchrotron_power(u_b, beta_sq.sqrt(), gamma).unwrap_or(0.0);
    let sync_ref = ((4.0 / 3.0) * SIGMA_T * C).max(1e-30);
    let beaming = ((raw_sync / sync_ref) / (1.0 + raw_sync / sync_ref)).clamp(0.0, 1.0) as f32;
    let spectral_alpha =
        inverse_compton_spectral_index(2.2 + 0.6 * f64::from(beaming)).unwrap_or(0.6) as f32;

    let spread = if total_slots > 1 {
        0.85 + 0.35 * (slot as f64) / ((total_slots - 1) as f64)
    } else {
        1.0
    };
    let grav = sample.clock_rate.clamp(0.04, 1.6);
    let carrier_hz = clamp_freq_hz((220.0 * spread * grav * doppler) as f32);
    let gain = ((0.24 + 0.56 * beaming) * (grav.sqrt() as f32) * voice_scale).clamp(0.03, 0.85);
    let cutoff =
        clamp_freq_hz(480.0 + (4800.0 + 900.0 * spectral_alpha) * beaming + 0.8 * carrier_hz);

    VoiceSpec {
        freq_hz: carrier_hz,
        gain,
        pan: sample.pan.clamp(-1.0, 1.0),
        fm_mod_hz: clamp_freq_hz(carrier_hz * 0.5),
        fm_index: (0.15 + 1.6 * beaming).clamp(0.1, 2.4),
        filter_cutoff_hz: cutoff,
        transient_impulse: 0.0,
    }
}

/// Maps planar [`GalacticDiskSystem`](math_explorer::physics::astrophysics::GalacticDiskSystem)
/// tracers (prioritizing custom `SlingshotProbe` and `TidalStream` stars) to an 8-voice
/// Doppler-shifted and synchrotron-beamed [`SonificationFrame`].
#[must_use]
pub fn galactic_orbits_frame(
    particles: &[OrbitalState],
    outer_v_circ_kms: f64,
    disk_scale_kpc: f64,
) -> SonificationFrame {
    if particles.is_empty() {
        return SonificationFrame::silent();
    }
    let mut selected = [0usize; MAX_VOICES];
    let count = select_galactic_tracers(particles, &mut selected);
    if count == 0 {
        return SonificationFrame::silent();
    }

    let v_scale = outer_v_circ_kms.abs().max(25.0) * 1.85;
    let r_scale = disk_scale_kpc.abs().max(0.35);
    let mut frame = SonificationFrame::silent();
    frame.waveform = WaveformPreset::FourierWarm;
    frame.active_voices = count as u8;
    frame.master_gain = 0.74;

    let voice_scale = 1.0 / (count as f32).sqrt();
    for slot in 0..count {
        let p = &particles[selected[slot]];
        frame.voices[slot] = build_galactic_voice(p, slot, v_scale, r_scale, voice_scale);
    }
    frame.sanitize()
}

fn select_galactic_tracers(particles: &[OrbitalState], out: &mut [usize; MAX_VOICES]) -> usize {
    let mut count = 0;
    for kind in [
        OrbitParticleKind::SlingshotProbe,
        OrbitParticleKind::TidalStream,
    ] {
        for (idx, p) in particles.iter().enumerate() {
            if count >= MAX_VOICES {
                return count;
            }
            if !p.captured && p.kind == kind && p.x_kpc.is_finite() && p.vx_kms.is_finite() {
                out[count] = idx;
                count += 1;
            }
        }
    }
    let remaining = MAX_VOICES.saturating_sub(count);
    if remaining > 0 {
        let mut disk_indices = [0usize; 64];
        let mut n_disk = 0;
        for (idx, p) in particles.iter().take(64).enumerate() {
            if !p.captured && p.kind == OrbitParticleKind::DiskTracer && p.x_kpc.is_finite() {
                disk_indices[n_disk] = idx;
                n_disk += 1;
            }
        }
        let take_disk = remaining.min(n_disk);
        for k in 0..take_disk {
            out[count] = disk_indices[(k * n_disk) / take_disk];
            count += 1;
        }
    }
    count
}

fn build_galactic_voice(
    p: &OrbitalState,
    slot: usize,
    v_scale: f64,
    r_scale: f64,
    voice_scale: f32,
) -> VoiceSpec {
    let beta = clamp_subluminal_beta([p.vx_kms / v_scale, p.vy_kms / v_scale, 0.0], 0.86);
    // Edge-inclined observer looking along +x so approaching/receding sides Doppler-shift
    let los = [0.866_025_4, 0.0, 0.5];
    let doppler = FourVector::new(1.0, los[0], los[1], los[2])
        .relativistic_doppler_factor(beta)
        .unwrap_or(1.0)
        .clamp(0.25, 4.0);
    let beta_sq = beta[0] * beta[0] + beta[1] * beta[1];
    let gamma = (1.0 / (1.0 - beta_sq).max(1e-4).sqrt()).max(1.0);
    let r_kpc = (p.x_kpc * p.x_kpc + p.y_kpc * p.y_kpc).sqrt().max(0.05);
    let u_b = 2.5 * (-r_kpc / (2.2 * r_scale)).exp();
    let raw_sync = synchrotron_power(u_b, beta_sq.sqrt(), gamma).unwrap_or(0.0);
    let sync_ref = ((4.0 / 3.0) * SIGMA_T * C).max(1e-30);
    let beaming = ((raw_sync / sync_ref) / (0.35 + raw_sync / sync_ref)).clamp(0.0, 1.0) as f32;
    let spectral_alpha = inverse_compton_spectral_index(2.4).unwrap_or(0.7) as f32;

    let base_chord = [165.0_f64, 220.0, 275.0, 330.0, 396.0, 440.0, 550.0, 660.0];
    let kind_mult = match p.kind {
        OrbitParticleKind::SlingshotProbe => 1.25,
        OrbitParticleKind::TidalStream => 1.0,
        OrbitParticleKind::DiskTracer => 0.75,
    };
    let freq_hz = clamp_freq_hz((base_chord[slot % MAX_VOICES] * kind_mult * doppler) as f32);
    let pan = (p.x_kpc / (4.5 * r_scale)).clamp(-1.0, 1.0) as f32;

    VoiceSpec {
        freq_hz,
        gain: ((0.22 + 0.58 * beaming) * voice_scale).clamp(0.04, 0.82),
        pan,
        fm_mod_hz: clamp_freq_hz(freq_hz * 0.5),
        fm_index: (0.12 + 1.2 * beaming).clamp(0.1, 2.0),
        filter_cutoff_hz: clamp_freq_hz(650.0 + (4000.0 + 900.0 * spectral_alpha) * beaming),
        transient_impulse: 0.0,
    }
}

fn clamp_subluminal_beta(raw: [f64; 3], max_mag: f64) -> [f64; 3] {
    if !raw.iter().all(|v| v.is_finite()) {
        return [0.0, 0.0, 0.0];
    }
    let mag_sq = raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2];
    if mag_sq <= max_mag * max_mag {
        return raw;
    }
    let scale = max_mag / mag_sq.sqrt().max(1e-12);
    [raw[0] * scale, raw[1] * scale, raw[2] * scale]
}

fn normalize_los(raw: [f64; 3]) -> [f64; 3] {
    if !raw.iter().all(|v| v.is_finite()) {
        return [1.0, 0.0, 0.0];
    }
    let norm_sq = raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2];
    if norm_sq <= 1e-12 {
        return [1.0, 0.0, 0.0];
    }
    let inv = 1.0 / norm_sq.sqrt();
    [raw[0] * inv, raw[1] * inv, raw[2] * inv]
}
