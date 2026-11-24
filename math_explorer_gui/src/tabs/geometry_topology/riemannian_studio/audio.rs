#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Curvature-modulated polyphonic [`SonificationFrame`] synthesis for [`super::RiemannianPdeStudio`].

use super::mesh_gen::ManifoldGeometry;
use super::solver::RiemannianSimulationEngine;
use super::state::PdeRegime;
use crate::audio::frame::{
    HarmonicWeights, SonificationFrame, VoiceSpec, WaveformPreset, NUM_HARMONICS,
};

/// Synthesizes a stack-resident [`SonificationFrame`] from the active manifold curvature,
/// scalar PDE state, and wavefront velocity without heap allocations.
#[must_use]
pub fn synthesize_manifold_frame(
    geom: &ManifoldGeometry,
    engine: &RiemannianSimulationEngine,
    regime: PdeRegime,
) -> SonificationFrame {
    let n = geom.positions.len().min(engine.scalar_field.len());
    if n == 0 {
        return SonificationFrame::silent();
    }

    let stats = compute_field_and_curvature_moments(geom, engine, n);
    let base_waveform = match regime {
        PdeRegime::CorticalHodgkinHuxley => WaveformPreset::PatchClamp,
        PdeRegime::SchnakenbergTuring | PdeRegime::GrayScottTuring => WaveformPreset::QuantumModal,
        PdeRegime::MeanCurvatureSmoothing => WaveformPreset::FourierWarm,
        PdeRegime::HeatBeltrami | PdeRegime::FreeSurferMorphometryGlm => {
            WaveformPreset::JustHarmonic
        }
    };

    let carrier_hz = map_regime_carrier_hz(regime, stats.norm_mean, stats.mean_h);
    let fm_index = (stats.flux_rms * 18.0 + stats.lap_rms * 0.45).clamp(0.05, 6.0) as f32;
    let cutoff_hz =
        (520.0 + 3600.0 * stats.spread + 1800.0 * stats.flux_rms).clamp(240.0, 6800.0) as f32;
    let pan = (stats.centroid_x * 0.65).clamp(-0.85, 0.85) as f32;

    let mut voices = [VoiceSpec::silent(); 8];
    voices[0] = VoiceSpec {
        freq_hz: carrier_hz,
        gain: 0.48,
        pan,
        fm_mod_hz: carrier_hz * 1.5,
        fm_index,
        filter_cutoff_hz: cutoff_hz,
        transient_impulse: engine.transient_impulse.clamp(0.0, 1.0),
    };
    voices[1] = VoiceSpec {
        freq_hz: (carrier_hz * 1.498).clamp(40.0, 7200.0),
        gain: (0.22 + 0.16 * stats.curv_mod as f32).clamp(0.08, 0.40),
        pan: -pan * 0.8,
        fm_mod_hz: carrier_hz * 0.5,
        fm_index: fm_index * 0.55,
        filter_cutoff_hz: (cutoff_hz * 1.15).clamp(240.0, 7600.0),
        transient_impulse: 0.0,
    };
    voices[2] = VoiceSpec {
        freq_hz: (carrier_hz * 2.0).clamp(60.0, 7800.0),
        gain: (0.12 + 0.14 * stats.spread as f32).clamp(0.06, 0.28),
        pan: (stats.centroid_y * 0.55).clamp(-0.75, 0.75) as f32,
        fm_mod_hz: carrier_hz,
        fm_index: fm_index * 0.35,
        filter_cutoff_hz: cutoff_hz,
        transient_impulse: 0.0,
    };

    let harmonics = build_curvature_harmonics(stats.mean_h, stats.spread, stats.flux_rms);
    SonificationFrame {
        voices,
        active_voices: 3,
        waveform: base_waveform,
        custom_harmonics: Some(harmonics),
        noise_gain: (stats.flux_rms as f32 * 0.12).clamp(0.0, 0.18),
        noise_cutoff_hz: cutoff_hz,
        master_gain: 0.68,
    }
    .sanitize()
}

struct ManifoldAudioMoments {
    norm_mean: f64,
    spread: f64,
    mean_h: f64,
    curv_mod: f64,
    flux_rms: f64,
    lap_rms: f64,
    centroid_x: f64,
    centroid_y: f64,
}

fn compute_field_and_curvature_moments(
    geom: &ManifoldGeometry,
    engine: &RiemannianSimulationEngine,
    n: usize,
) -> ManifoldAudioMoments {
    let mut min_u = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut area_sum = 1e-12;
    let mut weighted_u = 0.0;
    let mut weighted_h = 0.0;
    let mut flux_sq = 0.0;
    let mut lap_sq = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;

    for i in 0..n {
        let u = engine.scalar_field[i];
        let prev = engine.prev_field.get(i).copied().unwrap_or(u);
        if !u.is_finite() {
            continue;
        }
        min_u = min_u.min(u);
        max_u = max_u.max(u);
        let a = geom.laplacian.vertex_area(i);
        let h = geom.mean_curvature.get(i).copied().unwrap_or(0.0).abs();
        area_sum += a;
        weighted_u += a * u;
        weighted_h += a * h;
        let du = u - prev;
        flux_sq += a * du * du;
        let lu = geom.laplacian.apply_at(&engine.scalar_field, i);
        if lu.is_finite() {
            lap_sq += a * lu * lu;
        }
        let weight = a * (u.abs() + 0.05);
        cx += weight * geom.positions[i][0];
        cy += weight * geom.positions[i][1];
    }

    if !min_u.is_finite() || !max_u.is_finite() {
        min_u = 0.0;
        max_u = 1.0;
    }
    let span = (max_u - min_u).max(1e-6);
    let mean_u = weighted_u / area_sum;
    let norm_mean = ((mean_u - min_u) / span).clamp(0.0, 1.0);
    let mean_h = (weighted_h / area_sum).clamp(0.0, 6.0);
    ManifoldAudioMoments {
        norm_mean,
        spread: (span / (span + 1.5)).clamp(0.0, 1.0),
        mean_h,
        curv_mod: (mean_h / (mean_h + 1.0)).clamp(0.0, 1.0),
        flux_rms: (flux_sq / area_sum).sqrt().clamp(0.0, 4.0),
        lap_rms: (lap_sq / area_sum).sqrt().clamp(0.0, 8.0),
        centroid_x: (cx / (area_sum * (mean_u.abs() + 0.25))).clamp(-1.5, 1.5),
        centroid_y: (cy / (area_sum * (mean_u.abs() + 0.25))).clamp(-1.5, 1.5),
    }
}

fn map_regime_carrier_hz(regime: PdeRegime, norm_mean: f64, mean_h: f64) -> f32 {
    let base = match regime {
        PdeRegime::CorticalHodgkinHuxley => 140.0 + 380.0 * norm_mean,
        PdeRegime::SchnakenbergTuring | PdeRegime::GrayScottTuring => 165.0 + 290.0 * norm_mean,
        PdeRegime::MeanCurvatureSmoothing => 130.0 + 210.0 * (mean_h / (mean_h + 1.2)),
        PdeRegime::HeatBeltrami | PdeRegime::FreeSurferMorphometryGlm => 180.0 + 260.0 * norm_mean,
    };
    (base * (1.0 + 0.12 * mean_h.min(2.5))).clamp(65.0, 1760.0) as f32
}

fn build_curvature_harmonics(mean_h: f64, spread: f64, flux_rms: f64) -> HarmonicWeights {
    let mut cos_a = [0.0_f32; NUM_HARMONICS];
    let mut sin_b = [0.0_f32; NUM_HARMONICS];
    let tilt = (0.35 + 0.45 * spread + 0.25 * flux_rms).clamp(0.2, 1.4) as f32;
    let h_factor = (mean_h * 0.35).clamp(0.0, 1.2) as f32;
    for k in 0..NUM_HARMONICS {
        let n = (k + 1) as f32;
        sin_b[k] =
            (1.0 / n.powf(1.35 - 0.35 * tilt.min(1.1))) * (1.0 + 0.15 * (n * h_factor).sin());
        cos_a[k] = if k % 2 == 1 {
            0.22 * h_factor / n
        } else {
            0.08 * tilt / (n * n)
        };
    }
    HarmonicWeights { cos_a, sin_b }.normalized()
}
