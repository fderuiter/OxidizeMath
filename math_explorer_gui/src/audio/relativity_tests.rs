#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Unit and 1,000-step zero-allocation verification tests for relativistic Doppler, synchrotron
//! beaming, and horizon-guard sonification in `relativity_astrophysics`.

use verified_engine::engine::VerifiedEngine;

use super::frame::{WaveformPreset, LIMITER_CEILING_LINEAR};
use super::mappings::{galactic_orbits_frame, geodesic_bundle_frame, GeodesicVoiceSample};
use super::synth::OfflinePcmSynthesizer;
use super::{MuteReason, SonificationBackend};
use math_explorer::physics::astrophysics::{
    DwarfMorphology, GalacticDiskSystem, HaloProfile, Mpc, OrbitPopulationConfig,
};

#[test]
fn test_relativity_doppler_beaming_and_1000_step_zero_alloc() {
    let approaching = GeodesicVoiceSample {
        clock_rate: 0.85,
        source_beta: [0.55, 0.0, 0.0],
        line_of_sight: [1.0, 0.0, 0.0],
        magnetic_u_b: 2.5,
        pan: -0.4,
        halted: false,
    };
    let receding = GeodesicVoiceSample {
        source_beta: [-0.55, 0.0, 0.0],
        pan: 0.4,
        ..approaching
    };
    let f_app = geodesic_bundle_frame(&[approaching], WaveformPreset::FourierWarm);
    let f_rec = geodesic_bundle_frame(&[receding], WaveformPreset::FourierWarm);
    assert!((20.0..=8000.0).contains(&f_app.voices[0].freq_hz));
    assert!((20.0..=8000.0).contains(&f_rec.voices[0].freq_hz));
    assert!(f_app.voices[0].freq_hz > f_rec.voices[0].freq_hz * 1.8);

    let galaxy = GalacticDiskSystem::from_morphology(
        DwarfMorphology::TypeCode10,
        Mpc(12.0),
        0.6,
        0.8,
        HaloProfile::BurkertCore {
            virial_mass_solar: 1.2e10,
            core_radius_kpc: 2.2,
            concentration: 12.0,
        },
        OrbitPopulationConfig::default(),
    );
    let mut synth = OfflinePcmSynthesizer::new(44_100, 2048);
    let _ = synth.set_muted_from_user_gesture(false);

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (peak, _) = VerifiedEngine::run_verified(|| {
        let mut max_abs = 0.0_f32;
        for step in 0..1_000usize {
            let frame = if step % 2 == 0 {
                geodesic_bundle_frame(&[approaching, receding], WaveformPreset::QuantumModal)
            } else {
                galactic_orbits_frame(galaxy.particles(), 65.0, 0.8)
            };
            synth.apply_frame(&frame);
            for &s in synth.render_frames(16) {
                max_abs = max_abs.max(s.abs());
            }
        }
        max_abs
    });
    assert!(peak > 0.02 && peak <= LIMITER_CEILING_LINEAR + 1e-6);
}

#[test]
fn test_relativity_horizon_guard_and_pause_decay_below_minus_80dbfs_in_30ms() {
    let active = GeodesicVoiceSample {
        clock_rate: 0.9,
        source_beta: [0.5, 0.0, 0.0],
        line_of_sight: [1.0, 0.0, 0.0],
        magnetic_u_b: 3.0,
        pan: 0.0,
        halted: false,
    };
    let mut synth = OfflinePcmSynthesizer::new(44_100, 2048);
    let _ = synth.set_muted_from_user_gesture(false);
    let active_frame = geodesic_bundle_frame(&[active], WaveformPreset::FourierWarm);
    synth.apply_frame(&active_frame);
    let _ = synth.render_frames(512);

    let horizon_frame = geodesic_bundle_frame(
        &[GeodesicVoiceSample {
            halted: true,
            ..active
        }],
        WaveformPreset::FourierWarm,
    );
    synth.apply_frame(&horizon_frame);
    let tail_horizon = synth.render_frames(1400);
    let peak_horizon = tail_horizon[1323 * 2..]
        .iter()
        .copied()
        .fold(0.0_f32, |a, b| a.max(b.abs()));
    assert!(peak_horizon < 1.0e-4, "got {peak_horizon}");

    synth.apply_frame(&active_frame);
    let _ = synth.render_frames(512);
    synth.mute_and_suspend(MuteReason::Paused);
    let tail_pause = synth.render_frames(1400);
    let peak_pause = tail_pause[1323 * 2..]
        .iter()
        .copied()
        .fold(0.0_f32, |a, b| a.max(b.abs()));
    assert!(peak_pause < 1.0e-4, "got {peak_pause}");
}
