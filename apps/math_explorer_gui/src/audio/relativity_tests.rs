#![cfg_attr(any(), verified(opt_out = "tests"))]
#![cfg(feature = "physics")]

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

fn verify_tool_wav_and_settings(domain: &str, tool_id: &str) {
    use super::{AcousticSettings, WAV_HEADER_BYTES};
    use crate::framework::{ExportFormat, SimulationFramework};

    let ctx = eframe::egui::Context::default();
    let custom = AcousticSettings {
        gain: 0.72,
        pitch_scale: 1.35,
    };
    let sub_dir = std::env::temp_dir().join(format!(
        "oxidize_relativity_wav_{tool_id}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&sub_dir);
    std::fs::create_dir_all(&sub_dir).expect("create temp wav dir");

    let mut fw = SimulationFramework::new(domain);
    fw.export_queue.destination_dir = Some(sub_dir.clone());
    assert!(fw.select_tool_by_name(tool_id) && fw.is_audio_muted());
    fw.set_acoustic_settings(custom);

    let preset_json = fw
        .active_tool
        .as_ref()
        .unwrap()
        .export_preset_json()
        .unwrap();
    let mut restored = SimulationFramework::new(domain);
    assert!(restored.select_tool_by_name(tool_id));
    restored
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&preset_json)
        .expect("preset round-trip");
    assert!(restored.is_audio_muted());
    let cfg = restored.active_tool.as_ref().unwrap().acoustic_settings();
    assert!((cfg.gain - custom.gain).abs() < 1e-5);
    assert!((cfg.pitch_scale - custom.pitch_scale).abs() < 1e-5);

    fw.queue_export(ExportFormat::Wav);
    fw.dispatch_pending_exports_with_ctx(&ctx);
    let wav_path = std::fs::read_dir(&sub_dir)
        .expect("read wav dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .unwrap_or_else(|| panic!("Missing exported .wav file for {tool_id}"));
    let bytes = std::fs::read(&wav_path).expect("read exported wav");
    let _ = std::fs::remove_dir_all(&sub_dir);
    assert!(bytes.len() > WAV_HEADER_BYTES && &bytes[0..4] == b"RIFF");
    let energy: f64 = bytes[WAV_HEADER_BYTES..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|s| f64::from(i16::from_le_bytes(*s)).powi(2) / (32_768.0 * 32_768.0))
        .sum();
    assert!(energy > 0.5, "{tool_id} WAV energy too low: {energy}");
}

#[test]
fn test_relativity_and_galactic_tools_framework_wav_export_and_acoustic_settings() {
    verify_tool_wav_and_settings("relativity", "GeodesicPlaygroundTool");
    verify_tool_wav_and_settings("astrophysics", "GalacticOrbitSimulator");
}
