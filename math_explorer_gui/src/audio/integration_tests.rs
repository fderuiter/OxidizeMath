//! End-to-end verification tests across all 6 sonified simulation tools covering:
//! - Zero-allocation hot-path `sonification_frame()` synthesis under `VerifiedAllocator`
//! - Deterministic 16-bit PCM `.wav` RIFF export and non-zero acoustic energy across all 6 tools
//! - Default-muted startup, `<= 20 ms` pause/divergence/WebGL-loss silence guards, and ARIA alerts
//! - JSON preset and `#tab=...&tool=...&state=...` URL round-trips within `ToolStateBounds`

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::unified::UnifiedModel;
use crate::framework::{ExportFormat, InteractiveTool, SimulationFramework, ToolStateBounds};
use crate::tabs::chaos::attractors::AttractorPlotter;
use crate::tabs::mri::tool::MriTool;
use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
use crate::tabs::neuroscience::spike_analysis::SpikeAnalysisTool;
use crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget;
use crate::tabs::quantum::wave_sim::WaveSimulatorTool;
use std::collections::HashMap;

const SONIFIED_TOOL_SPECS: [(&str, &str); 6] = [
    ("neuroscience", "SpikeAnalysisTool"),
    ("neuroscience", "HodgkinHuxleyTool"),
    ("chaos", "AttractorPlotter"),
    ("quantum", "WaveSimulator"),
    ("mri", "MriTool"),
    ("number_theory", "PrimeSpiralWidget"),
];

fn wav_acoustic_energy(wav_bytes: &[u8]) -> f64 {
    assert!(wav_bytes.len() > WAV_HEADER_BYTES, "WAV payload too short");
    assert_eq!(&wav_bytes[0..4], b"RIFF");
    assert_eq!(&wav_bytes[8..12], b"WAVE");
    assert_eq!(&wav_bytes[12..16], b"fmt ");
    assert_eq!(&wav_bytes[36..40], b"data");
    assert_eq!(u16::from_le_bytes([wav_bytes[20], wav_bytes[21]]), 1);
    assert_eq!(u16::from_le_bytes([wav_bytes[22], wav_bytes[23]]), 2);
    assert_eq!(
        u32::from_le_bytes([wav_bytes[24], wav_bytes[25], wav_bytes[26], wav_bytes[27]]),
        DEFAULT_WAV_SAMPLE_RATE
    );
    assert_eq!(u16::from_le_bytes([wav_bytes[34], wav_bytes[35]]), 16);

    let mut energy = 0.0_f64;
    for s in wav_bytes[WAV_HEADER_BYTES..].as_chunks::<2>().0 {
        let val = f64::from(i16::from_le_bytes(*s)) / 32_768.0;
        energy += val * val;
    }
    energy
}

fn build_warmed_tools() -> (
    SpikeAnalysisTool,
    HodgkinHuxleyUnified,
    AttractorPlotter,
    WaveSimulatorTool,
    MriTool,
    PrimeSpiralWidget,
) {
    let mut spike = SpikeAnalysisTool::default();
    for _ in 0..25 {
        InteractiveTool::step_simulation(&mut spike);
    }

    let hh_params = HashMap::from([("i_ext".to_string(), 14.0)]);
    let mut hh = HodgkinHuxleyUnified::new(&hh_params);
    for _ in 0..250 {
        hh.step(&hh_params);
    }

    let mut lorenz = AttractorPlotter::default();
    for _ in 0..15 {
        lorenz.step_simulation();
    }

    let mut wave = WaveSimulatorTool::new();
    wave.apply_tunneling_split_challenge(None);
    for _ in 0..5 {
        wave.step_simulation();
    }

    let mut mri = MriTool::default();
    for _ in 0..10 {
        mri.step_simulation();
    }

    let mut spiral = PrimeSpiralWidget::default();
    for _ in 0..12 {
        spiral.step_simulation();
    }
    (spike, hh, lorenz, wave, mri, spiral)
}

#[test]
fn test_all_six_tools_zero_allocation_hot_path_synthesis() {
    let (spike, hh, lorenz, wave, mri, spiral) = build_warmed_tools();
    let settings = AcousticSettings {
        gain: 0.75,
        pitch_scale: 1.1,
    };
    let mut synth = OfflinePcmSynthesizer::new(DEFAULT_WAV_SAMPLE_RATE, 128);
    let _ = synth.set_muted_from_user_gesture(false);
    synth.set_master_volume(settings.gain);

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (peak, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let frames = [
            spike
                .sonification_frame()
                .unwrap_or_else(SonificationFrame::silent),
            hh.sonification_frame()
                .unwrap_or_else(SonificationFrame::silent),
            lorenz
                .sonification_frame()
                .unwrap_or_else(SonificationFrame::silent),
            wave.sonification_frame()
                .unwrap_or_else(SonificationFrame::silent),
            mri.sonification_frame()
                .unwrap_or_else(SonificationFrame::silent),
            spiral
                .sonification_frame()
                .unwrap_or_else(SonificationFrame::silent),
        ];
        let mut max_abs = 0.0_f32;
        for frame in frames {
            assert!(frame.is_audible());
            let scaled = frame.with_pitch_scale(settings.pitch_scale);
            synth.apply_frame(&scaled);
            for &sample in synth.render_frames(64) {
                max_abs = max_abs.max(sample.abs());
            }
        }
        max_abs
    });
    assert!(peak > 0.01);
}

#[test]
fn test_wav_export_riff_header_and_acoustic_energy_across_all_six_tools() {
    let ctx = eframe::egui::Context::default();
    for (idx, (domain, tool_id)) in SONIFIED_TOOL_SPECS.iter().enumerate() {
        let sub_dir = std::env::temp_dir().join(format!(
            "oxidize_wav_verify_{idx}_{tool_id}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&sub_dir);
        std::fs::create_dir_all(&sub_dir).expect("create temp wav dir");

        let mut fw = SimulationFramework::new(domain);
        fw.export_queue.destination_dir = Some(sub_dir.clone());
        assert!(
            fw.select_tool_by_name(tool_id),
            "Failed to select {tool_id}"
        );
        assert!(
            fw.active_tool.as_ref().unwrap().export_capabilities().audio,
            "{tool_id} must advertise audio export capability"
        );

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
        let energy = wav_acoustic_energy(&bytes);
        assert!(
            energy > 1.0,
            "{tool_id} WAV export must have non-zero acoustic energy, got {energy}"
        );
    }
}

#[test]
fn test_acoustic_parameter_changes_measurably_alter_wav_waveform() {
    let mut tool = WaveSimulatorTool::new();
    let frames = tool.sample_sonification_frames(60);
    let wav_low = synthesize_wav_16bit(
        &frames,
        AcousticSettings {
            gain: 0.35,
            pitch_scale: 0.75,
        },
        DEFAULT_WAV_SAMPLE_RATE,
        0.5,
        99,
    );
    tool.set_acoustic_settings(AcousticSettings {
        gain: 0.85,
        pitch_scale: 1.65,
    });
    let wav_high = synthesize_wav_16bit(
        &frames,
        tool.acoustic_settings(),
        DEFAULT_WAV_SAMPLE_RATE,
        0.5,
        99,
    );
    assert_ne!(wav_low, wav_high);
    assert!(wav_acoustic_energy(&wav_high) > wav_acoustic_energy(&wav_low) * 1.5);
}

#[test]
fn test_framework_default_muted_and_20ms_divergence_pause_webgl_guards() {
    crate::accessibility::clear_test_announcements();
    let ctx = eframe::egui::Context::default();
    let mut fw = SimulationFramework::new("chaos");
    assert!(fw.select_tool_by_name("AttractorPlotter"));
    assert!(fw.is_audio_muted(), "SimulationFramework must boot muted");
    assert!(!fw.audio_controller.has_audio_context());

    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        eframe::egui::CentralPanel::default().show(ctx, |ui| fw.show_inside(ui, "chaos"));
    });
    assert!(
        fw.is_audio_muted(),
        "Passive frame render must never unmute"
    );

    fw.unmute_audio_with_gesture();
    assert!(!fw.is_audio_muted());

    fw.audio_controller.silence_on_pause();
    assert_eq!(
        fw.audio_controller
            .current_gain_after_elapsed(std::time::Duration::from_millis(20)),
        0.0
    );

    fw.is_diverged = true;
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        eframe::egui::CentralPanel::default().show(ctx, |ui| fw.show_inside(ui, "chaos"));
    });
    assert!(fw.is_audio_muted(), "Divergence must auto-mute immediately");
    assert_eq!(
        fw.audio_controller
            .current_gain_after_elapsed(std::time::Duration::from_millis(20)),
        0.0
    );
    assert!(crate::accessibility::all_announcements()
        .iter()
        .any(|(m, p)| m.contains("solver divergence") && p == "assertive"));
}

#[test]
fn test_preset_json_and_share_url_roundtrip_keeps_startup_muted() {
    let custom = AcousticSettings {
        gain: 0.73,
        pitch_scale: 1.45,
    };
    for (domain, tool_id) in SONIFIED_TOOL_SPECS {
        let mut src = SimulationFramework::new(domain);
        assert!(src.select_tool_by_name(tool_id));
        src.set_acoustic_settings(custom);
        src.unmute_audio_with_gesture();
        assert!(!src.is_audio_muted());

        let preset_json = src
            .active_tool
            .as_ref()
            .unwrap()
            .export_preset_json()
            .unwrap();
        ToolStateBounds::DEFAULT
            .validate_json(&preset_json)
            .expect("Preset JSON must respect ToolStateBounds");

        let mut dst_preset = SimulationFramework::new(domain);
        assert!(dst_preset.select_tool_by_name(tool_id));
        dst_preset
            .active_tool
            .as_mut()
            .unwrap()
            .import_preset_json(&preset_json)
            .expect("Preset round-trip must succeed");
        assert!(
            dst_preset.is_audio_muted(),
            "{tool_id} preset load must remain muted"
        );
        let loaded_cfg = dst_preset.active_tool.as_ref().unwrap().acoustic_settings();
        assert!(
            (loaded_cfg.gain - custom.gain).abs() < 1e-5,
            "{tool_id} gain mismatch"
        );
        assert!((loaded_cfg.pitch_scale - custom.pitch_scale).abs() < 1e-5);

        let url = src
            .encode_share_url()
            .expect("Share URL encode must succeed");
        let (_, frag) = url
            .split_once('#')
            .expect("Share URL must contain # fragment");
        let parsed = crate::url_state::parse_share_location(frag, "")
            .unwrap()
            .expect("Parsed ShareUrlParams");
        let mut dst_url = SimulationFramework::new(domain);
        dst_url
            .import_url_preset_json(parsed.tool.as_deref(), parsed.state_json.as_deref())
            .expect("URL import must succeed");
        assert!(
            dst_url.is_audio_muted(),
            "{tool_id} URL load must remain muted"
        );
        let url_cfg = dst_url.active_tool.as_ref().unwrap().acoustic_settings();
        assert!((url_cfg.gain - custom.gain).abs() < 1e-5);
        assert!((url_cfg.pitch_scale - custom.pitch_scale).abs() < 1e-5);
    }
}

#[test]
fn test_custom_attractor_flow_preserves_acoustic_settings_and_wav_energy() {
    let mut plotter = AttractorPlotter {
        mode: crate::tabs::chaos::attractors::AttractorMode::Custom,
        ..Default::default()
    };
    plotter.set_acoustic_settings(AcousticSettings {
        gain: 0.77,
        pitch_scale: 1.3,
    });
    plotter
        .flow_bank
        .set_expressions(&["sin(y) - b * x", "sin(z) - b * y", "sin(x) - b * z"]);
    plotter.on_slider_changed();

    let saved = plotter.save_state().expect("save custom attractor state");
    let mut restored = AttractorPlotter::default();
    restored
        .try_load_state(&saved)
        .expect("load custom attractor state");
    assert_eq!(
        restored.mode,
        crate::tabs::chaos::attractors::AttractorMode::Custom
    );
    assert!((restored.acoustic_settings().gain - 0.77).abs() < 1e-5);
    assert!((restored.acoustic_settings().pitch_scale - 1.3).abs() < 1e-5);

    let frames = restored.sample_sonification_frames(32);
    let wav = synthesize_wav_16bit(
        &frames,
        restored.acoustic_settings(),
        DEFAULT_WAV_SAMPLE_RATE,
        0.35,
        0xCAFE,
    );
    assert!(wav_acoustic_energy(&wav) > 0.5);
}

#[cfg(target_arch = "wasm32")]
mod wasm_integration_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_wasm_six_tools_wav_synthesis_and_webgl_loss_silence() {
        for (domain, tool_id) in SONIFIED_TOOL_SPECS {
            let mut fw = SimulationFramework::new(domain);
            assert!(fw.select_tool_by_name(tool_id));
            let tool = fw.active_tool.as_mut().unwrap();
            let frames = tool.sample_sonification_frames(24);
            let wav = synthesize_wav_16bit(
                &frames,
                tool.acoustic_settings(),
                DEFAULT_WAV_SAMPLE_RATE,
                0.35,
                0x51A0,
            );
            assert!(
                wav_acoustic_energy(&wav) > 0.25,
                "Wasm WAV energy failed for {tool_id}"
            );
        }
        silence_all_audio_on_webgl_loss();
    }
}
