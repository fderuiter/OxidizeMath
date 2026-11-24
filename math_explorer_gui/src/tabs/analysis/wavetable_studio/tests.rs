#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

use eframe::egui::{vec2, Context, Pos2, RawInput, Rect};
use verified_engine::engine::VerifiedEngine;

use super::*;
use crate::app::MathExplorerApp;
use crate::audio::studio_preview::snapshot_studio_preview;
use crate::audio::wav::{inspect_wav_riff, WavSampleFormat, WavSampleRate};
use crate::audio::wavetable::{MAX_WAVETABLE_FRAMES, WAVETABLE_FRAME_SAMPLES};
use crate::framework::{ExportFormat, SimulationFramework};
use crate::starter_gallery::StarterPresetMetadata;

#[test]
fn test_zero_alloc_2048_sample_formula_and_pde_frame_evaluation() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut tool = WavetableStudioTool::default();
    tool.engine.rebuild_pde_field(&tool.state);
    let compiled = tool
        .expr_bar
        .active_compiled()
        .expect("default formula compiled")
        .clone();
    let params = tool.expr_bar.param_slice().to_vec();

    let (peak, _metrics) = VerifiedEngine::run_verified(|| {
        tool.engine
            .evaluate_formula_frame_zero_alloc(&compiled, &params, 0.45, 96);
        let f_peak = tool
            .engine
            .active_frame
            .iter()
            .fold(0.0_f32, |m, &s| m.max(s.abs()));
        tool.engine
            .evaluate_pde_slice_frame_zero_alloc(0.55, 0.4, 96);
        let p_peak = tool
            .engine
            .active_frame
            .iter()
            .fold(0.0_f32, |m, &s| m.max(s.abs()));
        f_peak.min(p_peak)
    });
    assert!(peak > 0.5, "both formula and PDE slices must be audible");
}

#[test]
fn test_wavetable_studio_wav_export_1_to_256_frames_chunks_and_zero_dc() {
    let mut tool = WavetableStudioTool::default();
    // Add an explicit DC offset (+1.25) to verify spectral DC removal (< 1e-6) on every frame.
    tool.expr_bar
        .set_expression("1.25 + sin(pi * x + (1.0 + 4.0 * m) * sin(2.0 * pi * x))");
    tool.sync_state_params_from_bar();

    for (frames, rate, fmt) in [
        (1_u16, WavSampleRate::Hz44100, WavSampleFormat::Pcm16),
        (16_u16, WavSampleRate::Hz48000, WavSampleFormat::Float32),
        (
            MAX_WAVETABLE_FRAMES as u16,
            WavSampleRate::Hz48000,
            WavSampleFormat::Float32,
        ),
    ] {
        tool.state.frame_count = frames;
        tool.state.sample_rate = rate;
        tool.state.sample_format = fmt;
        let wav = tool.synthesize_and_encode_wav();
        let summary = inspect_wav_riff(&wav).expect("valid wavetable RIFF");
        assert_eq!(summary.channels, 1);
        assert_eq!(summary.sample_rate_hz, rate.hz());
        assert_eq!(summary.format_tag, fmt.format_tag());
        let expected_samples = WAVETABLE_FRAME_SAMPLES * usize::from(frames);
        assert_eq!(
            summary.data_bytes,
            (expected_samples as u32) * fmt.bytes_per_sample()
        );
        assert_eq!(summary.extra_chunks.len(), 2);
        assert_eq!(&summary.extra_chunks[0].id, b"clm ");
        assert!(std::str::from_utf8(&summary.extra_chunks[0].payload)
            .unwrap()
            .starts_with("<!>2048 "));
        assert_eq!(&summary.extra_chunks[1].id, b"uhWT");

        if fmt == WavSampleFormat::Float32 {
            let p = summary.data_offset;
            for f_idx in 0..usize::from(frames) {
                let mut sum = 0.0_f64;
                for s_idx in 0..WAVETABLE_FRAME_SAMPLES {
                    let off = p + (f_idx * WAVETABLE_FRAME_SAMPLES + s_idx) * 4;
                    let sample = f32::from_le_bytes(wav[off..off + 4].try_into().unwrap());
                    assert!(sample.is_finite());
                    sum += sample as f64;
                }
                let mean = (sum / (WAVETABLE_FRAME_SAMPLES as f64)).abs();
                assert!(mean < 1.0e-6, "frame {f_idx} DC mean {mean} >= 1e-6");
            }
        }
    }
}

#[test]
fn test_wavetable_starter_presets_url_roundtrip_exports_and_audition_gate() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.tool_name == Some("WavetableStudioTool"))
        .collect();
    assert_eq!(presets.len(), 3);

    let temp_dir = std::env::temp_dir().join(format!("oxidize_wt_exports_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let ctx = Context::default();
    let raw = RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    };

    for preset in presets {
        let mut fw = SimulationFramework::new("analysis");
        fw.export_queue.destination_dir = Some(temp_dir.clone());
        fw.apply_starter_card(
            preset.title,
            preset.tool_name,
            Some(preset.generate_preset()),
        )
        .expect("starter preset loads");
        let _ = ctx.run(raw.clone(), |ctx| fw.show(ctx, "analysis"));
        for fmt in [ExportFormat::Svg, ExportFormat::Csv, ExportFormat::Json] {
            fw.queue_export(fmt);
            let _ = ctx.run(raw.clone(), |ctx| fw.show(ctx, "analysis"));
        }
    }
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Verify URL hash roundtrip + state validation + gesture audition gate.
    let mut app = MathExplorerApp::default();
    let idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "analysis")
        .unwrap();
    app.selected_tab = idx;
    assert!(app.tabs[idx].select_tool_by_name("WavetableStudioTool"));
    let url = app.encode_current_share_url().unwrap();
    let hash = &url[url.find('#').unwrap()..];
    let mut restored = MathExplorerApp::default();
    assert!(restored
        .apply_share_location_strings(hash, "", None)
        .unwrap());
    assert_eq!(
        restored.tabs[idx].active_tool_id(),
        Some("WavetableStudioTool")
    );

    let mut tool = WavetableStudioTool::default();
    tool.state.morph_pos = f64::NAN;
    assert!(!tool.has_finite_parameters());
    assert!(tool.clamp_parameters());
    assert!(tool.has_finite_parameters());

    tool.toggle_audition_from_gesture();
    assert!(snapshot_studio_preview().wavetable_auditioning);
    crate::audio::notify_playback_paused();
    assert!(!snapshot_studio_preview().wavetable_auditioning);
}

#[test]
fn test_wavetable_try_load_state_enforces_ast_instruction_and_param_limits() {
    let mut tool = WavetableStudioTool::default();
    let baseline = tool.save_state().expect("baseline state");

    let over_budget_expr = std::iter::repeat_n("sin(x + m)", 50)
        .collect::<Vec<_>>()
        .join(" + ");
    let mut parsed: serde_json::Value = serde_json::from_str(&baseline).unwrap();
    parsed["formula"] = serde_json::Value::String(over_budget_expr);
    let err = tool
        .try_load_state(&parsed.to_string())
        .expect_err("formula exceeding MAX_WAVETABLE_EXPR_INSTRUCTIONS must fail");
    assert!(
        err.contains("instructions") || err.contains("Invalid"),
        "unexpected error message: {err}"
    );
    assert_eq!(tool.save_state().as_deref(), Some(baseline.as_str()));

    let mut bad_frames: serde_json::Value = serde_json::from_str(&baseline).unwrap();
    bad_frames["frame_count"] = serde_json::json!(0);
    assert!(tool.try_load_state(&bad_frames.to_string()).is_err());

    let mut bad_param: serde_json::Value = serde_json::from_str(&baseline).unwrap();
    bad_param["formula_params"] = serde_json::json!({ "a": 999.0 });
    assert!(tool.try_load_state(&bad_param.to_string()).is_err());
    assert_eq!(tool.save_state().as_deref(), Some(baseline.as_str()));
}
