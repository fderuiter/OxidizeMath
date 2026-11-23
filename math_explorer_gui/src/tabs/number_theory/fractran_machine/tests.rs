#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{ExportFormat, ExportStatusSeverity, SimulationFramework};

#[test]
fn test_primegame_breakpoints_and_frame_cap() {
    let mut tool = FractranMachineTool::default();
    assert_eq!(
        tool.run_until_power_of_two(),
        MachineStopReason::HitPowerOfTwo(2)
    );
    assert_eq!(tool.step_count, 19);
    assert_eq!(
        tool.run_until_power_of_two(),
        MachineStopReason::HitPowerOfTwo(3)
    );
    assert_eq!(tool.step_count, 69);
    assert_eq!(
        tool.run_until_power_of_two(),
        MachineStopReason::HitPowerOfTwo(5)
    );
    assert_eq!(tool.step_count, 281);
    assert_eq!(
        tool.run_until_power_of_two(),
        MachineStopReason::HitPowerOfTwo(7)
    );
    assert_eq!(tool.step_count, 710);
    assert_eq!(
        tool.run_until_power_of_two(),
        MachineStopReason::HitPowerOfTwo(11)
    );
    assert_eq!(tool.step_count, 2375);
    assert_eq!(
        tool.run_until_power_of_two(),
        MachineStopReason::HitPowerOfTwo(13)
    );
    let step_at_13 = tool.step_count;
    assert!(step_at_13 > 2375);

    // Reaching 2^17 from 2^13 requires > 4096 steps, so one frame hits the 4096-step cap and pauses.
    assert_eq!(
        tool.run_until_power_of_two(),
        MachineStopReason::FrameCapReached
    );
    assert_eq!(
        tool.step_count,
        step_at_13 + MAX_BREAKPOINT_STEPS_PER_FRAME as u64
    );
    assert!(!tool.is_playing);
}

#[test]
fn test_presets_validation_overflow_and_state_roundtrip() {
    let mut tool = FractranMachineTool::default();
    tool.select_preset(PresetChoice::Mult);
    while tool.step_once() == MachineStopReason::Running {}
    assert_eq!(tool.stop_reason, MachineStopReason::Halted);

    // Reject zero/negative and malformed custom fractions without corrupting state
    let valid_before = tool.save_state().unwrap();
    assert!(tool
        .try_load_state(
            r#"{"preset":"Custom","fractions_input":"-3/2","initial_value":2,"steps_per_frame":1,"stop_on_power_of_two":true}"#
        )
        .is_err());
    assert!(tool
        .try_load_state(
            r#"{"preset":"Custom","fractions_input":"10007/2","initial_value":2,"steps_per_frame":1,"stop_on_power_of_two":true}"#
        )
        .is_err());
    assert_eq!(tool.save_state().unwrap(), valid_before);

    // Roundtrip valid custom state
    let custom_json = r#"{"preset":"Fibonacci","fractions_input":"17/65, 133/34, 17/19, 23/17, 2233/69, 23/29, 31/23, 74/341, 31/37, 41/31, 129/287, 41/43, 13/41, 1/13, 1/3","initial_value":75,"steps_per_frame":8,"stop_on_power_of_two":false}"#;
    tool.try_load_state(custom_json).unwrap();
    let saved = tool.save_state().unwrap();
    let expected: serde_json::Value = serde_json::from_str(custom_json).unwrap();
    let actual: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(actual, expected);

    // Verify u32::MAX overflow pauses cleanly
    tool.try_load_state(
        r#"{"preset":"Custom","fractions_input":"4/2","initial_value":2,"steps_per_frame":1,"stop_on_power_of_two":false}"#,
    )
    .unwrap();
    tool.registers[0] = u32::MAX;
    assert_eq!(
        tool.step_once(),
        MachineStopReason::Overflow {
            rule_index: 0,
            prime: 2
        }
    );
    assert!(!tool.is_playing);
}

#[test]
fn test_fractran_csv_json_png_svg_and_gif_exports() {
    let mut tool = FractranMachineTool::default();
    let _ = tool.run_until_power_of_two();

    let mut framework = SimulationFramework::new("number_theory");
    framework.export_options.width = 360;
    framework.export_options.height = 240;
    framework.export_gif_frames = 4;
    framework.active_tool = Some(Box::new(tool));

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let raw = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        ..Default::default()
    };

    for format in [
        ExportFormat::Csv,
        ExportFormat::Json,
        ExportFormat::Svg,
        ExportFormat::Png,
        ExportFormat::Gif,
    ] {
        let _ = ctx.run(raw(), |ctx| {
            framework.show_export_modal = true;
            framework.queue_export(format);
            framework.show(ctx, "fractran_export_test");
        });
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Success,
            "Export format {format:?} failed: {:?}",
            framework.export_status_message
        );
    }
}

#[test]
fn test_state_revision_tracks_persisted_fields_and_ignores_runtime_steps() {
    let mut tool = FractranMachineTool::default();
    let rev0 = tool.state_revision();
    assert!(rev0 > 0);

    let _ = tool.step_once();
    assert_eq!(tool.state_revision(), rev0);

    tool.steps_per_frame = 8;
    let rev1 = tool.state_revision();
    assert!(rev1 > 0 && rev1 != rev0);

    tool.stop_on_power_of_two = !tool.stop_on_power_of_two;
    let rev2 = tool.state_revision();
    assert!(rev2 > 0 && rev2 != rev1);

    tool.select_preset(PresetChoice::Add);
    let rev3 = tool.state_revision();
    assert!(rev3 > 0 && rev3 != rev2);
}

#[test]
fn test_fractran_wav_midi_scala_exports_and_zero_alloc_pcm() {
    crate::audio::set_offline_duration_secs(1.0);
    let tool = FractranMachineTool::default();
    let wav = tool.export_wav_audio(44_100, 1.0).expect("FRACTRAN WAV");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(wav.len(), 44 + 44_100 * 4);
    let non_silent = wav[44..]
        .as_chunks::<2>()
        .0
        .iter()
        .any(|b| i16::from_le_bytes(*b).unsigned_abs() > 64);
    assert!(
        non_silent,
        "FRACTRAN OfflinePcmSynthesizer WAV must be audible"
    );

    let midi = tool.export_midi_score(1.5).expect("FRACTRAN MIDI");
    assert_eq!(&midi[0..4], b"MThd");
    assert_eq!(u16::from_be_bytes(midi[8..10].try_into().unwrap()), 1);

    let (scl, kbm) = tool.export_scala_tuning().expect("FRACTRAN Scala");
    assert!(scl.contains("3/2") && scl.contains("5/4") && scl.contains("2/1"));
    assert!(kbm.contains("220.000000"));

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut regs = tool.vm.encode_value(2).expect("initial regs");
    let mut scratch = [(0_u64, 0_u32); crate::audio::frame::MAX_VOICES];
    let mut synth = crate::audio::OfflinePcmSynthesizer::new(44_100, 735);
    let _ = crate::audio::SonificationBackend::set_muted_from_user_gesture(&mut synth, false);
    let mut pcm = vec![0.0_f32; 7_350 * 2];
    let (written, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        acoustics::render_fractran_pcm_into(
            &tool.vm,
            &mut regs,
            &mut scratch,
            &mut synth,
            735,
            2,
            &mut pcm,
        )
    });
    assert_eq!(written, 7_350);
    assert!(pcm.iter().any(|s| s.abs() > 0.01));
}

#[test]
fn test_fractran_sonification_chords_pow2_chime_halt_mute_and_meter() {
    let mut tool = FractranMachineTool::default();
    let _ = tool.run_until_power_of_two();
    let pow2_frame = tool.current_sonification_frame();
    assert!(pow2_frame.is_audible());
    assert!(pow2_frame.voices[0].transient_impulse >= 0.85);

    tool.select_preset(PresetChoice::Fibonacci);
    let _ = tool.step_once();
    let fib_frame = tool.current_sonification_frame();
    assert!(fib_frame.is_audible());
    assert!(fib_frame.active_voices >= 2);

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (frame_check, _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut last = crate::audio::SonificationFrame::silent();
        for _ in 0..200 {
            last = tool.current_sonification_frame();
        }
        last
    });
    assert_eq!(frame_check, fib_frame);
    drop(_alloc_guard);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let out = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    let ak = out.platform_output.accesskit_update.unwrap();
    assert!(ak
        .nodes
        .iter()
        .any(|(_, n)| n.role() == egui::accesskit::Role::Meter));

    tool.select_preset(PresetChoice::Mult);
    if crate::audio::snapshot_ui_state().muted {
        crate::audio::toggle_mute_from_user_gesture();
    }
    tool.is_playing = true;
    while tool.step_once() == MachineStopReason::Running {}
    assert_eq!(tool.stop_reason, MachineStopReason::Halted);
    assert_eq!(
        tool.current_sonification_frame(),
        crate::audio::SonificationFrame::silent()
    );
    assert!(
        crate::audio::snapshot_ui_state().muted,
        "Halting during live FRACTRAN playback must suspend audio cleanly"
    );
    crate::audio::export_current_tool_wav("fractran_machine");
}
