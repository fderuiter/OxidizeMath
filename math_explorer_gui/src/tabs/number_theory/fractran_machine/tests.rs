#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{ExportFormat, ExportStatusSeverity, SimulationFramework};

#[test]
fn test_primegame_breakpoints_and_frame_cap() {
    let mut tool = FractranMachineTool::default();
    assert_eq!(tool.run_until_power_of_two(), MachineStopReason::HitPowerOfTwo(2));
    assert_eq!(tool.step_count, 19);
    assert_eq!(tool.run_until_power_of_two(), MachineStopReason::HitPowerOfTwo(3));
    assert_eq!(tool.step_count, 69);
    assert_eq!(tool.run_until_power_of_two(), MachineStopReason::HitPowerOfTwo(5));
    assert_eq!(tool.step_count, 281);
    assert_eq!(tool.run_until_power_of_two(), MachineStopReason::HitPowerOfTwo(7));
    assert_eq!(tool.step_count, 710);
    assert_eq!(tool.run_until_power_of_two(), MachineStopReason::HitPowerOfTwo(11));
    assert_eq!(tool.step_count, 2375);
    assert_eq!(tool.run_until_power_of_two(), MachineStopReason::HitPowerOfTwo(13));
    let step_at_13 = tool.step_count;
    assert!(step_at_13 > 2375);

    // Reaching 2^17 from 2^13 requires > 4096 steps, so one frame hits the 4096-step cap and pauses.
    assert_eq!(tool.run_until_power_of_two(), MachineStopReason::FrameCapReached);
    assert_eq!(tool.step_count, step_at_13 + MAX_BREAKPOINT_STEPS_PER_FRAME as u64);
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
