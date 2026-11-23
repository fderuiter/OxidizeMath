#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::PRESET_FRACTRAN_PRIMEGAME;
use super::state::{FractranMachineTool, FractranPreset, FractranToolConfig, MAX_STEPS_PER_FRAME};
use super::AlgorithmicInfoTab;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::tabs::ExplorerTab;
use eframe::egui;

#[test]
fn test_algorithmic_info_tab_domain_slug_and_tool_selection() {
    let mut tab = AlgorithmicInfoTab::default();
    assert_eq!(tab.name(), "Algorithmic Information");
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "algorithmic_information");
    assert_eq!(tab.active_tool_id(), Some("AlgorithmicFractranTool"));
    assert!(tab.select_tool_by_name("FRACTRAN Virtual Machine"));
}

#[test]
fn test_fractran_tool_headless_egui_render_and_prime_tape() {
    let mut tool = FractranMachineTool::default();
    assert_eq!(tool.emitted_primes, vec![2]);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| tool.show(ctx));
    assert!(out.platform_output.accesskit_update.is_some());

    tool.import_preset_json(PRESET_FRACTRAN_PRIMEGAME).unwrap();
    assert_eq!(tool.emitted_primes, vec![2, 3]);
    assert!(tool.trajectory.len() >= 70);

    let datasets = tool.export_datasets();
    let opts = egui_plot::export::ExportOptions {
        width: 1200,
        height: 720,
        ..Default::default()
    };
    let png = egui_plot::export::export_custom_png(&datasets, &opts).unwrap();
    assert!(!png.is_empty());
}

#[test]
fn test_floyd_cycle_detection_auto_pauses_live_playback() {
    let mut tool = FractranMachineTool::default();
    let cfg = FractranToolConfig {
        playback_rate: 1.0,
        steps_per_frame: MAX_STEPS_PER_FRAME,
        warmup_steps: 0,
        initial_value: 2,
        preset: FractranPreset::Custom,
        custom_fractions: "3/2, 2/3".to_string(),
    };
    tool.apply_config(cfg).unwrap();
    tool.is_playing = true;
    tool.step_frame_batch();

    assert!(
        !tool.is_playing,
        "Live playback must pause automatically upon Floyd cycle detection"
    );
    assert_eq!(tool.live.cycle.map(|c| c.period), Some(2));
}

#[test]
fn test_custom_fraction_validation_and_finite_state_guard_rollback() {
    let mut tool = FractranMachineTool::default();
    let baseline = tool.save_state().unwrap();

    assert!(tool
        .try_load_state(r#"{"playback_rate":1.0,"steps_per_frame":16,"warmup_steps":0,"initial_value":2,"preset":"Custom","custom_fractions":"5/0"}"#)
        .is_err());
    assert!(tool
        .try_load_state(r#"{"playback_rate":1.0,"steps_per_frame":16,"warmup_steps":0,"initial_value":2,"preset":"Custom","custom_fractions":"-3/2"}"#)
        .is_err());
    assert!(tool
        .try_load_state(r#"{"playback_rate":1.0,"steps_per_frame":16,"warmup_steps":0,"initial_value":2,"preset":"Custom","custom_fractions":"313/2"}"#)
        .is_err());
    assert!(tool
        .try_load_state(r#"{"playback_rate":1.0,"steps_per_frame":512,"warmup_steps":0,"initial_value":2,"preset":"Primegame","custom_fractions":"3/2"}"#)
        .is_err());
    assert_eq!(tool.save_state().unwrap(), baseline);

    let mut framework = SimulationFramework::new("algorithmic_information");
    framework.cache_valid_snapshot();
    framework
        .active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<FractranMachineTool>()
        .unwrap()
        .config
        .playback_rate = f64::NAN;
    assert!(framework.inspect_runtime_state().is_err());
    assert!(framework.rollback_step_back());
    assert!(!framework.is_diverged());
    assert_eq!(
        framework
            .active_tool
            .as_ref()
            .unwrap()
            .save_state()
            .unwrap(),
        baseline
    );
}

// [cite:algorithmic_information_rust]
