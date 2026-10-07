#![cfg_attr(any(), verified(opt_out = "tests"))]
use super::*;
use eframe::egui;

#[test]
fn test_algorithm_visualizer_defaults() {
    let tool = AlgorithmVisualizerTool::default();
    assert!(!tool.is_playing);
    assert_eq!(tool.playback_speed, 2.0);
    assert_eq!(tool.step_timer, 0.0);
    assert_eq!(tool.animation_step, 0);
}

#[test]
fn test_algorithm_visualizer_timer_progression() {
    let ctx = egui::Context::default();
    let mut tool = AlgorithmVisualizerTool {
        is_playing: true,
        playback_speed: 2.0, // 0.5 sec per step
        ..Default::default()
    };

    let raw_input = egui::RawInput {
        predicted_dt: 0.6,
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        tool.show(ctx);
    });

    assert_eq!(tool.animation_step, 1);
    assert!(tool.is_playing);
}

#[test]
fn test_algorithm_visualizer_stop_at_end() {
    let ctx = egui::Context::default();
    let mut tool = AlgorithmVisualizerTool {
        is_playing: true,
        ..Default::default()
    };
    tool.animation_step = tool.visit_order.len();

    let raw_input = egui::RawInput {
        predicted_dt: 1.0,
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        tool.show(ctx);
    });

    assert!(!tool.is_playing);
    assert_eq!(tool.animation_step, tool.visit_order.len());
    assert_eq!(tool.step_timer, 0.0);
}

#[test]
fn test_run_algorithm_resets_playback_state() {
    let mut tool = AlgorithmVisualizerTool {
        is_playing: true,
        step_timer: 0.4,
        animation_step: 3,
        ..Default::default()
    };

    tool.run_algorithm();

    assert!(!tool.is_playing);
    assert_eq!(tool.step_timer, 0.0);
    assert_eq!(tool.animation_step, 0);
}
