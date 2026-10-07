#![cfg_attr(any(), verified(opt_out = "unit_tests"))]

use super::*;

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
    let mut tool = AlgorithmVisualizerTool::default();
    tool.is_playing = true;
    tool.playback_speed = 2.0; // 0.5 sec per step

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
    let mut tool = AlgorithmVisualizerTool::default();
    tool.animation_step = tool.visit_order.len();
    tool.is_playing = true;

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
    let mut tool = AlgorithmVisualizerTool::default();
    tool.is_playing = true;
    tool.step_timer = 0.4;
    tool.animation_step = 3;

    tool.run_algorithm();

    assert!(!tool.is_playing);
    assert_eq!(tool.step_timer, 0.0);
    assert_eq!(tool.animation_step, 0);
}
