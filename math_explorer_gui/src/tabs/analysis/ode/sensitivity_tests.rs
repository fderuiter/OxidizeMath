#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

fn compact_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(960.0, 640.0),
        )),
        ..Default::default()
    }
}

#[test]
fn test_ode_sensitivity_converges_and_stops_repainting_across_presets() {
    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool::default();

    // Exponential has 2 parameters (k, y0) -> converges in 1 frame, then settles on idle pass
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    let out_exp_idle = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(!out_exp_idle.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());

    // HarmonicOscillator has 3 parameters (k, y0, v0) -> converges in 2 frames
    tool.preset = OdePreset::HarmonicOscillator;
    tool.recalculate();
    let out_osc_1 = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(!tool.sensitivity_engine.is_converged());
    assert!(out_osc_1.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());

    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    let out_osc_idle = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(!out_osc_idle.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());
}

#[test]
fn test_ode_sensitivity_invalidates_on_preset_solver_dt_time_and_custom_ast() {
    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        param_k: 0.2,
        ic_y0: 1.0,
        dt: 0.1,
        total_time: 2.0,
        ..Default::default()
    };
    tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    let baseline_rk4 = tool.sensitivity_engine.baseline_output;

    // 1. Solver method invalidation (RK4 -> Euler)
    tool.set_method(IntegrationMethod::Euler);
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    let baseline_euler = tool.sensitivity_engine.baseline_output;
    assert!(
        (baseline_euler - baseline_rk4).abs() > 1e-6,
        "Changing solver method must invalidate sensitivity cache"
    );

    // 2. Step size (dt) invalidation
    tool.dt = 0.01;
    tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    let baseline_small_dt = tool.sensitivity_engine.baseline_output;
    assert!(
        (baseline_small_dt - baseline_euler).abs() > 1e-5,
        "Changing dt must invalidate sensitivity cache"
    );

    // 3. Total time invalidation
    tool.total_time = 5.0;
    tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    let baseline_long_time = tool.sensitivity_engine.baseline_output;
    assert!(
        (baseline_long_time - baseline_small_dt).abs() > 0.1,
        "Changing total_time must invalidate sensitivity cache"
    );

    // 4. Preset invalidation (Exponential -> Custom) and 5. Custom AST invalidation
    tool.preset = OdePreset::Custom;
    tool.math_input.set_expression("-k * y");
    tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    let baseline_decay_ast = tool.sensitivity_engine.baseline_output;
    assert!(
        baseline_decay_ast < baseline_long_time,
        "Switching to Custom preset must invalidate sensitivity cache"
    );

    tool.math_input.set_expression("2 * k * y");
    tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    let baseline_growth_ast = tool.sensitivity_engine.baseline_output;
    assert!(
        baseline_growth_ast > baseline_decay_ast + 1.0,
        "Editing Custom AST expression must invalidate sensitivity cache"
    );
}

#[test]
fn test_ode_custom_multi_parameter_sensitivity_discovery_and_perturbation() {
    use std::collections::BTreeMap;

    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ic_y0: 1.5,
        dt: 0.05,
        total_time: 3.0,
        ..Default::default()
    };
    tool.math_input
        .set_expression("alpha * y - beta * |y| * sin^2(t)");
    tool.math_input.set_params_map(&BTreeMap::from([
        ("alpha".to_string(), 0.4),
        ("beta".to_string(), 0.8),
    ]));
    tool.recalculate();

    // 3 parameters (alpha, beta, y0) converge across 2 frames
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(!tool.sensitivity_engine.is_converged());
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());

    let names: Vec<&str> = tool
        .sensitivity_engine
        .metrics
        .iter()
        .map(|m| m.param_name.as_str())
        .collect();
    assert_eq!(names, vec!["alpha", "beta", "y0"]);
    assert!(tool.sensitivity_engine.metrics[0].partial_derivative > 0.0);
    assert!(tool.sensitivity_engine.metrics[1].partial_derivative < 0.0);
    assert!(tool.sensitivity_engine.metrics[2].elasticity > 0.0);
}

#[test]
fn test_ode_sensitivity_nan_initial_condition_converges_cleanly() {
    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        ic_y0: f64::NAN,
        ..Default::default()
    };
    tool.recalculate();

    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(!tool.sensitivity_engine.is_converged());

    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    let out_idle = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(!out_idle.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());
}

#[test]
fn test_ode_custom_logistic_preset_yields_finite_sensitivity_metrics() {
    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ic_y0: 2.0,
        dt: 0.05,
        total_time: 5.0,
        ..Default::default()
    };
    tool.math_input.set_expression("r * y * (1 - y / cap_k)");
    tool.recalculate();

    for _ in 0..3 {
        let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
        if tool.sensitivity_engine.is_converged() {
            break;
        }
    }
    assert!(tool.sensitivity_engine.is_converged());
    assert!(tool.sensitivity_engine.baseline_output.is_finite());

    let names: Vec<&str> = tool
        .sensitivity_engine
        .metrics
        .iter()
        .map(|m| m.param_name.as_str())
        .collect();
    assert_eq!(names, vec!["r", "cap_k", "y0"]);
    for metric in &tool.sensitivity_engine.metrics {
        assert!(
            metric.partial_derivative.is_finite() && metric.elasticity.is_finite(),
            "Non-finite sensitivity metric for {}: {metric:?}",
            metric.param_name
        );
        assert!(
            metric.partial_derivative > 0.0,
            "Expected positive logistic partial derivative for {}, got {}",
            metric.param_name,
            metric.partial_derivative
        );
    }
}

fn pointer_button_input(pressed: bool) -> egui::RawInput {
    let mut input = compact_input();
    input.events.push(egui::Event::PointerButton {
        pos: egui::pos2(820.0, 520.0),
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    });
    input
}

#[test]
fn test_ode_sweep_defers_perturbations_then_settles_on_pause() {
    use crate::reflective_ui::slider::{save_slider_state, AnimatedSliderState, SweepMode};

    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        param_k: 0.4,
        ic_y0: 1.0,
        dt: 0.05,
        total_time: 4.0,
        ..Default::default()
    };
    tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    let pre_sweep_baseline = tool.sensitivity_engine.baseline_output;

    let slider_id = egui::Id::new("ode_param_k_exp");
    save_slider_state(
        &ctx,
        slider_id,
        AnimatedSliderState {
            playing: true,
            direction: 1.0,
            sweep_mode: SweepMode::Increasing,
        },
    );
    for _ in 0..3 {
        let mut sweep_in = compact_input();
        sweep_in.predicted_dt = 0.05;
        let _ = ctx.run(sweep_in, |ctx| tool.show(ctx));
        assert!(tool.param_k > 0.4);
        assert_eq!(tool.sensitivity_engine.baseline_output, pre_sweep_baseline);
        assert_eq!(tool.sensitivity_engine.metrics[0].baseline_value, 0.4);
    }

    save_slider_state(
        &ctx,
        slider_id,
        AnimatedSliderState {
            playing: false,
            direction: 1.0,
            sweep_mode: SweepMode::Increasing,
        },
    );
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    assert_eq!(
        tool.sensitivity_engine.baseline_output.to_bits(),
        tool.y_series.last().expect("final y").to_bits()
    );
}

#[test]
fn test_ode_pointer_drag_defers_sensitivity_and_settles_on_release() {
    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        param_k: 0.5,
        ic_y0: 1.0,
        dt: 0.05,
        total_time: 4.0,
        ..Default::default()
    };
    tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    let settled_baseline = tool.sensitivity_engine.baseline_output;

    tool.param_k = 0.85;
    tool.recalculate();
    let _ = ctx.run(pointer_button_input(true), |ctx| tool.show(ctx));
    assert_eq!(tool.sensitivity_engine.baseline_output, settled_baseline);

    let _ = ctx.run(pointer_button_input(false), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    assert_eq!(
        tool.sensitivity_engine.baseline_output.to_bits(),
        tool.y_series.last().expect("released final y").to_bits()
    );
}

#[test]
fn test_ode_baseline_fallbacks_and_recovery_sensitivity_sync() {
    use std::collections::BTreeMap;

    let ctx = egui::Context::default();
    let mut custom_shadow = OdeSolverTool {
        preset: OdePreset::Custom,
        ic_y0: 2.0,
        dt: 0.1,
        total_time: 1.0,
        ..Default::default()
    };
    custom_shadow.math_input.set_expression("y0 * y");
    custom_shadow
        .math_input
        .set_params_map(&BTreeMap::from([("y0".to_string(), 0.5)]));
    custom_shadow.recalculate();
    let _ = ctx.run(compact_input(), |ctx| custom_shadow.show(ctx));
    assert!(custom_shadow.sensitivity_engine.is_converged());
    assert!(custom_shadow.sensitivity_engine.baseline_output.is_finite());

    custom_shadow.math_input.set_expression("x * y");
    custom_shadow.recalculate();
    let _ = ctx.run(compact_input(), |ctx| custom_shadow.show(ctx));
    assert!(custom_shadow.sensitivity_engine.is_converged());

    let mut div_tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        param_k: 1e200,
        ic_y0: 1e200,
        dt: 0.1,
        total_time: 5.0,
        ..Default::default()
    };
    div_tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| div_tool.show(ctx));
    assert!(!div_tool.sensitivity_engine.baseline_output.is_finite());

    let (mut changed, mut py, mut pv) = (false, Vec::new(), Vec::new());
    recovery::apply_recovery_action(
        &mut div_tool,
        crate::framework::recovery::RecoveryAction::ClampParameters,
        &mut changed,
        &mut py,
        &mut pv,
    );
    div_tool.run_sensitivity_analysis(false);
    assert!(div_tool.sensitivity_engine.is_converged());
    assert!(div_tool.sensitivity_engine.baseline_output.is_finite());
}

#[test]
fn test_ode_custom_gesture_timing_with_platform_instant() {
    use crate::web_io::PlatformInstant;

    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        preset: OdePreset::Custom,
        ic_y0: 1.0,
        dt: 0.001,
        total_time: 10.0,
        ..Default::default()
    };
    tool.math_input.set_expression("-k * y * sin(t)");
    tool.recalculate();
    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    let settled_before = tool.sensitivity_engine.baseline_output;

    let start = PlatformInstant::now();
    let _ = ctx.run(pointer_button_input(true), |ctx| tool.show(ctx));
    for step_k in [1.2, 1.4, 1.6] {
        tool.param_k = step_k;
        tool.recalculate();
        let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
        assert_eq!(tool.sensitivity_engine.baseline_output, settled_before);
    }
    let drag_elapsed = start.elapsed();

    let release_start = PlatformInstant::now();
    let _ = ctx.run(pointer_button_input(false), |ctx| tool.show(ctx));
    let release_elapsed = release_start.elapsed();
    assert!(tool.sensitivity_engine.is_converged());
    assert_eq!(
        tool.sensitivity_engine.baseline_output.to_bits(),
        tool.y_series.last().expect("final custom y").to_bits()
    );
    assert!(drag_elapsed.as_secs_f64() >= 0.0 && release_elapsed.as_secs_f64() >= 0.0);
}
