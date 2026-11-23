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
