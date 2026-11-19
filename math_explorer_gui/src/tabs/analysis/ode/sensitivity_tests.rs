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
    assert!(
        !out_exp_idle.viewport_output[&egui::ViewportId::ROOT]
            .repaint_delay
            .is_zero()
    );

    // HarmonicOscillator has 3 parameters (k, y0, v0) -> converges in 2 frames
    tool.preset = OdePreset::HarmonicOscillator;
    tool.recalculate();
    let out_osc_1 = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(!tool.sensitivity_engine.is_converged());
    assert!(
        out_osc_1.viewport_output[&egui::ViewportId::ROOT]
            .repaint_delay
            .is_zero()
    );

    let _ = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    let out_osc_idle = ctx.run(compact_input(), |ctx| tool.show(ctx));
    assert!(
        !out_osc_idle.viewport_output[&egui::ViewportId::ROOT]
            .repaint_delay
            .is_zero()
    );
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
    assert!(
        !out_idle.viewport_output[&egui::ViewportId::ROOT]
            .repaint_delay
            .is_zero()
    );
}
