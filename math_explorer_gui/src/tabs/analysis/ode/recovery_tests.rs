#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::recovery::{
    emit_recovery_telemetry, render_recovery_action_buttons, render_recovery_banner,
    DiagnosticReport, FiniteStateGuard, RecoveryAction,
};
use crate::presets::OdePreset;
use eframe::egui;

#[test]
fn test_ode_structured_diagnostic_report_and_finite_state_guard() {
    let mut tool = OdeSolverTool::default();
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!(FiniteStateGuard::validate_state(&tool).is_ok());
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(FiniteStateGuard::diagnostic_report(&tool).is_none());

    tool.param_k = 1e300;
    tool.ic_y0 = 1e300;
    tool.recalculate();

    assert!(FiniteStateGuard::diverged(&tool));
    assert!(!FiniteStateGuard::is_finite_state(&tool));
    assert!(FiniteStateGuard::validate_state(&tool).is_err());

    let report: &DiagnosticReport =
        FiniteStateGuard::diagnostic_report(&tool).expect("Diagnostic report should be present");
    assert_eq!(report.step_index, 1);
    assert_eq!(report.t_diverged, 0.05);
    assert_eq!(report.variable, "y(t)");
    assert!(report.message.contains("diverged"));

    // Reset via FiniteStateGuard clears diagnostic report and restores finite state
    FiniteStateGuard::reset_parameters(&mut tool);
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!(FiniteStateGuard::diagnostic_report(&tool).is_none());
}

#[test]
fn test_ode_autoscale_step_size_recovery() {
    let mut tool = OdeSolverTool::default();
    assert_eq!(tool.dt, 0.05);

    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert_eq!(tool.dt, 0.025);

    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert_eq!(tool.dt, 0.0125);

    // Test recovering finite trajectories via step size auto-scaling
    // In Harmonic Oscillator with k = 50000.0 and dt = 0.05, RK4 diverges.
    // Halving dt to 0.025 and then 0.0125 brings the solver into the stability region.
    tool.preset = OdePreset::HarmonicOscillator;
    tool.param_k = 50000.0;
    tool.total_time = 10.0;
    tool.dt = 0.05;
    tool.ic_y0 = 1.0;
    tool.ic_v0 = 0.0;
    tool.recalculate();
    assert!(FiniteStateGuard::diverged(&tool));

    // Halve dt until integration succeeds and finite trajectory is restored
    for _ in 0..5 {
        if !FiniteStateGuard::diverged(&tool) {
            break;
        }
        FiniteStateGuard::auto_scale_step_size(&mut tool);
    }
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!(!tool.y_series.is_empty());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));

    let datasets = tool.export_datasets();
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/ode_recovered_trajectory.png", png_data);
    }
}

#[test]
fn test_ode_autoscale_minimum_floor() {
    let mut tool = OdeSolverTool {
        dt: 1e-6,
        ..Default::default()
    };
    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert_eq!(tool.dt, 1e-6);

    tool.dt = 1e-8;
    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert_eq!(tool.dt, 1e-6);

    tool.dt = f64::NAN;
    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert_eq!(tool.dt, 0.05);
}

#[test]
fn test_ode_clamp_instability_parameters_exponential_and_non_finite_edges() {
    let mut tool = OdeSolverTool {
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();
    assert!(FiniteStateGuard::diverged(&tool));

    FiniteStateGuard::clamp_parameters(&mut tool);
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(tool.param_k <= 5.0 && tool.param_k >= -5.0);
    assert!(tool.ic_y0 <= 10.0 && tool.ic_y0 >= -10.0);
    assert!(!tool.y_series.is_empty());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));

    // Verify NaN and Inf parameters/initial conditions clamp to preset defaults
    tool.param_k = f64::NAN;
    tool.ic_y0 = f64::INFINITY;
    tool.ic_v0 = f64::NEG_INFINITY;
    tool.recalculate();
    assert!(FiniteStateGuard::diverged(&tool));

    FiniteStateGuard::clamp_parameters(&mut tool);
    assert!(!FiniteStateGuard::diverged(&tool));
    assert_eq!(tool.param_k, 1.0);
    assert_eq!(tool.ic_y0, 1.0);
    assert_eq!(tool.ic_v0, 0.0);
}

#[test]
fn test_ode_clamp_instability_parameters_harmonic_logistic_and_custom() {
    // Harmonic Oscillator
    let mut tool_harmonic = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1e300,
        ic_y0: 1e300,
        ic_v0: 1e300,
        ..Default::default()
    };
    tool_harmonic.recalculate();
    assert!(FiniteStateGuard::diverged(&tool_harmonic));

    FiniteStateGuard::clamp_parameters(&mut tool_harmonic);
    assert!(!FiniteStateGuard::diverged(&tool_harmonic));
    assert!(tool_harmonic.param_k <= 10.0);
    assert!(tool_harmonic.ic_y0 <= 10.0);
    assert!(tool_harmonic.ic_v0 <= 10.0);

    // Logistic Growth (including NaN / Inf edge cases)
    let mut tool_logistic = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: f64::INFINITY,
        param_cap_k: f64::NAN,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool_logistic.recalculate();
    assert!(FiniteStateGuard::diverged(&tool_logistic));

    FiniteStateGuard::clamp_parameters(&mut tool_logistic);
    assert!(!FiniteStateGuard::diverged(&tool_logistic));
    assert_eq!(tool_logistic.param_r, 1.0);
    assert_eq!(tool_logistic.param_cap_k, 10.0);
    assert_eq!(tool_logistic.ic_y0, 10.0);

    // Custom preset with NaN parameter
    let mut tool_custom = OdeSolverTool {
        preset: OdePreset::Custom,
        param_k: f64::NAN,
        ic_y0: f64::NAN,
        ..Default::default()
    };
    tool_custom.recalculate();
    assert!(FiniteStateGuard::diverged(&tool_custom));

    FiniteStateGuard::clamp_parameters(&mut tool_custom);
    assert!(!FiniteStateGuard::diverged(&tool_custom));
    assert_eq!(tool_custom.param_k, 1.0);
    assert_eq!(tool_custom.ic_y0, 1.0);
}

#[test]
fn test_ode_divergence_and_recovery_telemetry() {
    let _ = federated_registry::global_registry().try_recv_all();

    let mut tool = OdeSolverTool {
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();

    let events = federated_registry::global_registry().try_recv_all();
    let div_event = events
        .iter()
        .find(|e| e.source == "OdeSolverTool" && e.severity == federated_registry::Severity::Error)
        .expect("Divergence telemetry event should be emitted");

    assert_eq!(
        div_event.metadata.get("diverged_variable"),
        Some(&"y(t)".to_string())
    );
    assert_eq!(div_event.metadata.get("t_diverged"), Some(&"0.0500".to_string()));
    assert_eq!(div_event.metadata.get("step_count"), Some(&"1".to_string()));

    // Centralized recovery action emits telemetry
    emit_recovery_telemetry(
        "OdeSolverTool",
        "Auto-scale Step Size (dt)",
        "Halved step size",
    );
    let recovery_events = federated_registry::global_registry().try_recv_all();
    assert!(recovery_events.iter().any(|e| {
        e.source == "OdeSolverTool"
            && e.severity == federated_registry::Severity::Info
            && e.message.contains("Auto-scale Step Size")
            && e.metadata.get("action") == Some(&"Auto-scale Step Size (dt)".to_string())
    }));
}

#[test]
fn test_ode_recovery_banner_and_action_dispatch() {
    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();
    assert!(FiniteStateGuard::diverged(&tool));

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut changed = false;
            let mut py = Vec::new();
            let mut pv = Vec::new();
            recovery::render_recovery_ui(ui, &mut tool, &mut changed, &mut py, &mut pv);
            let _ = render_recovery_banner(ui, FiniteStateGuard::diagnostic_report(&tool));
            let _ = render_recovery_action_buttons(ui);
        });
    });

    // Verify apply_recovery_action dispatches all three centralized RecoveryAction variants
    let mut changed = false;
    let mut py = Vec::new();
    let mut pv = Vec::new();
    recovery::apply_recovery_action(
        &mut tool,
        RecoveryAction::AutoScaleStepSize,
        &mut changed,
        &mut py,
        &mut pv,
    );
    assert!(changed);
    assert_eq!(tool.dt, 0.025);

    changed = false;
    recovery::apply_recovery_action(
        &mut tool,
        RecoveryAction::ClampParameters,
        &mut changed,
        &mut py,
        &mut pv,
    );
    assert!(changed);
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(!py.is_empty());

    changed = false;
    recovery::apply_recovery_action(
        &mut tool,
        RecoveryAction::ResetParameters,
        &mut changed,
        &mut py,
        &mut pv,
    );
    assert!(changed);
    assert_eq!(tool.dt, 0.05);
    assert_eq!(tool.param_k, 1.0);
}
