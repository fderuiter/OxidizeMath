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
    let err_report = FiniteStateGuard::validate_state(&tool)
        .expect_err("validate_state should fail when diverged");

    let report: &DiagnosticReport =
        FiniteStateGuard::diagnostic_report(&tool).expect("Diagnostic report should be present");
    assert_eq!(report, &err_report);
    assert_eq!(report.step_index, 1);
    assert_eq!(report.t_diverged, 0.05);
    assert_eq!(report.variable, "y(t)");
    assert!(report.message.contains("diverged"));

    // Reset via FiniteStateGuard clears diagnostic report and restores finite state
    FiniteStateGuard::reset_parameters(&mut tool);
    assert!(!FiniteStateGuard::diverged(&tool));
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!(FiniteStateGuard::validate_state(&tool).is_ok());
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
    assert!(!FiniteStateGuard::is_finite_state(&tool));
    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert_eq!(tool.dt, 0.05);
    assert!(FiniteStateGuard::is_finite_state(&tool));
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
    assert!(FiniteStateGuard::is_finite_state(&tool));
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
    assert!(FiniteStateGuard::is_finite_state(&tool_harmonic));
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
    assert!(FiniteStateGuard::is_finite_state(&tool_logistic));
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
    assert!(FiniteStateGuard::is_finite_state(&tool_custom));
    assert_eq!(tool_custom.param_k, 1.0);
    assert_eq!(tool_custom.ic_y0, 1.0);
}

#[test]
fn test_ode_clamp_nan_and_inf_parameters_and_initial_conditions() {
    // Exponential with NaN/Inf
    let mut exp_tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        param_k: f64::NAN,
        ic_y0: f64::INFINITY,
        dt: f64::NAN,
        total_time: f64::INFINITY,
        ..Default::default()
    };
    assert!(!FiniteStateGuard::is_finite_state(&exp_tool));
    FiniteStateGuard::clamp_parameters(&mut exp_tool);
    assert!(FiniteStateGuard::is_finite_state(&exp_tool));
    assert_eq!(exp_tool.param_k, 1.0);
    assert_eq!(exp_tool.ic_y0, 1.0);
    assert_eq!(exp_tool.dt, 0.05);
    assert_eq!(exp_tool.total_time, 10.0);

    // Harmonic Oscillator with -Inf/NaN
    let mut osc_tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: f64::NEG_INFINITY,
        ic_y0: f64::NAN,
        ic_v0: f64::INFINITY,
        ..Default::default()
    };
    assert!(!FiniteStateGuard::is_finite_state(&osc_tool));
    FiniteStateGuard::clamp_parameters(&mut osc_tool);
    assert!(FiniteStateGuard::is_finite_state(&osc_tool));
    assert_eq!(osc_tool.param_k, 1.0);
    assert_eq!(osc_tool.ic_y0, 1.0);
    assert_eq!(osc_tool.ic_v0, 0.0);

    // Logistic Growth with Inf/NaN
    let mut log_tool = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: f64::INFINITY,
        param_cap_k: f64::NAN,
        ic_y0: f64::NEG_INFINITY,
        ..Default::default()
    };
    assert!(!FiniteStateGuard::is_finite_state(&log_tool));
    FiniteStateGuard::clamp_parameters(&mut log_tool);
    assert!(FiniteStateGuard::is_finite_state(&log_tool));
    assert_eq!(log_tool.param_r, 1.0);
    assert_eq!(log_tool.param_cap_k, 10.0);
    assert_eq!(log_tool.ic_y0, 1.0);
}

#[test]
fn test_ode_divergence_and_recovery_telemetry() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut found_div_event = None;
    for _ in 0..5 {
        let _ = federated_registry::global_registry().try_recv_all();
        let mut tool = OdeSolverTool {
            param_k: 1e300,
            ic_y0: 1e300,
            ..Default::default()
        };
        tool.recalculate();

        let events = federated_registry::global_registry().try_recv_all();
        if let Some(ev) = events.into_iter().find(|e| {
            e.source == "OdeSolverTool"
                && e.severity == federated_registry::Severity::Error
                && e.message.contains("Simulation diverged")
        }) {
            found_div_event = Some(ev);
            break;
        }
    }

    let div_event = found_div_event.expect("Divergence telemetry event should be emitted");
    assert_eq!(
        div_event.metadata.get("diverged_variable"),
        Some(&"y(t)".to_string())
    );
    assert_eq!(
        div_event.metadata.get("t_diverged"),
        Some(&"0.0500".to_string())
    );
    assert_eq!(div_event.metadata.get("step_count"), Some(&"1".to_string()));

    // Centralized recovery telemetry helper emits expected event
    let mut found_recovery = false;
    for _ in 0..5 {
        emit_recovery_telemetry(
            "OdeSolverTool",
            "Auto-scale Step Size (dt)",
            "Halved step size",
        );
        let recovery_events = federated_registry::global_registry().try_recv_all();
        if recovery_events.iter().any(|e| {
            e.source == "OdeSolverTool"
                && e.severity == federated_registry::Severity::Info
                && e.message.contains("Auto-scale Step Size")
                && e.metadata.get("action") == Some(&"Auto-scale Step Size (dt)".to_string())
        }) {
            found_recovery = true;
            break;
        }
    }
    assert!(found_recovery, "Recovery telemetry event should be emitted");
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
            assert!(render_recovery_banner(ui, FiniteStateGuard::diagnostic_report(&tool)).is_none());
            assert!(render_recovery_action_buttons(ui).is_none());
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
    assert!(FiniteStateGuard::is_finite_state(&tool));
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
    assert!(FiniteStateGuard::is_finite_state(&tool));
}

#[test]
fn test_ode_multi_solver_divergence_warning() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1.0,
        ic_y0: 1.0,
        ic_v0: 0.0,
        dt: 0.05,
        total_time: 10.0,
        ..Default::default()
    };
    tool.set_comparison_mode(true);
    assert!(!tool.diverged());

    // Large time step causes lower-order Euler to diverge or produce non-finite values
    tool.dt = 2.5;
    tool.param_k = 1e150;
    tool.ic_y0 = 1e150;
    tool.recalculate();

    assert!(tool.diverged());
    // Ensure all stored points remain finite
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    for snapshot in tool.comparison_manager().snapshots() {
        for series in &snapshot.series {
            assert!(series
                .points
                .iter()
                .all(|pt| pt[0].is_finite() && pt[1].is_finite()));
        }
    }
}

#[test]
fn test_ode_autoscale_and_clamp_align_with_max_steps() {
    use crate::presets::{checked_ode_step_count, MAX_ODE_STEPS};

    let mut tool = OdeSolverTool {
        dt: 0.001,
        total_time: 100.0,
        ..Default::default()
    };
    for _ in 0..12 {
        FiniteStateGuard::auto_scale_step_size(&mut tool);
        assert!(checked_ode_step_count(tool.dt, tool.total_time).is_some());
        assert!(!tool.time_series.is_empty());
        assert!(tool.time_series.len() <= MAX_ODE_STEPS + 1);
    }

    let mut clamped = OdeSolverTool {
        dt: 1e-300,
        total_time: 100.0,
        ..Default::default()
    };
    FiniteStateGuard::clamp_parameters(&mut clamped);
    assert!(checked_ode_step_count(clamped.dt, clamped.total_time).is_some());
    assert!(!clamped.time_series.is_empty());
    assert!(clamped.time_series.len() <= MAX_ODE_STEPS + 1);
}

#[test]
fn test_ode_nan_initial_condition() {
    let mut tool = OdeSolverTool {
        ic_y0: f64::NAN,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.is_empty());
}

#[test]
fn test_ode_untrusted_preset_and_session_step_caps() {
    use crate::framework::InteractiveTool;
    use crate::presets::MAX_ODE_STEPS;
    use std::collections::HashMap;
    use systems::eval_ode_final_state;

    let mut ode = OdeSolverTool::default();
    let res = ode.import_preset_json(r#"{"metadata":{"schema_version":"1.0","title":"t","domain":"analysis","timestamp":"2026-11-18T00:00:00Z"},"config":{"preset":"Exponential","dt":1e-300,"total_time":10.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"custom_expr":""}}"#);
    assert!(res.is_err());
    assert!(ode.time_series.len() <= 1_000_001);
    assert!(ode.time_series.len() <= MAX_ODE_STEPS + 1);

    ode.load_state(r#"{"preset":"Exponential","dt":5e-17,"total_time":100.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0}"#);
    assert!(ode.time_series.len() <= MAX_ODE_STEPS + 1);

    let final_tiny = eval_ode_final_state(
        OdePreset::Exponential,
        IntegrationMethod::Euler,
        1e-12,
        10.0,
        &HashMap::new(),
        None,
    );
    assert!(final_tiny.is_nan());
    let traj = simulate_preset(
        OdePreset::Exponential,
        IntegrationMethod::Euler,
        1e-300,
        10.0,
        1.0,
        1.0,
        10.0,
        1.0,
        0.0,
        None,
    );
    assert!(traj.time_series.is_empty());
}

