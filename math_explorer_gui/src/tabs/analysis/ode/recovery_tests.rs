#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::presets::OdePreset;
use eframe::egui;

#[test]
fn test_ode_structured_diagnostic_report() {
    let mut tool = OdeSolverTool {
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    let report = tool.diagnostic_report().expect("Diagnostic report should be present");
    assert_eq!(report.step_index, 1);
    assert_eq!(report.t_diverged, 0.05);
    assert_eq!(report.variable, "y(t)");
    assert!(report.message.contains("diverged"));

    // Reset clears diagnostic report
    tool.reset_parameters();
    assert!(!tool.diverged());
    assert!(tool.diagnostic_report().is_none());
}

#[test]
fn test_ode_autoscale_step_size_recovery() {
    let mut tool = OdeSolverTool::default();
    assert_eq!(tool.dt, 0.05);

    tool.auto_scale_step_size();
    assert_eq!(tool.dt, 0.025);

    tool.auto_scale_step_size();
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
    assert!(tool.diverged());

    // Halve dt until integration succeeds and finite trajectory is restored
    for _ in 0..5 {
        if !tool.diverged() {
            break;
        }
        tool.auto_scale_step_size();
    }
    assert!(!tool.diverged());
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
    tool.auto_scale_step_size();
    assert!(tool.dt >= 1e-6);

    tool.dt = 1e-8;
    tool.auto_scale_step_size();
    assert_eq!(tool.dt, 1e-6);
}

#[test]
fn test_ode_clamp_instability_parameters_exponential() {
    let mut tool = OdeSolverTool {
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();
    assert!(tool.diverged());

    tool.clamp_instability_parameters();
    assert!(!tool.diverged());
    assert!(tool.param_k <= 5.0 && tool.param_k >= -5.0);
    assert!(tool.ic_y0 <= 10.0 && tool.ic_y0 >= -10.0);
    assert!(!tool.y_series.is_empty());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
}

#[test]
fn test_ode_clamp_instability_parameters_harmonic_and_logistic() {
    // Harmonic Oscillator
    let mut tool_harmonic = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1e300,
        ic_y0: 1e300,
        ic_v0: 1e300,
        ..Default::default()
    };
    tool_harmonic.recalculate();
    assert!(tool_harmonic.diverged());

    tool_harmonic.clamp_instability_parameters();
    assert!(!tool_harmonic.diverged());
    assert!(tool_harmonic.param_k <= 10.0);
    assert!(tool_harmonic.ic_y0 <= 10.0);
    assert!(tool_harmonic.ic_v0 <= 10.0);

    // Logistic Growth
    let mut tool_logistic = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: 1e300,
        param_cap_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool_logistic.recalculate();
    assert!(tool_logistic.diverged());

    tool_logistic.clamp_instability_parameters();
    assert!(!tool_logistic.diverged());
    assert!(tool_logistic.param_r <= 5.0);
    assert!(tool_logistic.param_cap_k <= 100.0);
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
    assert!(div_event.metadata.contains_key("t_diverged"));
    assert!(div_event.metadata.contains_key("step_count"));

    // Recovery action emits telemetry
    recovery::emit_recovery_telemetry("Auto-scale Step Size (dt)", "Halved step size");
    let recovery_events = federated_registry::global_registry().try_recv_all();
    assert!(recovery_events.iter().any(|e| {
        e.source == "OdeSolverTool"
            && e.severity == federated_registry::Severity::Info
            && e.message.contains("Auto-scale Step Size")
    }));
}

#[test]
fn test_ode_recovery_banner_ui_rendering() {
    let ctx = egui::Context::default();
    let mut tool = OdeSolverTool {
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();
    assert!(tool.diverged());

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut changed = false;
            let mut py = Vec::new();
            let mut pv = Vec::new();
            recovery::render_recovery_ui(ui, &mut tool, &mut changed, &mut py, &mut pv);
        });
    });
}
