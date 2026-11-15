#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, TraceSeries};

#[test]
fn test_ode_divergence_detection() {
    let mut tool = OdeSolverTool::default();
    assert!(!tool.diverged());

    // Set exponential growth rate and initial value that causes f64 overflow/divergence
    tool.param_k = 1e300;
    tool.ic_y0 = 1e300;
    tool.recalculate();

    assert!(tool.diverged());
    // Verify time_series and y_series do not contain non-finite values
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    assert!(tool.time_series.iter().all(|t| t.is_finite()));

    // Reset parameters recovers from divergence
    tool.reset_parameters();
    assert!(!tool.diverged());
    assert!(!tool.y_series.is_empty());
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
fn test_ode_harmonic_oscillator_divergence() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    assert!(tool.v_series.iter().all(|v| v.is_finite()));
}

#[test]
fn test_ode_logistic_growth_divergence() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: 1e300,
        param_cap_k: 1.0,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
}

#[test]
fn test_ode_trace_snapshot_pinning_and_export() {
    let mut tool = OdeSolverTool {
        param_k: 1.0,
        ..Default::default()
    };
    tool.recalculate();

    // Pin baseline trajectory k=1.0
    let summary = tool.param_summary();
    let pts: Vec<[f64; 2]> = tool
        .time_series
        .iter()
        .zip(&tool.y_series)
        .map(|(&t, &y)| [t, y])
        .collect();
    tool.snapshot_manager.add_snapshot(
        "Baseline 1",
        summary,
        vec![TraceSeries::new("y(t)", pts.clone())],
    );

    assert_eq!(tool.snapshot_manager.len(), 1);

    // Change parameter k=2.5 and recalculate active simulation
    tool.param_k = 2.5;
    tool.recalculate();

    // Pinned snapshot retains baseline curve points
    assert_eq!(
        tool.snapshot_manager.snapshots()[0].param_summary,
        r"k=1.00, \Delta t=0.05"
    );
    assert_eq!(tool.snapshot_manager.snapshots()[0].series[0].points, pts);

    // Export datasets includes active + pinned traces
    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Baseline 1")));

    // State serialization roundtrip preserves snapshot
    let state = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&state);
    assert_eq!(restored.snapshot_manager.len(), 1);
}

#[test]
fn test_ode_auto_history_buffering() {
    let mut tool = OdeSolverTool::default();
    assert_eq!(tool.snapshot_manager.history_len(), 0);

    for i in 1..=7 {
        let old_summary = tool.param_summary();
        let pts: Vec<[f64; 2]> = tool.time_series.iter().zip(&tool.y_series).map(|(&t, &y)| [t, y]).collect();
        tool.snapshot_manager.push_history_run(old_summary, vec![TraceSeries::new("y(t)", pts)]);
        tool.param_k = i as f64 * 0.5;
        tool.recalculate();
    }

    assert_eq!(tool.snapshot_manager.history_len(), 5);
    assert!(tool.snapshot_manager.history_runs()[0].param_summary.contains("k=1.00"));
    assert!(tool.snapshot_manager.history_runs()[4].param_summary.contains("k=3.00"));

    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Run")));
}

#[test]
fn test_ode_auto_callout_annotations_and_persistence() {
    let mut tool = OdeSolverTool::default();
    assert_eq!(tool.callout_manager.len(), 2);
    let initial_callout = tool
        .callout_manager
        .callouts()
        .iter()
        .find(|c| c.tag.as_deref() == Some("initial_condition"))
        .expect("initial callout");
    assert_eq!(initial_callout.target_point, [0.0, 1.0]);
    assert!(initial_callout.text.contains("Initial"));

    let final_callout = tool
        .callout_manager
        .callouts()
        .iter()
        .find(|c| c.tag.as_deref() == Some("final_state"))
        .expect("final callout");
    assert_eq!(final_callout.target_point[0], 10.0);
    assert!(final_callout.text.contains("Final"));

    // Add a custom user callout
    tool.callout_manager
        .add_callout([5.0, 2.0], "Equilibrium Y=2.0", [15.0, -20.0]);
    assert_eq!(tool.callout_manager.len(), 3);

    // Save and load state
    let state = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&state);
    assert_eq!(restored.callout_manager.len(), 3);
    assert!(restored
        .callout_manager
        .callouts()
        .iter()
        .any(|c| c.text == "Equilibrium Y=2.0"));
}

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

    assert_eq!(div_event.metadata.get("diverged_variable"), Some(&"y(t)".to_string()));
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

#[test]
fn test_ode_annotations_toggle_and_rendering() {
    let mut tool = OdeSolverTool::default();
    assert!(tool.show_annotations);

    // Toggle off and on
    tool.show_annotations = false;
    assert!(!tool.show_annotations);
    tool.show_annotations = true;

    // Test serialization roundtrip with show_annotations
    tool.show_annotations = false;
    let saved = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&saved);
    assert!(!restored.show_annotations);

    // Test rendering with each preset
    let ctx = egui::Context::default();

    // 1. Exponential
    tool.show_annotations = true;
    tool.preset = OdePreset::Exponential;
    tool.param_k = -0.5;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // 2. Harmonic Oscillator
    tool.preset = OdePreset::HarmonicOscillator;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // 3. Logistic Growth
    tool.preset = OdePreset::LogisticGrowth;
    tool.param_cap_k = 12.0;
    tool.ic_y0 = 2.0;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // Disable annotations and ensure clean rendering
    tool.show_annotations = false;
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // Export plot screenshot for verification
    let datasets = tool.export_datasets();
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/ode_solver_trajectory.png", png_data);
    }
}
