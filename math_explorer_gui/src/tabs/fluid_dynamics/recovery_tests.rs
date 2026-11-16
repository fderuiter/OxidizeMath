#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::lattice_boltzmann::*;
use crate::async_sim::recovery;
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::SimCommand;
use eframe::egui::{self, Color32};
use std::collections::HashMap;

#[test]
fn test_lbm_divergence_detection_and_stepping_halt() {
    let mut params = HashMap::new();
    params.insert("viscosity".to_string(), 0.02);
    let mut model = LbmUnified::new(&params);

    assert!(!model.diverged());
    assert!(model.diagnostic_report().is_none());

    // Step with normal parameters
    model.step(&params);
    assert!(!model.diverged());

    // Step with non-finite / unstable viscosity
    let mut bad_params = HashMap::new();
    bad_params.insert("viscosity".to_string(), -0.05); // tau <= 0.5 divergence
    model.step(&bad_params);

    assert!(model.diverged());
    let report = model.diagnostic_report().expect("Diagnostic report should be present");
    assert_eq!(report.step_index, 2);
    assert!(report.variable.contains("viscosity"));
    assert!(report.message.contains("divergence"));

    // Stepping while diverged must halt iterative stepping (no-op)
    model.step(&bad_params);
    assert_eq!(model.diagnostic_report().unwrap().step_index, 2);
}

#[test]
fn test_lbm_divergence_telemetry_emission() {
    let _ = federated_registry::global_registry().try_recv_all();

    let mut model = LbmUnified::new(&HashMap::new());
    let mut bad_params = HashMap::new();
    bad_params.insert("viscosity".to_string(), f64::NAN);
    model.step(&bad_params);

    assert!(model.diverged());

    let events = federated_registry::global_registry().try_recv_all();
    let div_event = events
        .iter()
        .find(|e| {
            e.source == "LatticeBoltzmannTool"
                && e.severity == federated_registry::Severity::Error
                && e.metadata.get("diverged_variable").is_some()
        })
        .expect("LBM divergence telemetry event should be emitted");

    assert!(div_event.message.contains("divergence"));
    assert_eq!(div_event.metadata.get("diverged_variable"), Some(&"viscosity".to_string()));
    assert_eq!(div_event.metadata.get("preset"), Some(&"Lattice Boltzmann (Demo)".to_string()));
    assert!(div_event.metadata.contains_key("t_diverged"));
    assert!(div_event.metadata.contains_key("step_count"));
}

#[test]
fn test_lbm_recovery_action_reset_and_telemetry() {
    let _ = federated_registry::global_registry().try_recv_all();

    let mut model = LbmUnified::new(&HashMap::new());
    let mut bad_params = HashMap::new();
    bad_params.insert("viscosity".to_string(), -0.01);
    model.step(&bad_params);
    assert!(model.diverged());

    // Recovery action: Reset
    let default_params = LbmUnified::default_parameters();
    model.process_command(SimCommand::Reset, &default_params);

    assert!(!model.diverged());
    assert!(model.diagnostic_report().is_none());

    recovery::emit_recovery_telemetry(
        LbmUnified::tool_source(),
        "Reset All Parameters",
        "Restored default baseline configuration",
    );

    let events = federated_registry::global_registry().try_recv_all();
    assert!(events.iter().any(|e| {
        e.source == "LatticeBoltzmannTool"
            && e.severity == federated_registry::Severity::Info
            && e.message.contains("Reset All Parameters")
    }));
}

#[test]
fn test_lbm_parameter_clamp_recovery() {
    let mut params = HashMap::new();
    params.insert("viscosity".to_string(), -0.5);

    LbmUnified::clamp_parameters(&mut params);
    assert_eq!(*params.get("viscosity").unwrap(), 0.02);

    let mut high_params = HashMap::new();
    high_params.insert("viscosity".to_string(), 10.0);
    LbmUnified::clamp_parameters(&mut high_params);
    assert_eq!(*high_params.get("viscosity").unwrap(), 0.5);
}

#[test]
fn test_lbm_buffer_snapshot_preservation_on_divergence() {
    let mut params = HashMap::new();
    params.insert("viscosity".to_string(), 0.02);
    let mut model = LbmUnified::new(&params);

    model.step(&params);
    let valid_snap = model.get_snapshot();
    let valid_guard = valid_snap.pixels.read().unwrap();
    assert!(!valid_guard.is_empty());
    assert!(valid_guard.iter().any(|&p| p != Color32::BLACK));
    drop(valid_guard);

    // Trigger divergence
    let mut bad_params = HashMap::new();
    bad_params.insert("viscosity".to_string(), f64::NAN);
    model.step(&bad_params);
    assert!(model.diverged());

    // Snapshot after divergence must NOT collapse into blank/black display
    let diverged_snap = model.get_snapshot();
    let div_guard = diverged_snap.pixels.read().unwrap();
    assert!(!div_guard.is_empty());
    assert!(div_guard.iter().any(|&p| p != Color32::BLACK));
}

#[test]
fn test_lbm_recovery_banner_ui_rendering() {
    let ctx = egui::Context::default();
    let mut model = LbmUnified::new(&HashMap::new());
    let mut bad_params = HashMap::new();
    bad_params.insert("viscosity".to_string(), -0.1);
    model.step(&bad_params);
    assert!(model.diverged());

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let action = recovery::render_recovery_banner(ui, model.diagnostic_report().as_ref());
            // No button clicked yet
            assert!(action.is_none());
        });
    });
}

#[test]
fn test_lbm_unified_sim_tool_divergence_guards() {
    let tool = UnifiedSimTool::<LbmUnified>::new();
    assert!(!tool.state().is_running());
    assert!(!tool.is_diverged());
}
