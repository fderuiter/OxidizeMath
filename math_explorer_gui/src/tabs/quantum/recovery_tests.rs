#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::wave_sim::*;
use crate::async_sim::recovery;
use crate::async_sim::unified::UnifiedModel;
use crate::async_sim::SimCommand;
use eframe::egui;
use num_complex::Complex;
use std::collections::HashMap;

#[test]
fn test_quantum_divergence_detection_and_evolution_halt() {
    let params = HashMap::new();
    let mut sim = WaveUnified::new(&params);

    assert!(!sim.diverged());
    assert!(sim.diagnostic_report().is_none());

    // Step with valid state
    sim.step(&params);
    assert!(!sim.diverged());

    // Inject non-finite amplitude to trigger divergence guard
    sim.psi.vector[0] = Complex::new(f64::NAN, 0.0);
    sim.step(&params);

    assert!(sim.diverged());
    let report = sim.diagnostic_report().expect("Diagnostic report should be present");
    assert_eq!(report.step_index, 2);
    assert_eq!(report.variable, "psi(x)");
    assert!(report.message.contains("divergence") || report.message.contains("failure"));

    // Stepping while diverged must halt iterative stepping (no-op)
    sim.step(&params);
    assert_eq!(sim.diagnostic_report().unwrap().step_index, 2);
}

#[test]
fn test_quantum_divergence_telemetry_emission() {
    let _ = federated_registry::global_registry().try_recv_all();

    let mut sim = WaveUnified::new(&HashMap::new());
    sim.psi.vector[5] = Complex::new(f64::INFINITY, 0.0);
    sim.step(&HashMap::new());

    assert!(sim.diverged());

    let events = federated_registry::global_registry().try_recv_all();
    let div_event = events
        .iter()
        .find(|e| {
            e.source == "WaveSimulator"
                && e.severity == federated_registry::Severity::Error
                && e.metadata.get("diverged_variable").is_some()
        })
        .expect("Quantum divergence telemetry event should be emitted");

    assert_eq!(div_event.metadata.get("diverged_variable"), Some(&"psi(x)".to_string()));
    assert_eq!(div_event.metadata.get("preset"), Some(&"Wave Simulator".to_string()));
    assert!(div_event.metadata.contains_key("t_diverged"));
    assert!(div_event.metadata.contains_key("step_count"));
}

#[test]
fn test_quantum_parameter_clamp_and_recovery() {
    let _ = federated_registry::global_registry().try_recv_all();

    let mut sim = WaveUnified::new(&HashMap::new());
    sim.psi.vector[10] = Complex::new(f64::NAN, 0.0);
    sim.step(&HashMap::new());
    assert!(sim.diverged());

    // Recovery action: ClampParameters
    let mut params = HashMap::new();
    params.insert("potential_type".to_string(), 0.0);
    sim.process_command(SimCommand::Custom("ClampParameters".to_string()), &params);

    assert!(!sim.diverged());
    assert!(sim.diagnostic_report().is_none());

    // Recovery telemetry
    recovery::emit_recovery_telemetry(
        WaveUnified::tool_source(),
        "Clamp Instability Parameters",
        "Clamped parameters to stable bounds",
    );

    let events = federated_registry::global_registry().try_recv_all();
    assert!(events.iter().any(|e| {
        e.source == "WaveSimulator"
            && e.severity == federated_registry::Severity::Info
            && e.message.contains("Clamp Instability Parameters")
    }));
}

#[test]
fn test_quantum_parameter_reset_recovery() {
    let mut sim = WaveUnified::new(&HashMap::new());
    sim.psi.vector[10] = Complex::new(f64::NAN, 0.0);
    sim.step(&HashMap::new());
    assert!(sim.diverged());

    // Recovery action: Reset
    let default_params = WaveUnified::default_parameters();
    sim.process_command(SimCommand::Reset, &default_params);

    assert!(!sim.diverged());
    assert!(sim.diagnostic_report().is_none());
}

#[test]
fn test_quantum_buffer_snapshot_preservation_on_divergence() {
    let params = HashMap::new();
    let mut sim = WaveUnified::new(&params);

    // Initial valid snapshot
    let valid_snap = sim.get_snapshot();
    let any_data = valid_snap.structured_data.as_ref().unwrap();
    let wave_data = any_data.downcast_ref::<WaveData>().unwrap();
    assert!(!wave_data.prob_density.is_empty());
    assert!(wave_data.prob_density.iter().all(|p| p.is_finite()));

    // Divergence triggered
    sim.psi.vector[0] = Complex::new(f64::NAN, 0.0);
    sim.step(&params);
    assert!(sim.diverged());

    // Diverged snapshot must preserve valid finite prob_density instead of NaNs
    let div_snap = sim.get_snapshot();
    let div_data = div_snap.structured_data.as_ref().unwrap().downcast_ref::<WaveData>().unwrap();
    assert!(!div_data.prob_density.is_empty());
    assert!(div_data.prob_density.iter().all(|p| p.is_finite()));
}

#[test]
fn test_quantum_recovery_banner_ui_rendering() {
    let ctx = egui::Context::default();
    let mut sim = WaveUnified::new(&HashMap::new());
    sim.psi.vector[0] = Complex::new(f64::NAN, 0.0);
    sim.step(&HashMap::new());
    assert!(sim.diverged());

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let action = recovery::render_recovery_banner(ui, sim.diagnostic_report().as_ref());
            assert!(action.is_none());
        });
    });
}
