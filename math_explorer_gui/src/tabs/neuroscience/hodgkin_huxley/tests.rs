#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::presets::{serialize_compact_preset_envelope, PresetDialogState};
use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
use federated_registry::Severity;
use std::sync::mpsc::channel;

#[test]
fn test_unified_sim_tool_enforces_parameter_constraints_on_flat_and_extended_json() {
    let mut lbm = UnifiedSimTool::<LbmUnified>::new();
    assert!(lbm.active_challenge_id().is_none());
    let lbm_baseline = lbm.save_state().expect("lbm save_state");

    for bad_lbm in [
        r#"{"viscosity": -0.1}"#,
        r#"{"viscosity": 0.001}"#,
        r#"{"viscosity": 0.9}"#,
        r#"{"params": {"viscosity": 0.001}, "atlas_visible": false}"#,
    ] {
        assert!(
            lbm.try_load_state(bad_lbm).is_err(),
            "expected out-of-range LBM payload {bad_lbm} to fail"
        );
        assert_eq!(lbm.save_state().as_deref(), Some(lbm_baseline.as_str()));
    }
    assert!(lbm.try_load_state(r#"{"viscosity": 0.02}"#).is_ok());

    let mut hh = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    let hh_baseline = hh.save_state().expect("hh save_state");
    for bad_hh in [
        r#"{"g_na": -5.0, "g_k": 36.0, "g_l": 0.3, "i_ext": 10.0}"#,
        r#"{"g_na": 250.0, "g_k": 36.0, "g_l": 0.3, "i_ext": 10.0}"#,
        r#"{"g_na": 120.0, "g_k": 36.0, "g_l": 0.3, "i_ext": 500.0}"#,
        r#"{"params": {"g_na": 120.0, "g_k": -1.0, "g_l": 0.3, "i_ext": 10.0}, "atlas_visible": true}"#,
    ] {
        assert!(
            hh.try_load_state(bad_hh).is_err(),
            "expected out-of-range HH payload {bad_hh} to fail"
        );
        assert_eq!(hh.save_state().as_deref(), Some(hh_baseline.as_str()));
    }
    assert!(hh
        .try_load_state(r#"{"g_na": 120.0, "g_k": 36.0, "g_l": 0.3, "i_ext": 12.0}"#)
        .is_ok());
}

#[test]
fn test_unified_preset_and_url_import_fail_fast_with_telemetry() {
    let _ = crate::telemetry::poll_app_telemetry();
    let mut fluid_fw = SimulationFramework::new("fluid_dynamics");
    assert!(fluid_fw.select_tool_by_name("LatticeBoltzmannTool"));
    let baseline = fluid_fw.active_tool.as_ref().unwrap().save_state().unwrap();

    let bad_env = serialize_compact_preset_envelope(
        "Lattice Boltzmann Fluid",
        "fluid_dynamics",
        serde_json::json!({"viscosity": 0.95}),
    )
    .expect("envelope");

    let mut dialog = PresetDialogState::default();
    let (tx, rx) = channel();
    dialog.import_rx = Some(rx);
    tx.send(Ok(("bad_lbm.json".to_string(), bad_env.clone())))
        .unwrap();
    let loaded = dialog.poll_pending_import(|js| {
        fluid_fw
            .active_tool
            .as_mut()
            .unwrap()
            .import_preset_json(js)
    });
    assert!(!loaded);
    assert!(dialog.toast_error.is_some());
    let events = crate::telemetry::poll_app_telemetry();
    assert!(events
        .iter()
        .any(|e| e.source == "PresetSerialization" && e.severity == Severity::Warning));
    assert!(fluid_fw
        .import_url_preset_json(Some("LatticeBoltzmannTool"), Some(&bad_env))
        .is_err());
    assert_eq!(
        fluid_fw
            .active_tool
            .as_ref()
            .unwrap()
            .save_state()
            .as_deref(),
        Some(baseline.as_str())
    );
}

#[test]
fn test_hodgkin_huxley_finite_state_guard_detects_nan_and_inf() {
    let valid_params = HashMap::from([
        ("g_na".to_string(), 120.0),
        ("g_k".to_string(), 36.0),
        ("g_l".to_string(), 0.3),
        ("i_ext".to_string(), 10.0),
    ]);
    let mut hh = HodgkinHuxleyUnified::new(&valid_params);
    assert!(hh.is_finite_state());
    assert!(hh.diagnostic_report().is_none());

    hh.step(&valid_params);
    assert!(hh.is_finite_state());

    // Non-finite conductance must not be masked to 0.0 by sanitize_conductance
    let mut nan_cond = valid_params.clone();
    nan_cond.insert("g_na".to_string(), f64::NAN);
    hh.step(&nan_cond);
    assert!(!hh.is_finite_state());
    assert!(hh.diagnostic_report().is_some());
    assert!(hh.get_snapshot().custom_data.iter().all(|x| x.is_finite()));

    // Subsequent steps halt while diverged
    let halted_step = hh.step_count;
    hh.step(&valid_params);
    assert_eq!(hh.step_count, halted_step);

    // Reset via process_command and reset_model restores finite baseline
    hh.process_command(SimCommand::Reset, &valid_params);
    assert!(hh.is_finite_state());
    assert!(hh.diagnostic_report().is_none());

    // Zero dt triggers divergence diagnostic on "dt"
    hh.apply_scalar_step(120.0, 36.0, 0.3, 10.0, 0.0);
    assert!(!hh.is_finite_state());
    assert_eq!(hh.diagnostic_report().unwrap().variable, "dt");
    hh.reset_model(&valid_params);
    assert!(hh.is_finite_state());
}

#[test]
fn test_hodgkin_huxley_worker_faults_on_divergence_and_recovers() {
    let mut tool = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    assert_eq!(tool.state(), SimWorkerState::Ready);
    assert!(tool.has_finite_parameters());

    // Extreme finite parameters overflow Euler integration into non-finite voltage 'v'
    {
        let mut guard = tool.params.write().unwrap_or_else(|e| e.into_inner());
        guard.insert("i_ext".to_string(), f64::MAX);
        guard.insert("g_na".to_string(), f64::MAX);
    }
    assert!(tool.controller.send_command(SimCommand::Step(1)).is_ok());

    for _ in 0..400 {
        let _ = tool.controller.update();
        if tool.state() == SimWorkerState::Faulted {
            break;
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    assert_eq!(tool.state(), SimWorkerState::Faulted);
    assert!(!tool.has_finite_parameters());
    let report = tool
        .controller
        .diagnostic_report()
        .expect("worker should publish DiagnosticReport on HH divergence");
    assert_eq!(report.variable, "v");

    assert!(tool.clamp_parameters());
    assert_eq!(tool.state(), SimWorkerState::Ready);
    assert!(tool.has_finite_parameters());

    for _ in 0..300 {
        tool.step_simulation();
    }
    let datasets = tool.export_datasets();
    assert!(!datasets.is_empty());
    assert!(datasets[0].1.iter().any(|p| p[1] > 20.0));
    let opts = egui_plot::export::ExportOptions::default();
    if let Ok(png) = egui_plot::export::export_custom_png(&datasets, &opts) {
        let _ = std::fs::write("/tmp/hodgkin_huxley_recovered_action_potential.png", png);
    }
}
