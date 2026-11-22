#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::tabs::chaos::attractors::{
    AttractorPlotter, AttractorRunner, SolverExecutionState, MAX_SIMULATION_SPEED,
};
use crate::tabs::number_theory::factorization::{FactorizationTool, MAX_SYNC_FACTORIZATION_INPUT};
use std::collections::VecDeque;
use std::sync::Arc;

fn make_preset(domain: &str, title: &str, config: serde_json::Value) -> String {
    serde_json::json!({
        "metadata": {
            "schema_version": "1.0",
            "title": title,
            "domain": domain,
            "timestamp": "2026-11-19T00:00:00Z"
        },
        "config": config
    })
    .to_string()
}

fn make_framework_session(tool_name: &str, config: serde_json::Value) -> String {
    serde_json::json!({
        "selected_tool_index": null,
        "selected_tool_name": tool_name,
        "show_theory_portal": false,
        "show_starter_gallery": false,
        "tool_states": {
            tool_name: config
        }
    })
    .to_string()
}

#[test]
fn test_tool_state_bounds_scanner_bytes_depth_arrays_and_strings() {
    let bounds = ToolStateBounds::new(128, 4, 3);
    assert!(bounds
        .validate_json("\u{feff}{\"pts\":[[1,2],[3,4]],\"note\":\"[[[[1,2,3,4,5]]]]\\\"comma,ok\"}")
        .is_ok());

    let oversized = format!("{{\"pad\":\"{}\"}}", "x".repeat(200));
    assert!(bounds.validate_json(&oversized).is_err());

    let too_deep = "[[[[[1]]]]]";
    assert!(bounds.validate_json(too_deep).is_err());

    let too_many_items = "{\"items\":[1,2,3,4]}";
    assert!(bounds.validate_json(too_many_items).is_err());

    let zero_array_bounds = ToolStateBounds::new(128, 4, 0);
    assert!(zero_array_bounds.validate_json("[]").is_ok());
    assert!(zero_array_bounds.validate_json("[1]").is_err());
}

#[test]
fn test_import_preset_and_framework_session_reject_bounds_before_deserialization() {
    let mut fw = SimulationFramework::new("analysis");
    assert!(fw.select_tool_by_name("OdeSolverTool"));
    let baseline = fw.active_tool.as_ref().unwrap().save_state().unwrap();

    // 1. Deeply nested JSON rejected on both import_preset_json and load_state
    let deep_nested = format!(
        "{{\"metadata\":{{\"schema_version\":\"1.0\",\"title\":\"Deep\",\"domain\":\"analysis\",\"timestamp\":\"2026-11-19T00:00:00Z\"}},\"config\":{}{}{}}}",
        "[".repeat(MAX_JSON_NESTING_DEPTH + 4),
        "0",
        "]".repeat(MAX_JSON_NESTING_DEPTH + 4)
    );
    assert!(fw
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&deep_nested)
        .is_err());

    // 2. Oversized array rejected on both import_preset_json and SimulationFramework::load_state
    let huge_array_items: String = (0..=MAX_JSON_ARRAY_LEN)
        .map(|_| "0")
        .collect::<Vec<_>>()
        .join(",");
    let huge_array_preset = format!(
        "{{\"metadata\":{{\"schema_version\":\"1.0\",\"title\":\"Array\",\"domain\":\"analysis\",\"timestamp\":\"2026-11-19T00:00:00Z\"}},\"config\":{{\"arr\":[{huge_array_items}]}}}}"
    );
    assert!(fw
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&huge_array_preset)
        .is_err());

    let huge_array_session = format!(
        "{{\"selected_tool_index\":0,\"selected_tool_name\":\"OdeSolverTool\",\"show_theory_portal\":true,\"tool_states\":{{\"ODE Solvers\":{{\"arr\":[{huge_array_items}]}}}}}}"
    );
    fw.load_state(&huge_array_session);
    assert!(!fw.show_theory_portal);
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state().as_deref(),
        Some(baseline.as_str())
    );
}

#[test]
fn test_framework_session_per_entry_and_session_byte_limits() {
    let mut fw = SimulationFramework::new("climate");
    assert!(fw.select_tool_by_name("CeraTool"));

    // Per-entry limit: valid CERA state restores while oversized >2 MiB sibling entry is dropped
    let valid_cera = serde_json::json!({
        "in_channels": 5,
        "latent_channels": 10,
        "aligned_channels": 5,
        "num_levels": 20,
        "output_size": 4
    });
    let oversized_blob = "a".repeat(MAX_TOOL_STATE_BYTES + 128);
    let mixed_session = format!(
        "{{\"selected_tool_index\":null,\"selected_tool_name\":\"CERA Model\",\"show_theory_portal\":false,\"tool_states\":{{\"CERA Model\":{},\"OversizedTool\":{{\"pad\":\"{}\"}}}}}}",
        valid_cera, oversized_blob
    );
    fw.load_state(&mixed_session);
    assert!(!fw.saved_tool_states.contains_key("OversizedTool"));
    let restored_val: serde_json::Value =
        serde_json::from_str(&fw.active_tool.as_ref().unwrap().save_state().unwrap()).unwrap();
    assert_eq!(restored_val, valid_cera);

    // Escaped legacy string hiding deep nesting in tool_states is also dropped
    let deep_escaped = serde_json::to_string(&format!(
        "{}0{}",
        "[".repeat(MAX_JSON_NESTING_DEPTH + 5),
        "]".repeat(MAX_JSON_NESTING_DEPTH + 5)
    ))
    .unwrap();
    let escaped_session = format!(
        "{{\"selected_tool_index\":null,\"selected_tool_name\":\"CERA Model\",\"show_theory_portal\":false,\"tool_states\":{{\"LegacyDeep\":{deep_escaped}}}}}"
    );
    fw.load_state(&escaped_session);
    assert!(!fw.saved_tool_states.contains_key("LegacyDeep"));

    // Session-wide >8 MiB payload rejected before parsing
    let huge_session = format!(
        "{{\"selected_tool_index\":0,\"show_theory_portal\":true,\"tool_states\":{{\"Pad\":\"{}\"}}}}",
        "b".repeat(MAX_SESSION_STATE_BYTES + 64)
    );
    fw.load_state(&huge_session);
    assert!(!fw.show_theory_portal);
}

fn assert_vr_and_cera_proof_cases() {
    let mut vr_fw = SimulationFramework::new("geometry_topology");
    assert!(vr_fw.select_tool_by_name("VietorisRipsTool"));
    let vr_baseline = vr_fw.active_tool.as_ref().unwrap().save_state().unwrap();
    let pts: Vec<[f64; 2]> = (0..1500).map(|i| [(i as f64) * 1e-4, 0.0]).collect();
    let vr_preset = make_preset(
        "geometry_topology",
        "Unbounded VR",
        serde_json::json!({"radius": 10.0, "points": pts}),
    );
    assert!(vr_fw
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&vr_preset)
        .is_err());

    let vr_session = make_framework_session(
        "Vietoris-Rips Complex",
        serde_json::json!({"radius": 10.0, "points": pts}),
    );
    vr_fw.load_state(&vr_session);
    assert_eq!(
        vr_fw.active_tool.as_ref().unwrap().save_state().as_deref(),
        Some(vr_baseline.as_str())
    );

    let mut cera_fw = SimulationFramework::new("climate");
    assert!(cera_fw.select_tool_by_name("CeraTool"));
    let cera_preset = make_preset(
        "climate",
        "Overflow CERA",
        serde_json::json!({
            "in_channels": 2_usize,
            "latent_channels": usize::MAX,
            "aligned_channels": 2_usize,
            "num_levels": (usize::MAX / 2) + 2,
            "output_size": 2_usize
        }),
    );
    assert!(cera_fw
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&cera_preset)
        .is_err());
    let misaligned = make_preset(
        "climate",
        "Aligned > Latent",
        serde_json::json!({
            "in_channels": 2, "latent_channels": 4, "aligned_channels": 5, "num_levels": 10, "output_size": 2
        }),
    );
    assert!(cera_fw
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&misaligned)
        .is_err());
}

fn assert_attractor_and_factorization_proof_cases() {
    let mut chaos_fw = SimulationFramework::new("chaos");
    assert!(chaos_fw.select_tool_by_name("AttractorPlotter"));
    let attr_session = make_framework_session(
        "Attractor Plotter",
        serde_json::json!({
            "sigma": 10.0, "rho": 28.0, "beta": 2.666, "dt": 0.01,
            "simulation_speed": 2_000_000_000_usize, "execution_state": "Running", "show_annotations": false
        }),
    );
    chaos_fw.load_state(&attr_session);
    chaos_fw.active_tool.as_mut().unwrap().step_simulation();
    let post_attr: serde_json::Value =
        serde_json::from_str(&chaos_fw.active_tool.as_ref().unwrap().save_state().unwrap())
            .unwrap();
    assert!(post_attr["simulation_speed"].as_u64().unwrap() <= MAX_SIMULATION_SPEED as u64);

    let mut nt_fw = SimulationFramework::new("number_theory");
    assert!(nt_fw.select_tool_by_name("FactorizationTool"));
    let fact_preset = make_preset(
        "number_theory",
        "Max Prime Factorization",
        serde_json::json!({"input_text": "18446744073709551557"}),
    );
    assert!(nt_fw
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&fact_preset)
        .is_err());
}

#[test]
fn test_acceptance_proof_cases_across_all_four_tools() {
    assert_vr_and_cera_proof_cases();
    assert_attractor_and_factorization_proof_cases();
}

#[test]
fn test_attractor_runner_set_speed_and_factorization_boundary_caps() {
    let plotter = AttractorPlotter::default();
    let mut runner = AttractorRunner {
        system: math_explorer::physics::chaos::lorenz::LorenzBuilder::new().build(
            math_explorer::physics::chaos::lorenz::LorenzState::new(1.0, 1.0, 1.0),
        ),
        params: Arc::clone(&plotter.params),
        history: VecDeque::with_capacity(100),
        step_count: 0,
        execution_state: SolverExecutionState::Ready,
        rng: oxidize_core::rng::OxidizeRng::default(),
    };
    runner.process_command(SimCommand::SetSpeed(0));
    assert_eq!(runner.params.read().unwrap().simulation_speed, 1);
    runner.process_command(SimCommand::SetSpeed(999_999));
    assert_eq!(
        runner.params.read().unwrap().simulation_speed,
        MAX_SIMULATION_SPEED
    );

    let mut fact = FactorizationTool::default();
    let over_cap = format!(
        "{{\"input_text\":\"{}\"}}",
        MAX_SYNC_FACTORIZATION_INPUT + 1
    );
    assert!(fact.try_load_state(&over_cap).is_err());
    assert!(fact
        .try_load_state(&format!(
            "{{\"input_text\":\"{MAX_SYNC_FACTORIZATION_INPUT}\"}}"
        ))
        .is_ok());
}
