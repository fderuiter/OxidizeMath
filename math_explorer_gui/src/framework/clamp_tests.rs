#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::recovery::*;
use super::SimulationFramework;

fn active_json(fw: &SimulationFramework) -> serde_json::Value {
    serde_json::from_str(&fw.active_tool.as_ref().unwrap().save_state().unwrap()).unwrap()
}

fn load_active(fw: &mut SimulationFramework, state: &str) {
    fw.active_tool.as_mut().unwrap().load_state(state);
}

fn assert_clamp_emits(
    mut run: impl FnMut(&mut SimulationFramework),
    fw: &mut SimulationFramework,
    needle: &str,
) {
    for _ in 0..5 {
        let _ = federated_registry::global_registry().try_recv_all();
        run(fw);
        if drained_telemetry_contains(needle) {
            return;
        }
    }
    panic!("Expected SimulationFramework telemetry containing '{needle}'");
}

fn drained_telemetry_contains(needle: &str) -> bool {
    federated_registry::global_registry()
        .try_recv_all()
        .iter()
        .any(|e| {
            e.source == "SimulationFramework"
                && e.metadata
                    .get("details")
                    .is_some_and(|d| d.contains(needle))
        })
}

#[test]
fn test_clamp_parameters_multistep_lorenz_restores_plots_and_stabilizes() {
    let mut fw = SimulationFramework::new("chaos");
    assert!(fw.select_tool_by_name("AttractorPlotter"));
    let unstable = r#"{"sigma":250.0,"rho":800.0,"beta":25.0,"dt":0.25,"simulation_speed":5,"execution_state":{"Diverged":{"invalid_vec":["NaN","Infinity","-Infinity"],"diverged_at_step":10}},"show_annotations":true}"#;
    load_active(&mut fw, unstable);
    let saved_plot = vec![("Trajectory".to_string(), vec![[0.0, 1.0], [1.0, 2.5]])];
    fw.current_valid_plot_datasets = saved_plot.clone();
    fw.cache_valid_snapshot();

    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&[("Trajectory".to_string(), vec![[0.0, f64::NAN]])]);
    assert!(fw.inspect_runtime_state().is_err());

    {
        let _g = crate::reflective_ui::tests::TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert_clamp_emits(
            |f| {
                load_active(f, unstable);
                f.current_valid_plot_datasets = saved_plot.clone();
                f.cache_valid_snapshot();
                f.handle_state_guard_action(StateGuardAction::ClampParameters);
            },
            &mut fw,
            "Clamped snapshot JSON",
        );
        assert_eq!(egui_plot::get_last_plot_datasets(), saved_plot);
    }
    assert!(!fw.is_diverged() && fw.diagnostic_report().is_none());

    let post = active_json(&fw);
    assert!(post["sigma"].as_f64().unwrap() <= 50.0 && post["rho"].as_f64().unwrap() <= 100.0);
    assert!(post["beta"].as_f64().unwrap() <= 10.0 && post["dt"].as_f64().unwrap() <= 0.01);
    assert_eq!(post["execution_state"], serde_json::json!("Ready"));

    for _ in 0..40 {
        fw.active_tool.as_mut().unwrap().step_simulation();
    }
    let post_ds = fw.active_tool.as_ref().unwrap().export_datasets();
    assert!(inspect_datasets_for_divergence(&post_ds).is_ok());
    if let Ok(png) = egui_plot::export::export_custom_png(&post_ds, &Default::default()) {
        let _ = std::fs::write("/tmp/clamp_parameters_lorenz_recovery.png", png);
    }
}

#[test]
#[cfg(feature = "climate")]
fn test_clamp_parameters_dispatches_domain_hooks_and_restores_datasets() {
    let mut cera_fw = SimulationFramework::new("climate");
    assert!(cera_fw.select_tool_by_name("CeraTool"));
    load_active(
        &mut cera_fw,
        r#"{"in_channels":99,"latent_channels":4,"aligned_channels":18,"num_levels":90,"output_size":40}"#,
    );
    let plots = vec![("CERA".to_string(), vec![[0.0, 0.4], [1.0, 0.8]])];
    cera_fw.current_valid_plot_datasets = plots.clone();
    cera_fw.cache_valid_snapshot();
    {
        let _g = crate::reflective_ui::tests::TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert_clamp_emits(
            |f| {
                load_active(
                    f,
                    r#"{"in_channels":99,"latent_channels":4,"aligned_channels":18,"num_levels":90,"output_size":40}"#,
                );
                f.current_valid_plot_datasets = plots.clone();
                f.cache_valid_snapshot();
                f.clamp_parameters();
            },
            &mut cera_fw,
            "domain hook",
        );
        assert_eq!(egui_plot::get_last_plot_datasets(), plots);
    }
    let cera_val = active_json(&cera_fw);
    assert_eq!(cera_val["in_channels"].as_u64(), Some(10));
    assert_eq!(cera_val["aligned_channels"].as_u64(), Some(4));

    let mut ode_fw = SimulationFramework::new("analysis");
    assert!(ode_fw.select_tool_by_name("OdeSolverTool"));
    let mut ode_json: serde_json::Value =
        serde_json::from_str(ode_fw.baseline_tool_state.as_ref().unwrap()).unwrap();
    ode_json["param_k"] = serde_json::json!(1e200);
    ode_json["dt"] = serde_json::json!(5.0);
    load_active(&mut ode_fw, &ode_json.to_string());
    ode_fw.cache_valid_snapshot();
    ode_fw.clamp_parameters();
    let ode_post = active_json(&ode_fw);
    assert!(
        ode_post["param_k"].as_f64().unwrap() <= 5.0 && ode_post["dt"].as_f64().unwrap() <= 0.1
    );

    let mut lbm_fw = SimulationFramework::new("fluid_dynamics");
    assert!(!lbm_fw
        .active_tool
        .as_ref()
        .unwrap()
        .theory()
        .theory_parameters()
        .is_empty());
    load_active(&mut lbm_fw, r#"{"viscosity":99.0}"#);
    lbm_fw.cache_valid_snapshot();
    lbm_fw.clamp_parameters();
    assert!(active_json(&lbm_fw)["viscosity"].as_f64().unwrap() < 99.0);
}

#[test]
fn test_clamp_parameters_baseline_fallback_and_telemetry() {
    let mut fw = SimulationFramework::new("chaos");
    assert!(fw.select_tool_by_name("AttractorPlotter"));
    let baseline = fw.baseline_tool_state.clone().unwrap();
    let drifted = baseline.replace("\"sigma\":10.0", "\"sigma\":42.0");
    let _g = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    assert_clamp_emits(
        |f| {
            load_active(f, &drifted);
            f.clamp_parameters();
        },
        &mut fw,
        "Fallback",
    );
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state(),
        Some(baseline.clone())
    );

    assert_clamp_emits(
        |f| {
            load_active(f, &drifted);
            f.last_valid_snapshot = None;
            f.clamp_parameters();
        },
        &mut fw,
        "missing",
    );
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state(),
        Some(baseline)
    );
}
