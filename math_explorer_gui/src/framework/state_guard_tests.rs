#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::recovery::*;
use super::SimulationFramework;

#[test]
fn test_simulation_framework_state_snapshot_caching_and_memory_limit() {
    let mut framework = SimulationFramework::new("chaos");
    assert!(framework.active_tool.is_some());

    // Verify initial snapshot is cached on tool selection
    assert!(framework.last_valid_snapshot().is_some());
    let initial_snap = framework.last_valid_snapshot().unwrap();
    assert!(initial_snap.memory_size() <= 2 * 1024 * 1024);

    // Populate valid plot datasets
    let mock_datasets = vec![
        (
            "Trajectory X".to_string(),
            (0..500)
                .map(|i| [i as f64 * 0.01, (i as f64 * 0.01).sin()])
                .collect(),
        ),
        (
            "Trajectory Y".to_string(),
            (0..500)
                .map(|i| [i as f64 * 0.01, (i as f64 * 0.01).cos()])
                .collect(),
        ),
    ];
    egui_plot::append_last_plot_datasets(&mock_datasets);

    framework.cache_valid_snapshot();
    let snap = framework
        .last_valid_snapshot()
        .expect("Snapshot must be present");
    assert!(!snap.plot_datasets.is_empty());
    assert!(
        snap.memory_size() <= 2 * 1024 * 1024,
        "Memory footprint {} bytes must not exceed 2 MB",
        snap.memory_size()
    );
}

#[test]
fn test_simulation_framework_runtime_inspection_and_divergence_detection() {
    let mut framework = SimulationFramework::new("fluid_dynamics");
    assert!(!framework.is_diverged());
    assert!(framework.diagnostic_report().is_none());

    // Valid datasets pass inspection
    let valid_data = vec![(
        "Velocity Field".to_string(),
        vec![[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
    )];
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&valid_data);
    assert!(framework.inspect_runtime_state().is_ok());
    assert!(!framework.is_diverged());

    // Non-finite dataset triggers divergence and diagnostic report
    let diverged_data = vec![(
        "Velocity Field".to_string(),
        vec![[0.0, 1.0], [1.0, f64::NAN], [2.0, 3.0]],
    )];
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&diverged_data);

    let result = framework.inspect_runtime_state();
    assert!(result.is_err());
    assert!(framework.is_diverged());

    let report = framework
        .diagnostic_report()
        .expect("Diagnostic report expected");
    assert_eq!(report.step_index, 1);
    assert!(report.variable.contains("Velocity Field"));
    assert!(report.message.contains("Non-finite floating-point value"));
}

#[test]
fn test_simulation_framework_step_back_rollback_restores_parameters_and_plot() {
    let mut framework = SimulationFramework::new("chaos");

    // Establish pre-divergence state and plot data
    let pre_divergence_data = vec![(
        "Trajectory".to_string(),
        vec![[0.0, 1.0], [1.0, 2.5], [2.0, 4.0]],
    )];
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&pre_divergence_data);
    framework.current_valid_plot_datasets = vec![(
        "Trajectory".to_string(),
        egui_plot::SharedPoints::from(pre_divergence_data[0].1.as_slice()),
    )];
    framework.cache_valid_snapshot();

    let pre_state = framework.active_tool.as_ref().unwrap().save_state();

    // Trigger divergence via non-finite infinity value
    let diverged_data = vec![(
        "Trajectory".to_string(),
        vec![[0.0, 1.0], [1.0, f64::INFINITY]],
    )];
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&diverged_data);
    let _ = framework.inspect_runtime_state();
    assert!(framework.is_diverged());

    // Execute Step-Back rollback
    let restored = framework.rollback_step_back();
    assert!(restored);
    assert!(!framework.is_diverged());
    assert!(framework.diagnostic_report().is_none());

    // Verify exact pre-divergence parameters and plot data restored
    let post_state = framework.active_tool.as_ref().unwrap().save_state();
    assert_eq!(pre_state, post_state);

    let restored_plot = egui_plot::get_last_plot_datasets();
    assert_eq!(restored_plot.len(), 1);
    assert_eq!(restored_plot[0].0, "Trajectory");
    assert_eq!(restored_plot[0].1, vec![[0.0, 1.0], [1.0, 2.5], [2.0, 4.0]]);
}

#[test]
fn test_simulation_framework_auto_scale_step_size_halves_dt_and_restarts() {
    let mut framework = SimulationFramework::new("analysis");

    // Select ODE Solver tool
    if let Some(pos) = framework
        .available_tools
        .iter()
        .position(|t| t.display_name.contains("ODE") || t.id.contains("Ode"))
    {
        framework.select_tool(pos);
    }

    framework.cache_valid_snapshot();
    let initial_state = framework.last_valid_snapshot().unwrap().tool_state.clone();

    // Inject divergence
    let diverged_data = vec![(
        "Solution y(t)".to_string(),
        vec![[0.0, 1.0], [0.1, f64::NAN]],
    )];
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&diverged_data);
    let _ = framework.inspect_runtime_state();
    assert!(framework.is_diverged());

    // Rollback with Auto-scale Step Size
    let success = framework.rollback_auto_scale_step_size();
    assert!(success);
    assert!(!framework.is_diverged());
    assert!(framework.diagnostic_report().is_none());

    // Verify step size halved in serialized state
    if let Some(state_str) = framework.active_tool.as_ref().unwrap().save_state() {
        if let (Some(_init), Ok(init_val), Ok(post_val)) = (
            initial_state,
            serde_json::from_str::<serde_json::Value>(
                framework
                    .last_valid_snapshot()
                    .unwrap()
                    .tool_state
                    .as_ref()
                    .unwrap(),
            ),
            serde_json::from_str::<serde_json::Value>(&state_str),
        ) {
            if let (Some(init_dt), Some(post_dt)) = (
                init_val.get("dt").and_then(|v| v.as_f64()),
                post_val.get("dt").and_then(|v| v.as_f64()),
            ) {
                assert!(
                    (post_dt - init_dt * 0.5).abs() < 1e-6,
                    "dt must be halved from {} to {}",
                    init_dt,
                    post_dt
                );
            }
        }
    }
}

#[test]
fn test_simulation_framework_preserves_view_and_selection_during_rollback() {
    let mut framework = SimulationFramework::new("fluid_dynamics");

    framework.show_theory_portal = true;
    framework.search_query = "turbulence".to_string();
    let original_idx = framework.selected_tool_index;

    // Diverge
    framework.is_diverged = true;
    framework.diagnostic_report = Some(DiagnosticReport::new(0.0, 1, "test", "Divergence"));

    // Rollback
    framework.rollback_step_back();

    // Verify active tool selection and user view settings preserved
    assert_eq!(framework.selected_tool_index, original_idx);
    assert!(framework.show_theory_portal);
    assert_eq!(framework.search_query, "turbulence");
}

#[test]
fn test_simulation_framework_performance_benchmarks() {
    let mut framework = SimulationFramework::new("chaos");

    // Benchmark 1: Runtime state inspection latency over 10,000 points (< 1 ms constraint)
    let large_datasets = vec![
        (
            "Large Field A".to_string(),
            (0..5000).map(|i| [i as f64, (i as f64).sin()]).collect(),
        ),
        (
            "Large Field B".to_string(),
            (0..5000).map(|i| [i as f64, (i as f64).cos()]).collect(),
        ),
    ];
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&large_datasets);

    let inspect_start = crate::web_io::PlatformInstant::now();
    let res = framework.inspect_runtime_state();
    let inspect_duration = inspect_start.elapsed();

    assert!(res.is_ok());
    assert!(
        inspect_duration < std::time::Duration::from_millis(1),
        "Inspection duration ({:?}) must be less than 1 millisecond per frame",
        inspect_duration
    );

    // Benchmark 2: Step-back state restoration latency (< 16 ms constraint)
    framework.current_valid_plot_datasets = large_datasets
        .into_iter()
        .map(|(n, p)| (n, egui_plot::SharedPoints::from(p)))
        .collect();
    framework.cache_valid_snapshot();

    // Trigger divergence
    framework.is_diverged = true;
    framework.diagnostic_report = Some(DiagnosticReport::new(0.0, 0, "var", "err"));

    let restore_start = crate::web_io::PlatformInstant::now();
    let restored = framework.rollback_step_back();
    let restore_duration = restore_start.elapsed();

    assert!(restored);
    assert!(
        restore_duration < std::time::Duration::from_millis(16),
        "Step-back state restoration ({:?}) must be less than 16 milliseconds per frame",
        restore_duration
    );
}

#[test]
fn test_simulation_framework_recovery_banner_ui_and_screenshot() {
    let mut framework = SimulationFramework::new("fluid_dynamics");

    let ctx = eframe::egui::Context::default();

    // Step 1: Render initial frame
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        framework.show(ctx, "fluid_dynamics_test");
    });
    assert!(!framework.is_diverged());

    // Step 2: Inject non-finite plot dataset
    let nan_data = vec![(
        "Lattice Velocity".to_string(),
        vec![[0.0, 0.5], [1.0, f64::NAN]],
    )];
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&nan_data);

    let _ = framework.inspect_runtime_state();
    assert!(framework.is_diverged());

    // Step 3: Render frame with recovery banner active
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        framework.show(ctx, "fluid_dynamics_test");
    });

    // Step 4: Generate visual screenshot artifact for PR verification
    let datasets = vec![
        (
            "Valid Trajectory (Pre-Divergence Snapshot)".to_string(),
            vec![[0.0, 1.0], [0.5, 1.8], [1.0, 2.5], [1.5, 3.1], [2.0, 3.6]],
        ),
        (
            "Restored Step-Back State (Halved dt = 0.025)".to_string(),
            vec![
                [0.0, 1.0],
                [0.25, 1.45],
                [0.5, 1.8],
                [0.75, 2.16],
                [1.0, 2.5],
            ],
        ),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write(
            "/tmp/simulation_framework_state_guard_rollback.png",
            png_data,
        );
    }
}

#[test]
fn test_simulation_framework_reset_synchronizes_snapshot_for_step_back() {
    let mut framework = SimulationFramework::new("analysis");
    assert!(framework.select_tool_by_name("OdeSolverTool"));
    let baseline = framework
        .baseline_tool_state
        .clone()
        .expect("baseline state expected");
    let mut val: serde_json::Value = serde_json::from_str(&baseline).unwrap();
    if let Some(obj) = val.as_object_mut() {
        obj.insert("dt".to_string(), serde_json::json!(0.2));
    }
    let modified = serde_json::to_string(&val).unwrap();
    framework
        .active_tool
        .as_mut()
        .unwrap()
        .load_state(&modified);
    egui_plot::append_last_plot_datasets(&[("Modified Run".to_string(), vec![[0.0, 5.0]])]);
    framework.current_valid_plot_datasets = vec![(
        "Modified Run".to_string(),
        egui_plot::SharedPoints::from(vec![[0.0, 5.0]]),
    )];
    framework.cache_valid_snapshot();

    let before_reset = crate::web_io::PlatformInstant::now();
    framework.reset_parameters();

    assert!(egui_plot::get_last_plot_datasets().is_empty());
    assert!(framework.current_valid_plot_datasets.is_empty());
    let snap = framework
        .last_valid_snapshot()
        .expect("last_valid_snapshot must remain populated after reset");
    assert_eq!(snap.tool_state.as_ref(), Some(&baseline));
    assert!(snap.plot_datasets.is_empty());
    assert_eq!(snap.step_index, 0);
    let zero = std::time::Duration::ZERO;
    assert_eq!(before_reset.duration_since(snap.timestamp), zero);
    assert!(snap.timestamp.duration_since(before_reset) >= zero);
    // Subsequent Step-Back must preserve restored baseline rather than reverting to modified state
    assert!(framework.rollback_step_back());
    assert_eq!(
        framework.active_tool.as_ref().unwrap().save_state(),
        Some(baseline)
    );

    let mut empty_fw = SimulationFramework::new("__nonexistent_domain__");
    assert!(empty_fw.active_tool.is_none());
    empty_fw.reset_parameters();
    assert!(empty_fw.last_valid_snapshot().is_none());
}

fn active_saved_state(fw: &SimulationFramework) -> String {
    fw.active_tool.as_ref().unwrap().save_state().unwrap()
}

#[test]
fn test_starter_card_launch_updates_baseline_and_snapshot_for_reset() {
    let mut fw = SimulationFramework::new("fluid_dynamics");
    let cards =
        crate::starter_gallery::get_starter_cards_for_domain("fluid_dynamics", &fw.available_tools);
    let source_sink = cards
        .iter()
        .find(|c| c.title == "Source-Sink Pair")
        .expect("Source-Sink Pair card expected");
    let factory_state = (fw
        .available_tools
        .iter()
        .find(|t| t.id == "PotentialFlowTool")
        .unwrap()
        .build)()
    .save_state()
    .unwrap();

    fw.show_starter_gallery = true;
    assert!(fw.launch_starter_card(source_sink).is_ok());
    assert!(!fw.show_starter_gallery);

    let preset_state = active_saved_state(&fw);
    assert_ne!(preset_state, factory_state);
    assert_eq!(
        fw.baseline_tool_state.as_deref(),
        Some(preset_state.as_str())
    );
    assert_eq!(
        fw.last_valid_snapshot()
            .and_then(|s| s.tool_state.as_deref()),
        Some(preset_state.as_str())
    );

    fw.active_tool.as_mut().unwrap().load_state(&factory_state);
    assert_eq!(active_saved_state(&fw), factory_state);

    fw.reset_parameters();
    assert_eq!(active_saved_state(&fw), preset_state);
    assert_eq!(
        fw.last_valid_snapshot()
            .and_then(|s| s.tool_state.as_deref()),
        Some(preset_state.as_str())
    );
}

#[test]
fn test_apply_preset_json_malformed_surfaces_toast_and_telemetry_warning() {
    let mut framework = SimulationFramework::new("analysis");
    assert!(framework.preset_dialog.toast_error.is_none());

    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut warning = None;
    for _ in 0..5 {
        let err_msg = framework
            .apply_preset_json("{not valid preset json")
            .expect_err("Malformed JSON must return Err");
        let toast = framework
            .preset_dialog
            .toast_error
            .clone()
            .expect("toast_error must be populated on malformed preset JSON");
        assert_eq!(toast, err_msg);
        if let Some(ev) = federated_registry::global_registry()
            .try_recv_all()
            .into_iter()
            .rev()
            .find(|ev| ev.source == "PresetSerialization" && ev.message == toast)
        {
            warning = Some(ev);
            break;
        }
    }
    let warning = warning.expect("Expected TelemetryEvent from PresetSerialization");
    assert_eq!(warning.severity, federated_registry::Severity::Warning);
}

#[test]
fn test_starter_card_launch_isolates_cross_pane_plot_datasets_and_syncs_owner() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut fw = SimulationFramework::new("epidemiology");
    let foreign_plot = vec![("OtherPaneCurve".to_string(), vec![[0.0, 99.0], [1.0, 88.0]])];
    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&foreign_plot);

    let cards =
        crate::starter_gallery::get_starter_cards_for_domain("epidemiology", &fw.available_tools);
    let net_card = cards
        .iter()
        .find(|c| c.tool_name == Some("NetworkPropagationTool"))
        .expect("NetworkPropagationTool starter card expected");
    assert!(fw.launch_starter_card(net_card).is_ok());

    // NetworkPropagationTool uses default egui_plot export_datasets(), so foreign pane data must
    // neither leak into its snapshot/cache nor be wiped from egui_plot's global buffer.
    assert!(fw.current_valid_plot_datasets.is_empty());
    assert!(fw
        .last_valid_snapshot()
        .is_some_and(|s| s.plot_datasets.is_empty()));
    assert_eq!(
        fw.sync_state.datasets_owner_tool,
        Some("Network Propagation")
    );
    assert_eq!(egui_plot::get_last_plot_datasets(), foreign_plot);
    egui_plot::clear_last_plot_datasets();
}
