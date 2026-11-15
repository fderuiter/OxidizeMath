#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::{AppState, ThemeMode, WorkspaceEnvelope};

#[test]
fn test_callout_creation() {
    let mut manager = CalloutManager::new();
    assert_eq!(manager.len(), 0);
    assert!(manager.is_empty());

    let id1 = manager.add_callout([1.0, 2.0], "Equilibrium Y=2.0", [10.0, -15.0]);
    assert_eq!(id1, 1);
    assert_eq!(manager.len(), 1);
    assert!(!manager.is_empty());

    let callout = &manager.callouts()[0];
    assert_eq!(callout.id, 1);
    assert_eq!(callout.target_point, [1.0, 2.0]);
    assert_eq!(callout.label_offset, [10.0, -15.0]);
    assert_eq!(callout.text, "Equilibrium Y=2.0");
    assert!(callout.tag.is_none());
    assert!(callout.visible);
}

#[test]
fn test_callout_offset_updates() {
    let mut manager = CalloutManager::new();
    let id = manager.add_callout([0.0, 5.0], "Initial State", [20.0, -20.0]);

    let updated = manager.update_offset(id, [35.0, -50.0]);
    assert!(updated);
    assert_eq!(manager.callouts()[0].label_offset, [35.0, -50.0]);

    let non_existent = manager.update_offset(999, [0.0, 0.0]);
    assert!(!non_existent);
}

#[test]
fn test_auto_annotation_idempotency() {
    let mut manager = CalloutManager::new();
    manager.set_auto_callout(
        "initial_condition",
        [0.0, 1.0],
        "Initial (0, 1.00)",
        [-25.0, -25.0],
    );
    assert_eq!(manager.len(), 1);
    assert_eq!(manager.callouts()[0].text, "Initial (0, 1.00)");
    assert_eq!(
        manager.callouts()[0].tag.as_deref(),
        Some("initial_condition")
    );

    // Updating existing auto callout does not duplicate
    manager.set_auto_callout(
        "initial_condition",
        [0.0, 2.5],
        "Initial (0, 2.50)",
        [-25.0, -25.0],
    );
    assert_eq!(manager.len(), 1);
    assert_eq!(manager.callouts()[0].target_point, [0.0, 2.5]);
    assert_eq!(manager.callouts()[0].text, "Initial (0, 2.50)");

    // Add another auto callout
    manager.set_auto_callout(
        "final_state",
        [10.0, 4.0],
        "Final (10.00, 4.00)",
        [25.0, -25.0],
    );
    assert_eq!(manager.len(), 2);

    // Clear auto only removes tagged callouts
    manager.add_callout([5.0, 5.0], "User Manual Point", [0.0, 0.0]);
    assert_eq!(manager.len(), 3);
    manager.clear_auto();
    assert_eq!(manager.len(), 1);
    assert_eq!(manager.callouts()[0].text, "User Manual Point");
}

#[test]
fn test_callout_serde_roundtrip() {
    let mut manager = CalloutManager::new();
    manager.add_callout([1.5, 3.2], "Critical Point", [12.0, -18.0]);
    manager.set_auto_callout("root", [0.0, 0.0], "Origin", [10.0, 10.0]);

    let json = serde_json::to_string(&manager).expect("Serialization failed");
    let restored: CalloutManager = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(restored.len(), 2);
    assert_eq!(restored.callouts()[0].text, "Critical Point");
    assert_eq!(restored.callouts()[0].target_point, [1.5, 3.2]);
    assert_eq!(restored.callouts()[0].label_offset, [12.0, -18.0]);
    assert_eq!(restored.callouts()[1].tag.as_deref(), Some("root"));
}

#[test]
fn test_callout_capacity_limit() {
    let mut manager = CalloutManager::new();
    for i in 0..MAX_CALLOUTS + 5 {
        manager.add_callout([i as f64, i as f64], format!("Callout {}", i), [0.0, 0.0]);
    }
    assert_eq!(manager.len(), MAX_CALLOUTS);
    assert_eq!(manager.callouts()[0].text, "Callout 5");
}

#[test]
fn test_workspace_envelope_serialization() {
    let mut manager = CalloutManager::new();
    manager.add_callout([3.0, 7.0], "Peak", [10.0, -20.0]);

    let app_state = AppState {
        selected_tab: 0,
        selected_tab_name: Some("ODE Solvers".to_string()),
        show_info: true,
        show_warnings: true,
        show_errors: true,
        show_diagnostics_panel: true,
        theme: ThemeMode::Dark,
        tab_states: std::collections::HashMap::new(),
        viewport_layout: crate::app::ViewportLayout::Single,
        pane_tabs: vec![0, 1, 2, 3],
    };

    let envelope = WorkspaceEnvelope {
        schema_version: "1.0".to_string(),
        app_version: "0.1.0".to_string(),
        timestamp: "2026-11-14T00:00:00Z".to_string(),
        app_state,
        history_undo_stack: Vec::new(),
        history_redo_stack: Vec::new(),
        callout_manager: Some(manager.clone()),
    };

    let json = serde_json::to_string(&envelope).expect("Serialize envelope");
    let restored: WorkspaceEnvelope = serde_json::from_str(&json).expect("Deserialize envelope");

    assert!(restored.callout_manager.is_some());
    let restored_mgr = restored.callout_manager.unwrap();
    assert_eq!(restored_mgr.len(), 1);
    assert_eq!(restored_mgr.callouts()[0].text, "Peak");
    assert_eq!(restored_mgr.callouts()[0].target_point, [3.0, 7.0]);
}

#[test]
fn test_generate_callout_screenshot() {
    let datasets = vec![
        (
            "ODE Trajectory y(t)".to_string(),
            (0..=50)
                .map(|i| {
                    let t = i as f64 * 0.1;
                    [t, (t * 0.5).sin() * 5.0 + 10.0]
                })
                .collect(),
        ),
        (
            "Callout [Initial (0, 10.00)]".to_string(),
            vec![[0.0, 10.0]],
        ),
        (
            "Callout [Steady State Y=10.0]".to_string(),
            vec![[3.14, 15.0]],
        ),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/ode_callout_pipeline.png", png);
    }
}
