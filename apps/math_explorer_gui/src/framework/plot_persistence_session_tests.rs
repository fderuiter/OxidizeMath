#![cfg_attr(any(), verified(opt_out = "tests"))]

use eframe::egui;
use egui_plot::PlotOverlaySnapshot;

use super::plot_persistence_tests::rich_test_snapshot;
use crate::app::{AppState, MathExplorerApp, ViewportLayout, WorkspaceSessionManager};
#[cfg(feature = "physics")]
use crate::framework::InteractiveTool;

fn setup_split_app_with_overlay(
    ctx: &egui::Context,
) -> Option<(
    MathExplorerApp,
    WorkspaceSessionManager,
    usize,
    PlotOverlaySnapshot,
)> {
    let mut app = MathExplorerApp::default();
    let fw_tab_idx = app.tabs.iter().position(|t| t.framework().is_some())?;
    app.select_tab(ctx, fw_tab_idx);
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, fw_tab_idx);
    app.set_pane_tab(1, fw_tab_idx);

    let mut mgr = WorkspaceSessionManager::new();
    mgr.track_frame_changes(ctx, &app);

    let snap_primary = rich_test_snapshot();
    {
        let fw = app.tabs[fw_tab_idx].framework_mut().unwrap();
        let tool_name = fw.active_tool.as_ref().unwrap().name().to_string();
        fw.saved_plot_states
            .entry(tool_name)
            .or_default()
            .insert("primary_plot".to_string(), snap_primary.clone());
        fw.bump_plot_sync_generation();
        fw.bump_framework_revision();
    }
    mgr.track_frame_changes(ctx, &app);
    Some((app, mgr, fw_tab_idx, snap_primary))
}

#[test]
fn test_app_mws_and_json_autosave_roundtrip() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let Some((app, mgr, fw_tab_idx, snap_primary)) = setup_split_app_with_overlay(&ctx) else {
        return;
    };

    let mws_json = mgr.create_envelope(app.get_app_state());
    let mut loaded_mgr = WorkspaceSessionManager::new();
    let parsed_mws = loaded_mgr.parse_envelope(&mws_json).expect("parse .mws");
    let mut mws_app = MathExplorerApp::default();
    mws_app.apply_app_state(parsed_mws);
    assert_eq!(
        mws_app.tabs[fw_tab_idx]
            .framework()
            .unwrap()
            .saved_plot_states
            .values()
            .next()
            .and_then(|m| m.get("primary_plot")),
        Some(&snap_primary)
    );

    let raw_app_json = serde_json::to_string(&app.get_app_state()).expect("serialize AppState");
    let raw_parsed: AppState = serde_json::from_str(&raw_app_json).expect("deserialize AppState");
    let mut autosave_app = MathExplorerApp::default();
    autosave_app.apply_app_state(raw_parsed);
    assert_eq!(
        autosave_app.tabs[fw_tab_idx]
            .framework()
            .unwrap()
            .saved_plot_states
            .values()
            .next()
            .and_then(|m| m.get("primary_plot")),
        Some(&snap_primary)
    );
}

#[test]
fn test_app_undo_redo_and_webgl_recovery_roundtrip() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let Some((mut app, mut mgr, fw_tab_idx, snap_primary)) = setup_split_app_with_overlay(&ctx)
    else {
        return;
    };
    assert!(
        mgr.can_undo(),
        "plot overlay edit must record an undo snapshot"
    );

    let recovery = app.capture_recovery_snapshot();
    let fresh_ctx = egui::Context::default();
    let mut recovered_app = MathExplorerApp::default();
    recovered_app.restore_recovery_snapshot(&recovery, &fresh_ctx);
    assert_eq!(
        recovered_app.tabs[fw_tab_idx]
            .framework()
            .unwrap()
            .saved_plot_states
            .values()
            .next()
            .and_then(|m| m.get("primary_plot")),
        Some(&snap_primary)
    );

    mgr.execute_undo(&mut app);
    assert!(app.tabs[fw_tab_idx]
        .framework()
        .unwrap()
        .saved_plot_states
        .is_empty());

    mgr.execute_redo(&mut app);
    assert_eq!(
        app.tabs[fw_tab_idx]
            .framework()
            .unwrap()
            .saved_plot_states
            .values()
            .next()
            .and_then(|m| m.get("primary_plot")),
        Some(&snap_primary)
    );
}

#[cfg(feature = "physics")]
#[test]
fn test_attractor_plotter_persists_and_validates_fixed_callout_offsets() {
    use crate::tabs::chaos::attractors::{AttractorPlotter, DEFAULT_FIXED_CALLOUT_OFFSETS};

    let mut plotter = AttractorPlotter::default();
    let default_json = plotter.save_state().expect("serialize default attractor");
    assert!(!default_json.contains("fixed_callout_offsets"));

    let rev_before = plotter.state_revision();
    plotter.fixed_callout_offsets = [egui::Vec2::new(64.0, -48.0), egui::Vec2::new(-72.0, 55.0)];
    assert_ne!(plotter.state_revision(), rev_before);

    plotter.reset();
    assert_eq!(
        plotter.fixed_callout_offsets,
        [egui::Vec2::new(64.0, -48.0), egui::Vec2::new(-72.0, 55.0)]
    );

    let custom_json = plotter.save_state().expect("serialize custom offsets");
    assert!(custom_json.contains("fixed_callout_offsets"));

    let mut restored = AttractorPlotter::default();
    restored
        .try_load_state(&custom_json)
        .expect("restore dragged fixed_callout_offsets");
    assert_eq!(
        restored.fixed_callout_offsets,
        [egui::Vec2::new(64.0, -48.0), egui::Vec2::new(-72.0, 55.0)]
    );

    restored.reset_parameters();
    assert_eq!(
        restored.fixed_callout_offsets,
        DEFAULT_FIXED_CALLOUT_OFFSETS.map(egui::Vec2::from)
    );

    let mut bad_val: serde_json::Value = serde_json::from_str(&custom_json).unwrap();
    bad_val["fixed_callout_offsets"] = serde_json::json!([[null, 1.0], [2.0, 3.0]]);
    assert!(restored.try_load_state(&bad_val.to_string()).is_err());
}

#[test]
fn test_export_restored_session_overlay_figure() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let Some((app, mgr, fw_tab_idx, expected)) = setup_split_app_with_overlay(&ctx) else {
        return;
    };
    let mws_json = mgr.create_envelope(app.get_app_state());
    let mut restored_mgr = WorkspaceSessionManager::new();
    let parsed = restored_mgr.parse_envelope(&mws_json).expect("parse .mws");
    let mut restored_app = MathExplorerApp::default();
    restored_app.apply_app_state(parsed);

    let restored = restored_app.tabs[fw_tab_idx]
        .framework()
        .and_then(|fw| fw.saved_plot_states.values().next())
        .and_then(|m| m.get("primary_plot"))
        .expect("restored primary_plot snapshot");
    assert_eq!(restored, &expected);

    let datasets = vec![
        (
            "Model Trajectory y(x)".to_string(),
            (1..=40).map(|i| [i as f64 * 2.0, i as f64 * 4.0]).collect(),
        ),
        (
            restored.imported_datasets[0].name.clone(),
            restored.imported_datasets[0].points.clone(),
        ),
        (
            format!("Callout [{}]", restored.callouts[0].label),
            vec![restored.callouts[0].point],
        ),
    ];
    let opts = egui_plot::export::ExportOptions::default();
    if let Ok(svg) = egui_plot::export::export_custom_svg(&datasets, &opts) {
        let _ = std::fs::write("/tmp/restored_plot_overlay.svg", svg);
    }
    let _ = std::fs::write(
        "/tmp/restored_plot_overlay_snapshot.json",
        serde_json::to_string_pretty(restored).unwrap_or_default(),
    );
}
