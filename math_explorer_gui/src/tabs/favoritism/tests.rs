#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use eframe::egui::{self, CentralPanel, Frame};

fn active_favoritism_state(tab: &FavoritismTab) -> FavoritismToolState {
    let json = tab
        .framework
        .active_tool
        .as_ref()
        .and_then(|t| t.save_state())
        .expect("FavoritismTool must serialize state");
    serde_json::from_str(&json).expect("valid FavoritismToolState")
}

#[test]
fn test_favoritism_all_13_controls_and_metadata_update_state_revision_and_roundtrip() {
    let mut tool = FavoritismTool::default();
    let mut prev_rev = tool.state_revision();
    assert!(prev_rev > 0);

    let mutations: [fn(&mut FavoritismTool); 17] = [
        |t| t.children[0].inputs.time.t = 730.0,
        |t| t.children[0].inputs.time.x_0 = 125.0,
        |t| t.children[0].inputs.gifts.g_practical = 4.5,
        |t| t.children[0].inputs.gifts.g_emotional = 9.2,
        |t| t.children[0].inputs.personality.wealth = 5.5,
        |t| t.children[0].inputs.personality.talent = 8.1,
        |t| t.children[0].inputs.personality.intelligence = 9.4,
        |t| t.children[0].inputs.personality.emotional_sensitivity = 6.3,
        |t| t.children[0].inputs.social.helped_during_crisis = false,
        |t| t.children[0].inputs.social.helped_during_crisis = true,
        |t| t.children[0].inputs.social.active_on_social_media = false,
        |t| t.children[0].inputs.social.active_on_social_media = true,
        |t| t.children[0].inputs.social.birth_order_weight = 1.35,
        |t| t.children[0].inputs.contact.time_since_last_contact = 14.0,
        |t| t.children[0].inputs.contact.f_initial = 18.0,
        |t| t.selected_child_index = Some(1),
        |t| t.new_child_name = "Third Sibling".to_string(),
    ];

    for (idx, mutate) in mutations.into_iter().enumerate() {
        mutate(&mut tool);
        tool.children[0].update_score();
        let next_rev = tool.state_revision();
        assert_ne!(
            next_rev, prev_rev,
            "Favoritism mutation #{idx} did not change state_revision()"
        );
        prev_rev = next_rev;
    }

    let saved = tool.save_state().expect("save_state must succeed");
    let mut restored = FavoritismTool::default();
    restored
        .try_load_state(&saved)
        .expect("try_load_state must succeed");
    assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
    assert_eq!(restored.state_revision(), tool.state_revision());
    assert!((restored.children[0].score - tool.children[0].score).abs() < 1e-9);
}

#[test]
fn test_favoritism_clamps_selected_child_index_and_handles_legacy_json() {
    let mut tool = FavoritismTool::default();
    let mut state: FavoritismToolState = serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    state.selected_child_index = Some(99);
    let out_of_range_json = serde_json::to_string(&state).unwrap();
    tool.try_load_state(&out_of_range_json).unwrap();
    assert_eq!(tool.selected_child_index, Some(1));

    state.children.clear();
    state.selected_child_index = Some(5);
    let empty_children_json = serde_json::to_string(&state).unwrap();
    tool.try_load_state(&empty_children_json).unwrap();
    assert_eq!(tool.selected_child_index, None);

    // Verify legacy flat FavoritismState loading on FavoritismTab
    let mut tab = FavoritismTab::default();
    tab.load_state(r#"{"selected_child_index":42,"new_child_name":"Legacy Kid"}"#);
    let loaded = active_favoritism_state(&tab);
    assert_eq!(loaded.selected_child_index, Some(1));
    assert_eq!(loaded.new_child_name, "Legacy Kid");
    assert_eq!(loaded.children.len(), 2);
}

#[test]
fn test_favoritism_single_tool_hides_framework_sidebar_and_exports_data_and_png() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tab = FavoritismTab::default();
    let mut frame = eframe::Frame::_new_kittest();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));

    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    );

    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let has_redundant_tools_heading = ak
        .nodes
        .iter()
        .any(|(_, n)| n.label() == Some("Tools") && n.role() == egui::accesskit::Role::Heading);
    assert!(
        !has_redundant_tools_heading,
        "Single-tool FavoritismTab must hide redundant Tools sidebar"
    );

    // Export CSV, JSON, and PNG
    for fmt in [ExportFormat::Csv, ExportFormat::Json, ExportFormat::Png] {
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen_rect),
                ..Default::default()
            },
            |ctx| {
                tab.framework.queue_export(fmt);
                CentralPanel::default()
                    .frame(Frame::NONE)
                    .show(ctx, |ui| tab.show_inside(ui, &mut frame));
            },
        );
        assert_eq!(
            tab.framework.export_status_severity,
            ExportStatusSeverity::Success
        );
    }
    let _ = std::fs::write(
        "/tmp/favoritism_leaderboard.png",
        tab.framework
            .export_queue
            .last_exported_bytes
            .as_ref()
            .expect("favoritism PNG bytes"),
    );
}

fn click_button_by_label(ctx: &egui::Context, tab: &mut FavoritismTab, label_substr: &str) {
    let mut frame = eframe::Frame::_new_kittest();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        },
        |ctx| {
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    );
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let bounds = ak
        .nodes
        .iter()
        .find(|(_, n)| n.label().is_some_and(|l| l.contains(label_substr)))
        .and_then(|(_, n)| n.bounds())
        .unwrap_or_else(|| panic!("Button containing '{label_substr}' not found"));
    let center = egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    );
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            events: vec![
                egui::Event::PointerMoved(center),
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos: center,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| {
            CentralPanel::default()
                .frame(Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    );
}

#[test]
fn test_favoritism_reset_live_region_theory_and_preset_io() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut tab = FavoritismTab::default();
    tab.load_state(r#"{"selected_child_index":1,"new_child_name":"Mutated Sibling"}"#);
    assert_eq!(
        active_favoritism_state(&tab).new_child_name,
        "Mutated Sibling"
    );
    click_button_by_label(&ctx, &mut tab, "Reset Parameters");
    assert_eq!(
        active_favoritism_state(&tab).children[0].name,
        "Golden Child"
    );
    assert_eq!(active_favoritism_state(&tab).new_child_name, "New Child");
    let live_msg = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert_eq!(live_msg, "Parameters reset to baseline defaults");

    assert!(tab
        .framework
        .apply_starter_card(
            "Crisis & Proximity Showdown",
            Some("FavoritismTool"),
            Some(crate::starter_gallery::PRESET_FAVORITISM_SIBLING_RIVALRY),
        )
        .is_ok());
    let preset_json = tab
        .framework
        .active_tool
        .as_ref()
        .unwrap()
        .export_preset_json()
        .expect("preset JSON export");
    let mut fresh_tab = FavoritismTab::default();
    fresh_tab
        .framework
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&preset_json)
        .expect("preset JSON import");
    assert_eq!(
        active_favoritism_state(&fresh_tab).children[0].name,
        "Crisis Hero"
    );

    assert!(!tab.framework.show_theory_portal);
    click_button_by_label(&ctx, &mut tab, "Theory");
    assert!(tab.framework.show_theory_portal);
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        CentralPanel::default()
            .frame(Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
    assert!(
        scientific_metadata::citation_registry::CitationRegistry::get("Favoritism")
            .is_some_and(|c| c.contains("Unified Favoritism Theory"))
    );
}

#[test]
fn test_favoritism_workspace_undo_and_mws_envelope() {
    let ctx = egui::Context::default();
    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = crate::app::WorkspaceSessionManager::default();
    let fav_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Favoritism")
        .expect("Favoritism tab registered");
    app.selected_tab = fav_idx;
    mgr.track_frame_changes(&ctx, &app);
    assert!(!mgr.can_undo());

    let baseline_tab_json = app.tabs[fav_idx].save_state().unwrap();
    app.tabs[fav_idx].load_state(r#"{"selected_child_index":0,"new_child_name":"Undo Check"}"#);
    mgr.track_frame_changes(&ctx, &app);
    assert!(mgr.can_undo());

    let mws = mgr.create_envelope(app.get_app_state());
    let parsed = mgr.parse_envelope(&mws).unwrap();
    assert!(parsed.tab_states["Favoritism"].contains("Undo Check"));

    let prev = mgr.pop_undo().expect("undo snapshot");
    assert_eq!(prev.tab_states["Favoritism"], baseline_tab_json);
}
