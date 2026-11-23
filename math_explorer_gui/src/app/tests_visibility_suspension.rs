//! Integration tests for viewport visibility suspension, tool switching suspension,
//! logical `SimWorkerState::Running` session preservation, and `MorphogenesisTab::default` deduplication.

use crate::app::domain_bar;
use crate::app::state::AppState;
use crate::app::{MathExplorerApp, ViewportLayout};
use crate::tabs::morphogenesis::MorphogenesisTab;
use crate::tabs::ExplorerTab;
use eframe::egui;

fn find_tab_index(app: &MathExplorerApp, tab_name: &str) -> usize {
    app.tabs
        .iter()
        .position(|t| t.name() == tab_name)
        .unwrap_or_else(|| panic!("Expected tab '{tab_name}' in MathExplorerApp::tabs"))
}

fn load_tab_with_running_active_tool(tab: &mut dyn ExplorerTab) {
    let active_key = tab
        .framework()
        .and_then(|fw| fw.active_tool.as_ref())
        .map(|t| t.name().to_string())
        .expect("Expected active tool on framework tab");
    let base = tab.save_state().expect("Framework tab should save JSON");
    let mut root: serde_json::Value = serde_json::from_str(&base).unwrap();
    let tool_entry = &mut root["tool_states"][&active_key];
    if let Some(obj) = tool_entry.as_object_mut() {
        obj.insert(
            "state".to_string(),
            serde_json::Value::String("Running".to_string()),
        );
    } else if let Some(s) = tool_entry.as_str() {
        let mut inner: serde_json::Value = serde_json::from_str(s).unwrap();
        inner["state"] = serde_json::Value::String("Running".to_string());
        *tool_entry = serde_json::Value::String(serde_json::to_string(&inner).unwrap());
    }
    tab.load_state(&serde_json::to_string(&root).unwrap());
}

#[test]
fn default_app_suspends_hidden_tabs_and_deduplicates_morphogenesis_init() {
    let morph = MorphogenesisTab::default();
    assert_eq!(
        morph.framework.active_tool.as_ref().map(|t| t.name()),
        Some("Turing Morphogenesis")
    );
    let saved_json = morph
        .framework
        .save_state()
        .expect("MorphogenesisTab::default should serialize framework state");
    assert!(
        !saved_json.contains("Lenia Continuous CA"),
        "MorphogenesisTab::default must reuse initial MorphogenesisTool without stashing Lenia in saved_tool_states: {saved_json}"
    );

    let app = MathExplorerApp::default();
    assert_eq!(app.active_primary_tab_index(), 0);

    let mri_idx = find_tab_index(&app, "MRI Physics");
    assert_ne!(mri_idx, 0);
    assert!(
        app.tabs[mri_idx].framework().unwrap().is_suspended(),
        "Non-primary tabs must start suspended in MathExplorerApp::default()"
    );
    if let Some(primary_fw) = app.tabs[0].framework() {
        assert!(
            !primary_fw.is_suspended(),
            "Primary visible tab must start unsuspended"
        );
    }
}

#[test]
fn tab_switch_suspends_running_worker_preserves_json_state_and_resumes_on_return() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let solid_idx = find_tab_index(&app, "Solid State Physics");

    domain_bar::select_tab(&mut app, &ctx, solid_idx);
    assert!(!app.tabs[solid_idx].framework().unwrap().is_suspended());
    load_tab_with_running_active_tool(app.tabs[solid_idx].as_mut());
    assert!(app.tabs[solid_idx]
        .save_state()
        .unwrap()
        .contains("Running"));

    // Switch away to tab 0: hidden Solid State tab suspends immediately while preserving Running in JSON.
    domain_bar::select_tab(&mut app, &ctx, 0);
    assert!(app.tabs[solid_idx].framework().unwrap().is_suspended());
    let saved_while_hidden = app.save_state_to_json();
    assert!(
        saved_while_hidden.contains("Running"),
        "Hidden suspended tab must preserve logical Running state in serialized session JSON"
    );

    // Restoring that session JSON via AppState::apply keeps hidden tabs suspended while retaining Running.
    let parsed: AppState = serde_json::from_str(&saved_while_hidden).expect("Valid AppState JSON");
    let mut restored_app = MathExplorerApp::default();
    parsed.apply(&mut restored_app);
    assert!(restored_app.tabs[solid_idx]
        .framework()
        .unwrap()
        .is_suspended());
    let re_saved = restored_app.save_state_to_json();
    assert!(
        re_saved.contains("Running"),
        "AppState::apply on hidden tab must preserve Running metadata without unsuspending"
    );

    // Switching back to Solid State unsuspends it automatically and preserves Running state.
    domain_bar::select_tab(&mut restored_app, &ctx, solid_idx);
    assert!(!restored_app.tabs[solid_idx]
        .framework()
        .unwrap()
        .is_suspended());
    assert!(restored_app.tabs[solid_idx]
        .save_state()
        .unwrap()
        .contains("Running"));
}

#[test]
fn tool_switch_stashes_running_state_and_respects_framework_suspension() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mri_idx = find_tab_index(&app, "MRI Physics");
    domain_bar::select_tab(&mut app, &ctx, mri_idx);

    assert!(app.tabs[mri_idx].select_tool_by_name("MriTool"));
    load_tab_with_running_active_tool(app.tabs[mri_idx].as_mut());
    assert!(!app.tabs[mri_idx].framework().unwrap().is_suspended());

    // Switch to another tool in the same visible tab: outgoing tool state stays Running in JSON,
    // and returning to MriTool restores Running on the visible framework.
    app.tabs[mri_idx].framework_mut().unwrap().select_tool(1);
    assert!(!app.tabs[mri_idx].framework().unwrap().is_suspended());
    assert!(app.tabs[mri_idx].save_state().unwrap().contains("Running"));
    assert!(app.tabs[mri_idx].select_tool_by_name("MriTool"));
    assert!(app.tabs[mri_idx].save_state().unwrap().contains("Running"));
}

#[test]
fn multi_pane_layouts_and_swap_panes_keep_only_visible_panes_active() {
    let mut app = MathExplorerApp::default();
    let mri_idx = find_tab_index(&app, "MRI Physics");
    let morph_idx = find_tab_index(&app, "Morphogenesis");
    let quantum_idx = find_tab_index(&app, "Quantum Mechanics");
    let chaos_idx = find_tab_index(&app, "Chaos Theory");

    app.set_layout(ViewportLayout::QuadGrid);
    app.set_pane_tab(0, mri_idx);
    app.set_pane_tab(1, morph_idx);
    app.set_pane_tab(2, quantum_idx);
    app.set_pane_tab(3, chaos_idx);
    for pane in 1..=3 {
        let _ = app.pane_tab_mut(pane);
    }
    app.sync_viewport_visibility();

    for (idx, tab) in app.tabs.iter().enumerate() {
        if let Some(fw) = tab.framework() {
            assert_eq!(
                fw.is_suspended(),
                idx != mri_idx,
                "Only active_primary_tab_index ({mri_idx}) in app.tabs may be unsuspended (idx={idx})"
            );
        }
    }
    for pane in 1..=3 {
        let slot_tab = &app
            .pane_slots
            .slot(pane)
            .unwrap()
            .active
            .as_ref()
            .unwrap()
            .1;
        assert!(!slot_tab.framework().unwrap().is_suspended());
    }

    // Start MRI in Pane 0, then swap Pane 0 <-> Pane 1 in SplitHorizontal.
    app.set_layout(ViewportLayout::SplitHorizontal);
    assert!(app.pane_tab_mut(0).unwrap().select_tool_by_name("MriTool"));
    load_tab_with_running_active_tool(app.pane_tab_mut(0).unwrap().as_mut());
    app.swap_panes();
    assert_eq!(app.active_primary_tab_index(), morph_idx);
    assert!(app.tabs[mri_idx].framework().unwrap().is_suspended());

    let (_, ref pane1_tab) = app.pane_slots.slot(1).unwrap().active.as_ref().unwrap();
    assert_eq!(pane1_tab.name(), "MRI Physics");
    assert!(!pane1_tab.framework().unwrap().is_suspended());
    assert!(pane1_tab.save_state().unwrap().contains("Running"));

    // Collapsing to Single leaves only Pane 0 unsuspended.
    app.set_layout(ViewportLayout::Single);
    assert!(!app.tabs[morph_idx].framework().unwrap().is_suspended());
    assert!(app.tabs[mri_idx].framework().unwrap().is_suspended());
}
