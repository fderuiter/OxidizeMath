#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use eframe::App;

fn run_screen_events(ctx: &egui::Context, app: &mut MathExplorerApp, events: Vec<egui::Event>) {
    let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1280.0, 900.0));
    let mut frame = eframe::Frame::_new_kittest();
    let input = egui::RawInput {
        screen_rect: Some(screen),
        events,
        ..Default::default()
    };
    let _ = ctx.run(input, |c| app.update(c, &mut frame));
}

fn pane_tool_state(app: &mut MathExplorerApp, pane: usize) -> String {
    app.pane_tab_mut(pane)
        .unwrap()
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state()
        .unwrap()
}

#[test]
fn test_quad_focus_click_routes_domain_bar() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    if app.tabs.len() < 2 {
        return;
    }
    app.set_layout(ViewportLayout::QuadGrid);
    for p in 0..4 {
        app.set_pane_tab(p, 0);
    }
    run_screen_events(
        &ctx,
        &mut app,
        vec![egui::Event::PointerButton {
            pos: egui::pos2(200.0, 750.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert_eq!(app.focused_pane, 2);
    domain_bar::select_tab(&mut app, &ctx, 1);
    assert_eq!(app.pane_tabs[2], 1);
    assert_eq!(app.pane_tabs[0], 0);
}

#[test]
fn test_quad_typing_focus_routes_command_palette() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") else {
        return;
    };
    app.set_layout(ViewportLayout::QuadGrid);
    for p in 0..4 {
        app.set_pane_tab(p, 0);
    }
    run_screen_events(
        &ctx,
        &mut app,
        vec![
            egui::Event::PointerMoved(egui::pos2(950.0, 750.0)),
            egui::Event::Text("x".to_string()),
        ],
    );
    assert_eq!(app.focused_pane, 3);
    app.command_palette.open();
    app.command_palette.query = "NetworkPropagationTool".to_string();
    run_screen_events(&ctx, &mut app, Vec::new());
    let enter = egui::Event::Key {
        key: egui::Key::Enter,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    run_screen_events(&ctx, &mut app, vec![enter]);
    assert_eq!(app.pane_tabs[3], epi_idx);
    assert_eq!(app.pane_tabs[0], 0);
    let tool_name = app
        .pane_tab(3)
        .unwrap()
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .name();
    assert_eq!(tool_name, "Network Propagation");
}

#[test]
fn test_same_domain_swap_panes_and_session_undo_redo_persistence() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") else {
        return;
    };
    app.set_layout(ViewportLayout::SplitVertical);
    app.set_pane_tab(0, epi_idx);
    app.set_pane_tab(1, epi_idx);
    app.focus_pane(1);
    for (pane, beta) in [(0, 0.25), (1, 2.75)] {
        let fw = app.pane_tab_mut(pane).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("SirTool"));
        fw.active_tool.as_mut().unwrap().load_state(&format!(
            r#"{{"n":1000.0,"i0":10.0,"beta":{beta},"gamma":0.1,"duration":100.0}}"#
        ));
    }
    run_screen_events(&ctx, &mut app, Vec::new());
    app.swap_panes();
    let swapped_state = app.get_app_state();
    app.session_manager.track_state_changes(&ctx, swapped_state);
    assert!(pane_tool_state(&mut app, 0).contains("\"beta\":2.75"));
    assert!(pane_tool_state(&mut app, 1).contains("\"beta\":0.25"));

    let saved_json = app.save_state_to_json();
    let mut restored = MathExplorerApp::default();
    restored.load_state_from_json(&saved_json);
    assert_eq!(restored.focused_pane, 1);
    assert!(pane_tool_state(&mut restored, 0).contains("\"beta\":2.75"));
    assert!(pane_tool_state(&mut restored, 1).contains("\"beta\":0.25"));

    assert!(app.session_manager.can_undo());
    app.undo();
    assert!(pane_tool_state(&mut app, 0).contains("\"beta\":0.25"));
    assert!(pane_tool_state(&mut app, 1).contains("\"beta\":2.75"));
    assert!(app.session_manager.can_redo());
    app.redo();
    assert!(pane_tool_state(&mut app, 0).contains("\"beta\":2.75"));
    assert!(pane_tool_state(&mut app, 1).contains("\"beta\":0.25"));
}

#[test]
fn test_single_pane_startup_allocates_only_primary_catalog() {
    let app = MathExplorerApp::default();
    assert_eq!(app.viewport_layout, ViewportLayout::Single);
    assert_eq!(app.active_secondary_slot_count(), 0);
    assert_eq!(app.focused_pane, 0);
}

#[test]
fn test_quad_pane_contexts_inherit_active_theme_palette() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);
    for p in 0..4 {
        app.set_pane_tab(p, p % app.tabs.len().max(1));
    }

    for mode in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        app.theme = mode;
        run_screen_events(&ctx, &mut app, Vec::new());
        for (pane_idx, pane_ctx) in app.pane_contexts.iter().enumerate() {
            let pane_visuals = &pane_ctx.style().visuals;
            assert_eq!(
                ThemeMode::from_visuals(pane_visuals),
                mode,
                "pane {pane_idx} did not inherit ThemeMode::{mode:?}"
            );
            assert_eq!(
                SemanticPalette::from_visuals(pane_visuals),
                mode.palette(),
                "pane {pane_idx} did not resolve SemanticPalette for {mode:?}"
            );
        }
    }
}
