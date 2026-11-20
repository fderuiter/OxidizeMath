#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::egui;
use eframe::App;

fn shortcut_mods() -> egui::Modifiers {
    if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    }
}

fn key_shortcut_event(key: egui::Key, modifiers: egui::Modifiers) -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }]
}

#[test]
fn test_single_pane_input_guard_preserves_workspace_undo() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mods = shortcut_mods();

    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    app.theme = ThemeMode::Light;
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    assert!(app.session_manager.can_undo());

    let mut text = String::from("sin(x)");
    let te_id = egui::Id::new("test_formula_text_edit");
    let _ = ctx.run(
        egui::RawInput {
            events: key_shortcut_event(egui::Key::Z, mods),
            ..Default::default()
        },
        |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let resp = ui.add(egui::TextEdit::singleline(&mut text).id(te_id));
                resp.request_focus();
            });
            app.update(c, &mut frame);
        },
    );
    assert_eq!(app.theme, ThemeMode::Light);
    assert!(has_cmd(&get_registered_commands(&ctx), "Edit", "Undo"));
}

#[cfg(feature = "pure_math")]
fn focus_secondary_pane_custom_input(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
) {
    let analysis_idx = find_tab_index(app, "Analysis & Calculus");
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(1, analysis_idx);
    {
        let fw = app.pane_tab_mut(1).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("ComplexMappingTool"));
        fw.active_tool
            .as_mut()
            .unwrap()
            .try_load_state(
                r#"{"function":"Custom","grid_min":-2.0,"grid_max":2.0,"grid_density":10,"inspection_point":[1.0,1.0]}"#,
            )
            .unwrap();
    }
    let split_out = run_frame_with_rect(ctx, app, frame, Vec::new());
    let pane1_input_node = split_out
        .platform_output
        .accesskit_update
        .as_ref()
        .unwrap()
        .nodes
        .iter()
        .find_map(|(id, n)| {
            (n.role() == egui::accesskit::Role::TextInput && n.bounds()?.x0 >= 600.0).then_some(*id)
        })
        .expect("Secondary pane Custom ComplexMappingTool must expose TextInput");
    let _ = run_frame_with_rect(
        ctx,
        app,
        frame,
        vec![egui::Event::AccessKitActionRequest(
            egui::accesskit::ActionRequest {
                action: egui::accesskit::Action::Focus,
                target: pane1_input_node,
                data: None,
            },
        )],
    );
}

#[cfg(feature = "pure_math")]
#[test]
fn test_split_pane_input_guard_and_focus_surrender_on_single_layout() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mods = shortcut_mods();

    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    app.theme = ThemeMode::Light;
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    focus_secondary_pane_custom_input(&ctx, &mut app, &mut frame);
    assert!(ctx.wants_keyboard_input());
    app.theme = ThemeMode::HighContrast;
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    let _ = run_frame_with_rect(
        &ctx,
        &mut app,
        &mut frame,
        key_shortcut_event(egui::Key::Z, mods),
    );
    assert_eq!(app.theme, ThemeMode::HighContrast);

    app.set_layout(ViewportLayout::Single);
    let _ = run_frame_with_rect(
        &ctx,
        &mut app,
        &mut frame,
        key_shortcut_event(egui::Key::Z, mods),
    );
    assert_eq!(app.theme, ThemeMode::Light);
    if !cfg!(target_os = "macos") {
        let _ = run_frame_with_rect(
            &ctx,
            &mut app,
            &mut frame,
            key_shortcut_event(egui::Key::Y, mods),
        );
        assert_eq!(app.theme, ThemeMode::HighContrast);
        assert_eq!(app.viewport_layout, ViewportLayout::Single);
    }
}

#[cfg(feature = "pure_math")]
#[test]
fn test_complex_mapping_custom_formula_workspace_roundtrip() {
    let mut app = MathExplorerApp::default();
    let analysis_idx = find_tab_index(&app, "Analysis & Calculus");
    app.selected_tab = analysis_idx;
    {
        let fw = app.tabs[analysis_idx].framework_mut().unwrap();
        assert!(fw.select_tool_by_name("ComplexMappingTool"));
        fw.active_tool
            .as_mut()
            .unwrap()
            .try_load_state(
                r#"{"function":"Custom","grid_min":-2.0,"grid_max":2.0,"grid_density":12,"inspection_point":[1.0,0.5],"custom_expr":"exp(z) - z^2"}"#,
            )
            .expect("valid custom ComplexMappingState");
    }

    let env_json = app.session_manager.create_envelope(app.get_app_state());
    let mut restored = MathExplorerApp::default();
    restored.load_state_from_json(&env_json);
    let fw = restored.tabs[analysis_idx].framework().unwrap();
    let saved_tool = fw.active_tool.as_ref().unwrap().save_state().unwrap();
    assert!(saved_tool.contains(r#""custom_expr":"exp(z) - z^2""#));
}

#[cfg(feature = "pure_math")]
fn add_graph_node_in_pane_zero(app: &mut MathExplorerApp) {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0));
    let fw = app.pane_tab_mut(0).unwrap().framework_mut().unwrap();
    let tool = fw
        .active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<crate::tabs::graph_theory::graph_editor::GraphEditorTool>()
        .unwrap();
    tool.insert_node_from_keyboard(rect);
}

#[cfg(feature = "pure_math")]
#[test]
fn test_graph_editor_shortcuts_split_no_replay_and_session_fallback() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mods = shortcut_mods();

    let graph_idx = find_tab_index(&app, "Graph Theory");
    app.select_tab(&ctx, graph_idx);
    assert!(app.tabs[graph_idx].select_tool_by_name("GraphEditorTool"));
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    app.theme = ThemeMode::HighContrast;
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, graph_idx);
    app.focus_pane(0);
    app.pane_tab_mut(0)
        .unwrap()
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(r#"{"node_positions":{},"edges":[],"selected_node":null,"next_node_id":0}"#)
        .unwrap();
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    for _ in 0..2 {
        add_graph_node_in_pane_zero(&mut app);
        let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    }
    app.session_manager.push_redo(app.get_app_state());
    assert!(app.can_undo());

    let _ = run_frame_with_rect(
        &ctx,
        &mut app,
        &mut frame,
        key_shortcut_event(egui::Key::Z, mods),
    );
    assert!(app
        .pane_tab(0)
        .unwrap()
        .save_state()
        .unwrap()
        .contains(r#""next_node_id":1"#));
    assert!(app.session_manager.can_redo() && app.can_redo());

    let shift_z = key_shortcut_event(egui::Key::Z, mods | egui::Modifiers::SHIFT);
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, shift_z);
    assert!(app
        .pane_tab(0)
        .unwrap()
        .save_state()
        .unwrap()
        .contains(r#""next_node_id":2"#));
    for _ in 0..4 {
        let _ = run_frame_with_rect(
            &ctx,
            &mut app,
            &mut frame,
            key_shortcut_event(egui::Key::Z, mods),
        );
    }
    assert_eq!(app.theme, ThemeMode::Dark);
}
