#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
#![allow(clippy::field_reassign_with_default)]

use super::*;
use eframe::egui;
use eframe::App;
use egui_plot::commands::{CommandMetadata, CommandRegistryData, CommandTrigger};

fn run_frame_with_rect(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0, 800.0),
        )),
        events,
        ..Default::default()
    };
    ctx.run(raw_input, |ctx| {
        app.update(ctx, frame);
    })
}

fn get_registered_commands(ctx: &egui::Context) -> Vec<CommandMetadata> {
    ctx.data(|d| {
        d.get_temp::<CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
            .unwrap_or_default()
            .commands
    })
}

fn find_tab_index(app: &MathExplorerApp, tab_name: &str) -> usize {
    app.tabs
        .iter()
        .position(|t| t.name() == tab_name)
        .unwrap_or_else(|| panic!("Expected tab '{}' to be registered", tab_name))
}

fn has_cmd(cmds: &[CommandMetadata], context: &str, name: &str) -> bool {
    cmds.iter().any(|c| c.context == context && c.name == name)
}

#[test]
fn test_global_and_session_shortcuts_survive_framework_tab_with_descriptions() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let analysis_idx = find_tab_index(&app, "Analysis & Calculus");
    app.select_tab(&ctx, analysis_idx);
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    let cmds = get_registered_commands(&ctx);
    assert!(has_cmd(&cmds, "Global", "Command Palette"));
    assert!(has_cmd(&cmds, "Global", "Keyboard Shortcuts"));
    assert!(has_cmd(&cmds, "Global", "Toggle Issues & Diagnostics"));
    assert!(has_cmd(&cmds, "Edit", "Undo"));
    assert!(has_cmd(&cmds, "Edit", "Redo"));
    assert!(has_cmd(&cmds, "File", "Open Workspace Session..."));
    assert!(has_cmd(&cmds, "File", "Save Workspace Session"));
    assert!(has_cmd(&cmds, "File", "Save Workspace Session As..."));

    let question_registered = cmds.iter().any(|c| {
        c.context == "Global"
            && c.name == "Keyboard Shortcuts"
            && c.trigger == CommandTrigger::Key(egui::Key::Questionmark)
    });
    assert!(question_registered);

    for cmd in &cmds {
        assert!(
            !cmd.description.trim().is_empty(),
            "Shortcut '{}' in '{}' must have a non-empty description",
            cmd.name,
            cmd.context
        );
        let label = cmd.trigger.display_label();
        assert!(!label.contains("Modifiers"));
        assert!(!label.contains("Questionmark"));
    }
}

#[test]
fn test_plot_shortcuts_appear_unfocused_in_framework_and_standalone_tabs() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    // Framework tab with populated Plot datasets
    let analysis_idx = find_tab_index(&app, "Analysis & Calculus");
    app.select_tab(&ctx, analysis_idx);
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    let framework_cmds = get_registered_commands(&ctx);
    assert!(has_cmd(
        &framework_cmds,
        "Plot Accessibility",
        "Next Dataset"
    ));
    assert!(has_cmd(
        &framework_cmds,
        "Plot Accessibility",
        "Previous Dataset"
    ));
    assert!(has_cmd(&framework_cmds, "Plot Accessibility", "Next Point"));
    assert!(has_cmd(
        &framework_cmds,
        "Plot Accessibility",
        "Previous Point"
    ));

    // Unfocused ArrowDown must not trigger plot dataset navigation announcement
    let down_event = vec![egui::Event::Key {
        key: egui::Key::ArrowDown,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }];
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, down_event);
    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(aria_msg
        .as_deref()
        .is_none_or(|m| !m.starts_with("Dataset ")));

    // Standalone ExplorerTab (Favoritism) with populated Plot + Legend
    let fav_idx = find_tab_index(&app, "Favoritism");
    app.select_tab(&ctx, fav_idx);
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    let standalone_cmds = get_registered_commands(&ctx);
    assert!(has_cmd(&standalone_cmds, "Global", "Keyboard Shortcuts"));
    assert!(has_cmd(
        &standalone_cmds,
        "Plot Accessibility",
        "Next Dataset"
    ));
    assert!(has_cmd(
        &standalone_cmds,
        "Plot Accessibility",
        "Previous Dataset"
    ));
    assert!(has_cmd(&standalone_cmds, "Plot Legend", "Isolate Plot"));
}

#[test]
fn test_camera_3d_shortcuts_appear_unfocused_and_movement_stays_focus_gated() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let chaos_idx = find_tab_index(&app, "Chaos Theory");
    app.select_tab(&ctx, chaos_idx);
    assert!(app.tabs[chaos_idx].select_tool_by_name("AttractorPlotter"));
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    let cmds = get_registered_commands(&ctx);
    for expected in [
        "Pitch Up",
        "Pitch Down",
        "Yaw Left",
        "Yaw Right",
        "Zoom Out",
        "Zoom In",
    ] {
        assert!(
            has_cmd(&cmds, "Camera 3D", expected),
            "Expected Camera 3D shortcut '{}' to be registered unfocused",
            expected
        );
    }
    assert!(has_cmd(&cmds, "Global", "Command Palette"));

    // Direct Camera3D check: unfocused key_down registers commands without moving pitch
    let cam_ctx = egui::Context::default();
    let mut camera = crate::camera::Camera3D::default();
    let initial_pitch = camera.pitch;
    let raw_w = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::W,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let _ = cam_ctx.run(raw_w, |sub_ctx| {
        egui::CentralPanel::default().show(sub_ctx, |ui| {
            let (_, resp) = ui.allocate_exact_size(egui::vec2(200.0, 200.0), egui::Sense::click());
            camera.handle_interaction(&resp, ui);
        });
    });
    assert!((camera.pitch - initial_pitch).abs() < f32::EPSILON);
    assert!(has_cmd(
        &get_registered_commands(&cam_ctx),
        "Camera 3D",
        "Pitch Up"
    ));
}

#[test]
fn test_graph_editor_shortcuts_and_frame_reset_on_tab_switch() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let graph_idx = find_tab_index(&app, "Graph Theory");
    app.select_tab(&ctx, graph_idx);
    assert!(app.tabs[graph_idx].select_tool_by_name("GraphEditorTool"));
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    let graph_cmds = get_registered_commands(&ctx);
    assert!(has_cmd(&graph_cmds, "Graph Editor", "Undo"));
    assert!(has_cmd(&graph_cmds, "Graph Editor", "Redo"));
    assert!(has_cmd(&graph_cmds, "Edit", "Undo"));

    // Switch to Favoritism standalone tab and verify Graph Editor commands are cleared next frame
    let fav_idx = find_tab_index(&app, "Favoritism");
    app.select_tab(&ctx, fav_idx);
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    let fav_cmds = get_registered_commands(&ctx);
    assert!(!has_cmd(&fav_cmds, "Graph Editor", "Undo"));
    assert!(has_cmd(&fav_cmds, "Global", "Keyboard Shortcuts"));
}

#[test]
fn test_multi_pane_merges_and_deduplicates_pane_shortcuts() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let graph_idx = find_tab_index(&app, "Graph Theory");
    let chaos_idx = find_tab_index(&app, "Chaos Theory");
    let analysis_idx = find_tab_index(&app, "Analysis & Calculus");
    let fav_idx = find_tab_index(&app, "Favoritism");

    assert!(app.tabs[graph_idx].select_tool_by_name("GraphEditorTool"));
    assert!(app.tabs[chaos_idx].select_tool_by_name("AttractorPlotter"));

    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, graph_idx);
    app.set_pane_tab(1, chaos_idx);
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    let merged_cmds = get_registered_commands(&ctx);
    assert!(has_cmd(&merged_cmds, "Global", "Keyboard Shortcuts"));
    assert!(has_cmd(&merged_cmds, "Graph Editor", "Undo"));
    assert!(has_cmd(&merged_cmds, "Camera 3D", "Pitch Up"));

    // Assign two populated Plot tabs across panes: identical Plot Accessibility triggers must deduplicate
    app.set_pane_tab(0, analysis_idx);
    app.set_pane_tab(1, fav_idx);
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());

    let dedup_cmds = get_registered_commands(&ctx);
    assert!(!has_cmd(&dedup_cmds, "Graph Editor", "Undo"));
    assert!(!has_cmd(&dedup_cmds, "Camera 3D", "Pitch Up"));
    let next_dataset_count = dedup_cmds
        .iter()
        .filter(|c| c.context == "Plot Accessibility" && c.name == "Next Dataset")
        .count();
    assert_eq!(next_dataset_count, 1);
}

#[test]
fn test_keyboard_shortcuts_modal_renders_and_closes_with_updated_aria() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let q_event = || {
        vec![egui::Event::Key {
            key: egui::Key::Questionmark,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }]
    };

    let open_output = run_frame_with_rect(&ctx, &mut app, &mut frame, q_event());
    assert!(app.show_help_menu);
    assert!(!open_output.shapes.is_empty());

    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, q_event());
    assert!(!app.show_help_menu);
    let close_aria = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(close_aria, Some("Keyboard shortcuts closed".to_string()));
}

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
fn test_single_and_split_pane_input_guards_and_focus_surrender() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mods = shortcut_mods();

    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    app.theme = ThemeMode::Light;
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    assert!(app.session_manager.can_undo());

    // Focus a live TextEdit in single-pane mode: Ctrl+Z must stay local and not undo theme
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
    ctx.memory_mut(|m| m.surrender_focus(te_id));

    // Focus an input in split-pane Pane 1: Ctrl+Z must not hijack workspace undo
    app.set_layout(ViewportLayout::SplitHorizontal);
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    app.theme = ThemeMode::HighContrast;
    let _ = run_frame_with_rect(&ctx, &mut app, &mut frame, Vec::new());
    let split_input_id = egui::Id::new("mock_split_pane_input");
    app.pane_contexts[1].memory_mut(|m| m.request_focus(split_input_id));
    let _ = run_frame_with_rect(
        &ctx,
        &mut app,
        &mut frame,
        key_shortcut_event(egui::Key::Z, mods),
    );
    assert_eq!(app.theme, ThemeMode::HighContrast);

    // Switching back to Single layout surrenders inactive pane focus so Ctrl+Z works immediately
    app.set_layout(ViewportLayout::Single);
    assert!(!app.pane_contexts[1].wants_keyboard_input());
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
    }
}

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

    // Single Ctrl+Z in SplitHorizontal undoes 1 node without replay or clearing global redo
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

    // Ctrl+Shift+Z redoes back to 2 nodes, then exhausting local undo falls back to session undo
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
