#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
#![allow(clippy::field_reassign_with_default)]

use super::*;
use eframe::egui;
use eframe::App;
use egui_plot::commands::{CommandMetadata, CommandRegistryData};

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

#[cfg(feature = "pure_math")]
fn find_tab_index(app: &MathExplorerApp, tab_name: &str) -> usize {
    app.tabs
        .iter()
        .position(|t| t.name() == tab_name)
        .unwrap_or_else(|| panic!("Expected tab '{}' to be registered", tab_name))
}

fn has_cmd(cmds: &[CommandMetadata], context: &str, name: &str) -> bool {
    cmds.iter().any(|c| c.context == context && c.name == name)
}

#[cfg(feature = "pure_math")]
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
            && c.trigger == egui_plot::commands::CommandTrigger::Key(egui::Key::Questionmark)
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

#[cfg(feature = "applied")]
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

#[cfg(feature = "physics")]
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

#[cfg(feature = "applied")]
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

#[cfg(feature = "physics")]
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

mod history;
mod sweep_and_undo;
