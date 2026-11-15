#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::App;

#[test]
fn test_legacy_app_state_deserialization_defaults_to_true() {
    let legacy_json = r#"{
        "selected_tab": 0,
        "selected_tab_name": null,
        "show_info": true,
        "show_warnings": true,
        "show_errors": true,
        "theme": "Dark",
        "tab_states": {}
    }"#;

    let app_state: AppState =
        serde_json::from_str(legacy_json).expect("Legacy AppState JSON must deserialize");
    assert!(app_state.show_diagnostics_panel);
}

#[test]
fn test_legacy_workspace_envelope_deserialization() {
    let legacy_envelope_json = r#"{
        "schema_version": "1.0",
        "app_version": "0.1.0",
        "timestamp": "2026-11-14T00:00:00Z",
        "app_state": {
            "selected_tab": 0,
            "selected_tab_name": null,
            "show_info": true,
            "show_warnings": true,
            "show_errors": true,
            "theme": "Dark",
            "tab_states": {}
        },
        "history_undo_stack": [],
        "history_redo_stack": []
    }"#;

    let mut mgr = WorkspaceSessionManager::default();
    let state = mgr
        .parse_envelope(legacy_envelope_json)
        .expect("Legacy envelope must parse successfully");
    assert!(state.show_diagnostics_panel);
}

#[test]
fn test_explicit_false_diagnostics_panel_deserialization() {
    let json_false = r#"{
        "selected_tab": 0,
        "selected_tab_name": null,
        "show_info": true,
        "show_warnings": true,
        "show_errors": true,
        "show_diagnostics_panel": false,
        "theme": "Dark",
        "tab_states": {}
    }"#;

    let app_state: AppState =
        serde_json::from_str(json_false).expect("AppState with false should deserialize");
    assert!(!app_state.show_diagnostics_panel);
}

#[test]
fn test_diagnostics_panel_shortcut_toggle() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    assert!(app.show_diagnostics_panel);

    // Initial render
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });

    let modifiers = if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD | egui::Modifiers::SHIFT
    } else {
        egui::Modifiers::CTRL | egui::Modifiers::SHIFT
    };

    // Trigger toggle shortcut
    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::D,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }],
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        app.update(ctx, &mut frame);
    });

    assert!(!app.show_diagnostics_panel);

    // Trigger toggle shortcut again to restore
    let raw_input_again = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::D,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }],
        ..Default::default()
    };
    let _ = ctx.run(raw_input_again, |ctx| {
        app.update(ctx, &mut frame);
    });

    assert!(app.show_diagnostics_panel);
}

fn find_text_pos(shapes: &[egui::epaint::ClippedShape], target: &str) -> Option<egui::Pos2> {
    for shape in shapes {
        if let egui::epaint::Shape::Text(text_shape) = &shape.shape {
            if text_shape.galley.text() == target {
                return Some(text_shape.pos);
            }
        }
    }
    None
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_diagnostics_panel_hide_action() {
    let ctx = egui::Context::default();
    let mut events = Vec::new();
    let mut show_errors = true;
    let mut show_warnings = true;
    let mut show_info = true;
    let mut show_panel = true;

    // Frame 1: render issues panel to establish widget layout in egui
    let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            time: Some(0.0),
            ..Default::default()
        },
        |ctx| {
            ui_panels::render_issues_panel(
                ctx,
                &mut events,
                &mut show_errors,
                &mut show_warnings,
                &mut show_info,
                &mut show_panel,
            );
        },
    );

    let text_pos = find_text_pos(&out.shapes, "Hide").expect("Hide button text must exist");
    // Text is inside the button; offset slightly to be safely inside the button's click box
    let hide_pos = egui::pos2(text_pos.x + 8.0, text_pos.y + 6.0);
    assert!(show_panel);

    // Frame 2: mouse down on Hide button
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            time: Some(0.1),
            events: vec![
                egui::Event::PointerMoved(hide_pos),
                egui::Event::PointerButton {
                    pos: hide_pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| {
            ui_panels::render_issues_panel(
                ctx,
                &mut events,
                &mut show_errors,
                &mut show_warnings,
                &mut show_info,
                &mut show_panel,
            );
        },
    );

    // Frame 3: mouse release on Hide button (triggers click)
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            time: Some(0.2),
            events: vec![
                egui::Event::PointerMoved(hide_pos),
                egui::Event::PointerButton {
                    pos: hide_pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| {
            ui_panels::render_issues_panel(
                ctx,
                &mut events,
                &mut show_errors,
                &mut show_warnings,
                &mut show_info,
                &mut show_panel,
            );
        },
    );

    assert!(
        !show_panel,
        "Clicking Hide button must collapse the diagnostics panel"
    );
}

#[test]
fn test_telemetry_persisted_even_when_diagnostics_hidden() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.show_diagnostics_panel = false;

    global_registry().emit(TelemetryEvent {
        source: "DiagnosticsTest".to_string(),
        severity: Severity::Error,
        message: "Test failure notice".to_string(),
        metadata: std::collections::HashMap::new(),
        thread_name: None,
    });

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });

    assert!(
        app.diagnostic_events
            .iter()
            .any(|e| e.message == "Test failure notice"),
        "Telemetry event must be recorded even when panel is hidden"
    );
}

#[test]
fn test_viewport_space_reclamation_when_diagnostics_hidden() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 800.0));

    // When panel is shown
    app.show_diagnostics_panel = true;
    let mut central_height_shown = 0.0;
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| {
            app.update(ctx, &mut frame);
            central_height_shown = ctx.available_rect().height();
        },
    );

    // When panel is hidden
    app.show_diagnostics_panel = false;
    let mut central_height_hidden = 0.0;
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| {
            app.update(ctx, &mut frame);
            central_height_hidden = ctx.available_rect().height();
        },
    );

    assert!(
        central_height_hidden >= central_height_shown + 100.0,
        "Hiding diagnostics panel should recover at least 100px of vertical space (shown: {central_height_shown}, hidden: {central_height_hidden})"
    );

    let comparison_datasets = vec![
        (
            "Viewport Height (Panel Visible)".to_string(),
            vec![
                [0.0, central_height_shown as f64],
                [1.0, central_height_shown as f64],
            ],
        ),
        (
            "Viewport Height (Panel Hidden)".to_string(),
            vec![
                [0.0, central_height_hidden as f64],
                [1.0, central_height_hidden as f64],
            ],
        ),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&comparison_datasets, &options) {
        let _ = std::fs::write("/tmp/diagnostics_viewport_reclamation.png", png_data);
    }
}

#[test]
fn test_view_menu_toggle_diagnostics_panel() {
    let mut app = MathExplorerApp::default();
    assert!(app.show_diagnostics_panel);

    app.show_diagnostics_panel = false;
    assert!(!app.show_diagnostics_panel);

    let state = app.get_app_state();
    assert!(!state.show_diagnostics_panel);

    app.apply_app_state(AppState {
        selected_tab: 0,
        selected_tab_name: None,
        show_info: true,
        show_warnings: true,
        show_errors: true,
        show_diagnostics_panel: true,
        theme: ThemeMode::Dark,
        tab_states: std::collections::HashMap::new(),
        ..Default::default()
    });
    assert!(app.show_diagnostics_panel);
}
