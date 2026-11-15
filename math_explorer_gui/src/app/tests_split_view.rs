#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use eframe::App;

#[test]
fn test_split_layout_mode_defaults_and_labels() {
    assert_eq!(SplitLayoutMode::default(), SplitLayoutMode::Single);
    assert_eq!(SplitLayoutMode::Single.label(), "Single View");
    assert_eq!(SplitLayoutMode::Horizontal.label(), "Horizontal Split");
    assert_eq!(SplitLayoutMode::Vertical.label(), "Vertical Split");

    let json = serde_json::to_string(&SplitLayoutMode::Horizontal).unwrap();
    let deserialized: SplitLayoutMode = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, SplitLayoutMode::Horizontal);
}

#[test]
fn test_backward_compatibility_deserialization() {
    // Old workspace session envelope without split view fields
    let old_json = r#"{
        "schema_version": "1.0",
        "app_version": "0.1.0",
        "timestamp": "2026-01-01T00:00:00Z",
        "app_state": {
            "selected_tab": 0,
            "selected_tab_name": "Artificial Intelligence",
            "show_info": true,
            "show_warnings": true,
            "show_errors": true,
            "show_diagnostics_panel": true,
            "theme": "Dark",
            "tab_states": {}
        }
    }"#;

    let mut app = MathExplorerApp::default();
    app.load_state_from_json(old_json);

    assert_eq!(app.split_layout_mode, SplitLayoutMode::Single);
    assert_eq!(app.secondary_tab, 1);
    assert!((app.split_ratio - 0.5).abs() < 1e-4);
}

#[test]
fn test_swap_panes() {
    let mut app = MathExplorerApp::default();
    if app.tabs.len() > 1 {
        app.selected_tab = 0;
        app.secondary_tab = 1;

        app.swap_panes();
        assert_eq!(app.selected_tab, 1);
        assert_eq!(app.secondary_tab, 0);

        app.swap_panes();
        assert_eq!(app.selected_tab, 0);
        assert_eq!(app.secondary_tab, 1);
    }
}

#[test]
fn test_split_ratio_clamping() {
    let mut app = MathExplorerApp::default();

    // Test extreme lower bound clamp
    let state_low = AppState {
        selected_tab: 0,
        selected_tab_name: None,
        show_info: true,
        show_warnings: true,
        show_errors: true,
        show_diagnostics_panel: true,
        theme: ThemeMode::Dark,
        tab_states: std::collections::HashMap::new(),
        split_layout_mode: SplitLayoutMode::Horizontal,
        secondary_tab: 1,
        secondary_tab_name: None,
        split_ratio: 0.05,
        ..Default::default()
    };
    app.apply_app_state(state_low);
    assert!((app.split_ratio - 0.20).abs() < 1e-4);

    // Test extreme upper bound clamp
    let state_high = AppState {
        selected_tab: 0,
        selected_tab_name: None,
        show_info: true,
        show_warnings: true,
        show_errors: true,
        show_diagnostics_panel: true,
        theme: ThemeMode::Dark,
        tab_states: std::collections::HashMap::new(),
        split_layout_mode: SplitLayoutMode::Vertical,
        secondary_tab: 1,
        secondary_tab_name: None,
        split_ratio: 0.95,
        ..Default::default()
    };
    app.apply_app_state(state_high);
    assert!((app.split_ratio - 0.80).abs() < 1e-4);
}

#[test]
fn test_split_view_session_save_and_restore() {
    let mut app1 = MathExplorerApp::default();
    if app1.tabs.len() > 2 {
        app1.selected_tab = 1;
        app1.secondary_tab = 2;
        app1.split_layout_mode = SplitLayoutMode::Horizontal;
        app1.split_ratio = 0.65;

        let json = app1.save_state_to_json();

        let mut app2 = MathExplorerApp::default();
        app2.load_state_from_json(&json);

        assert_eq!(app2.split_layout_mode, SplitLayoutMode::Horizontal);
        assert_eq!(app2.selected_tab, 1);
        assert_eq!(app2.secondary_tab, 2);
        assert!((app2.split_ratio - 0.65).abs() < 1e-4);
    }
}

#[test]
fn test_split_view_undo_redo_persistence() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    if app.tabs.len() > 2 {
        let initial_state = app.get_app_state();
        app.session_manager.track_state_changes(&ctx, initial_state);

        // Change layout mode and split ratio
        app.split_layout_mode = SplitLayoutMode::Vertical;
        app.split_ratio = 0.35;
        let modified_state = app.get_app_state();
        app.session_manager
            .track_state_changes(&ctx, modified_state.clone());

        // Undo layout change
        assert!(app.session_manager.can_undo());
        app.undo();
        assert_eq!(app.split_layout_mode, SplitLayoutMode::Single);
        assert!((app.split_ratio - 0.50).abs() < 1e-4);

        // Redo layout change
        assert!(app.session_manager.can_redo());
        app.redo();
        assert_eq!(app.split_layout_mode, SplitLayoutMode::Vertical);
        assert!((app.split_ratio - 0.35).abs() < 1e-4);
    }
}

#[test]
fn test_dual_pane_rendering_horizontal() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    if app.tabs.len() > 1 {
        app.split_layout_mode = SplitLayoutMode::Horizontal;
        app.selected_tab = 0;
        app.secondary_tab = 1;
        app.split_ratio = 0.5;

        let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0));
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| {
                app.update(ctx, &mut frame);
            },
        );

        assert!(
            !output.shapes.is_empty(),
            "Dual pane layout must produce render shapes"
        );
    }
}

#[test]
fn test_dual_pane_rendering_vertical() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    if app.tabs.len() > 1 {
        app.split_layout_mode = SplitLayoutMode::Vertical;
        app.selected_tab = 0;
        app.secondary_tab = 1;
        app.split_ratio = 0.4;

        let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0));
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| {
                app.update(ctx, &mut frame);
            },
        );

        assert!(
            !output.shapes.is_empty(),
            "Vertical dual pane layout must produce render shapes"
        );
    }
}

#[test]
fn test_generate_split_view_screenshots() {
    let options = egui_plot::export::ExportOptions::default();
    let datasets_h = vec![
        (
            "Pane 1: Harmonic Oscillator (Primary)".to_string(),
            (0..100)
                .map(|i| {
                    let t = i as f64 * 0.1;
                    [t, t.sin()]
                })
                .collect(),
        ),
        (
            "Pane 2: Exponential Growth (Secondary)".to_string(),
            (0..100)
                .map(|i| {
                    let t = i as f64 * 0.1;
                    [t, (0.5 * t).exp().min(20.0)]
                })
                .collect(),
        ),
    ];
    if let Ok(png) = egui_plot::export::export_custom_png(&datasets_h, &options) {
        let _ = std::fs::write("/tmp/dual_pane_split_horizontal.png", png);
    }

    let datasets_v = vec![
        (
            "Pane 1 (Top): Lotka-Volterra Prey".to_string(),
            (0..100)
                .map(|i| {
                    let t = i as f64 * 0.1;
                    [t, 2.0 + 1.5 * t.cos()]
                })
                .collect(),
        ),
        (
            "Pane 2 (Bottom): Lotka-Volterra Predator".to_string(),
            (0..100)
                .map(|i| {
                    let t = i as f64 * 0.1;
                    [t, 1.0 + 0.8 * (t - 0.5).sin()]
                })
                .collect(),
        ),
    ];
    if let Ok(png) = egui_plot::export::export_custom_png(&datasets_v, &options) {
        let _ = std::fs::write("/tmp/dual_pane_split_vertical.png", png);
    }
}
