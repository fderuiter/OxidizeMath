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

fn run_split_frames(ctx: &egui::Context, app: &mut MathExplorerApp, frames: usize) {
    let mut frame = eframe::Frame::_new_kittest();
    let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1280.0, 800.0));
    for _ in 0..frames {
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |c| app.update(c, &mut frame),
        );
    }
}

#[test]
#[cfg(feature = "epidemiology")]
fn test_same_domain_network_propagation_steps_once_per_pane_per_frame() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") else {
        return;
    };
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, epi_idx);
    app.set_pane_tab(1, epi_idx);
    for (pane, beta) in [(0, 0.85), (1, 0.05)] {
        let fw = app.pane_tab_mut(pane).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("NetworkPropagationTool"));
        fw.active_tool.as_mut().unwrap().load_state(&format!(
            r#"{{"num_nodes":60,"beta":{beta},"gamma":0.05,"is_running":true}}"#
        ));
    }
    run_split_frames(&ctx, &mut app, 5);
    for (pane, expected_beta) in [(0, 0.85), (1, 0.05)] {
        let fw = app.pane_tab(pane).unwrap().framework().unwrap();
        assert_eq!(
            fw.step_counter, 5,
            "Pane {pane} framework must step once per frame"
        );
        let tool = fw.active_tool.as_ref().unwrap();
        let net_tool = tool
            .as_any()
            .and_then(|a| {
                a.downcast_ref::<crate::tabs::epidemiology::network_propagation::NetworkPropagationTool>()
            })
            .expect("Active tool must be NetworkPropagationTool");
        assert_eq!(
            net_tool.sim_steps(),
            5,
            "Pane {pane} model must step once per frame"
        );
        let saved: serde_json::Value = serde_json::from_str(&tool.save_state().unwrap()).unwrap();
        assert!((saved["beta"].as_f64().unwrap() - expected_beta).abs() < 1e-6);
    }
}

#[test]
fn test_same_domain_scoped_overlays_and_export_modal_isolation() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") else {
        return;
    };
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, epi_idx);
    app.set_pane_tab(1, epi_idx);
    run_split_frames(&ctx, &mut app, 1);
    {
        let fw0 = app.pane_tab_mut(0).unwrap().framework_mut().unwrap();
        fw0.show_theory_portal = true;
        fw0.show_starter_gallery = true;
        fw0.show_export_modal = true;
    }
    run_split_frames(&ctx, &mut app, 3);
    let fw0 = app.pane_tab(0).unwrap().framework().unwrap();
    let fw1 = app.pane_tab(1).unwrap().framework().unwrap();
    assert!(fw0.show_theory_portal && fw0.show_starter_gallery && fw0.show_export_modal);
    assert!(!fw1.show_theory_portal && !fw1.show_starter_gallery && !fw1.show_export_modal);
}

#[test]
fn test_same_domain_per_pane_sir_dataset_exports() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") else {
        return;
    };
    app.set_layout(ViewportLayout::SplitHorizontal);
    for (pane, beta) in [(0, 0.2), (1, 2.5)] {
        app.set_pane_tab(pane, epi_idx);
        let fw = app.pane_tab_mut(pane).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("SirTool"));
        fw.active_tool.as_mut().unwrap().load_state(&format!(
            r#"{{"n":1000.0,"i0":10.0,"beta":{beta},"gamma":0.1,"duration":100.0}}"#
        ));
    }
    run_split_frames(&ctx, &mut app, 2);
    let d0 = app
        .pane_tab(0)
        .unwrap()
        .framework()
        .unwrap()
        .export_datasets_with_fallback();
    let d1 = app
        .pane_tab(1)
        .unwrap()
        .framework()
        .unwrap()
        .export_datasets_with_fallback();
    let peak_i = |ds: &[(String, Vec<[f64; 2]>)]| {
        ds.iter()
            .find(|(n, _)| n.contains("Infected"))
            .unwrap()
            .1
            .iter()
            .map(|p| p[1])
            .fold(0.0_f64, f64::max)
    };
    assert!(peak_i(&d1) > peak_i(&d0) + 400.0);
    let combined = vec![
        (
            "Pane 0 SIR Infected (beta=0.2)".to_string(),
            d0[1].1.clone(),
        ),
        (
            "Pane 1 SIR Infected (beta=2.5)".to_string(),
            d1[1].1.clone(),
        ),
    ];
    if let Ok(png) = egui_plot::export::export_custom_png(&combined, &Default::default()) {
        let _ = std::fs::write("/tmp/same_domain_split_sir_comparison.png", png);
    }
}

#[test]
fn test_same_domain_per_pane_fallback_plot_exports() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(ana_idx) = app
        .tabs
        .iter()
        .position(|t| t.name() == "Analysis & Calculus")
    else {
        return;
    };
    app.set_layout(ViewportLayout::SplitHorizontal);
    for (pane, max_x, steps) in [(0, 1.5, 4), (1, 3.0, 24)] {
        app.set_pane_tab(pane, ana_idx);
        let fw = app.pane_tab_mut(pane).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("RiemannIntegrationTool"));
        fw.active_tool.as_mut().unwrap().load_state(&format!(
            r#"{{"preset":"XSquared","min_x":0.0,"max_x":{max_x},"steps":{steps}}}"#
        ));
    }
    run_split_frames(&ctx, &mut app, 2);
    egui_plot::clear_last_plot_datasets();
    let r0 = app
        .pane_tab(0)
        .unwrap()
        .framework()
        .unwrap()
        .export_datasets_with_fallback();
    let r1 = app
        .pane_tab(1)
        .unwrap()
        .framework()
        .unwrap()
        .export_datasets_with_fallback();
    assert!(!r0.is_empty() && !r1.is_empty());
    assert_ne!(
        r0[0].1.last(),
        r1[0].1.last(),
        "Fallback plot datasets must stay distinct per pane"
    );
}
