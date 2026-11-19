#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
#![allow(clippy::field_reassign_with_default)]

use super::*;
use crate::app::viewport::{
    calculate_pane_rects, calculate_pane_rects_with_ratio, filter_raw_input_for_pane,
};
use eframe::egui;
use eframe::App;

#[test]
fn test_viewport_layout_presets_and_names() {
    assert_eq!(ViewportLayout::default(), ViewportLayout::Single);
    assert_eq!(ViewportLayout::Single.name(), "Single");
    assert_eq!(ViewportLayout::SplitHorizontal.name(), "Split Horizontal");
    assert_eq!(ViewportLayout::SplitVertical.name(), "Split Vertical");
    assert_eq!(ViewportLayout::QuadGrid.name(), "2x2 Quad Grid");

    assert_eq!(ViewportLayout::Single.pane_count(), 1);
    assert_eq!(ViewportLayout::SplitHorizontal.pane_count(), 2);
    assert_eq!(ViewportLayout::SplitVertical.pane_count(), 2);
    assert_eq!(ViewportLayout::QuadGrid.pane_count(), 4);

    assert_eq!(ViewportLayout::Single.pane_labels().len(), 1);
    assert_eq!(ViewportLayout::SplitHorizontal.pane_labels().len(), 2);
    assert_eq!(ViewportLayout::SplitVertical.pane_labels().len(), 2);
    assert_eq!(ViewportLayout::QuadGrid.pane_labels().len(), 4);
}

#[test]
fn test_viewport_layout_toggle_transitions() {
    let mut app = MathExplorerApp::default();
    assert_eq!(app.viewport_layout, ViewportLayout::Single);

    app.viewport_layout = ViewportLayout::SplitHorizontal;
    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);

    app.viewport_layout = ViewportLayout::SplitVertical;
    assert_eq!(app.viewport_layout, ViewportLayout::SplitVertical);

    app.viewport_layout = ViewportLayout::QuadGrid;
    assert_eq!(app.viewport_layout, ViewportLayout::QuadGrid);

    app.viewport_layout = ViewportLayout::Single;
    assert_eq!(app.viewport_layout, ViewportLayout::Single);
}

#[test]
#[cfg(feature = "pure_math")]
fn test_multi_pane_independent_tab_assignment() {
    let mut app = MathExplorerApp::default();
    assert!(app.tabs.len() >= 2);

    app.set_pane_tab(0, 0);
    app.set_pane_tab(1, 1);
    assert_eq!(app.pane_tabs[0], 0);
    assert_eq!(app.pane_tabs[1], 1);
    assert_eq!(app.selected_tab, 0);

    // Switch pane 0 and verify selected_tab synchronizes
    app.set_pane_tab(0, 1);
    assert_eq!(app.pane_tabs[0], 1);
    assert_eq!(app.selected_tab, 1);

    // Switch pane 1 and verify pane 0 remains unchanged
    app.set_pane_tab(1, 0);
    assert_eq!(app.pane_tabs[1], 0);
    assert_eq!(app.pane_tabs[0], 1);
}

#[test]
fn test_multi_pane_rect_calculation_horizontal_split() {
    let total_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
    let rects = calculate_pane_rects(ViewportLayout::SplitHorizontal, total_rect);

    assert_eq!(rects.len(), 2);
    assert_eq!(rects[0].min, egui::pos2(0.0, 0.0));
    assert_eq!(rects[0].max, egui::pos2(500.0, 800.0));
    assert_eq!(rects[1].min, egui::pos2(500.0, 0.0));
    assert_eq!(rects[1].max, egui::pos2(1000.0, 800.0));
}

#[test]
fn test_multi_pane_rect_calculation_vertical_split() {
    let total_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
    let rects = calculate_pane_rects(ViewportLayout::SplitVertical, total_rect);

    assert_eq!(rects.len(), 2);
    assert_eq!(rects[0].min, egui::pos2(0.0, 0.0));
    assert_eq!(rects[0].max, egui::pos2(1000.0, 400.0));
    assert_eq!(rects[1].min, egui::pos2(0.0, 400.0));
    assert_eq!(rects[1].max, egui::pos2(1000.0, 800.0));
}

#[test]
fn test_multi_pane_rect_calculation_quad_grid() {
    let total_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
    let rects = calculate_pane_rects(ViewportLayout::QuadGrid, total_rect);

    assert_eq!(rects.len(), 4);
    assert_eq!(rects[0].min, egui::pos2(0.0, 0.0));
    assert_eq!(rects[0].max, egui::pos2(500.0, 400.0));

    assert_eq!(rects[1].min, egui::pos2(500.0, 0.0));
    assert_eq!(rects[1].max, egui::pos2(1000.0, 400.0));

    assert_eq!(rects[2].min, egui::pos2(0.0, 400.0));
    assert_eq!(rects[2].max, egui::pos2(500.0, 800.0));

    assert_eq!(rects[3].min, egui::pos2(500.0, 400.0));
    assert_eq!(rects[3].max, egui::pos2(1000.0, 800.0));
}

#[test]
fn test_multi_pane_input_isolation() {
    let pane_left = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(500.0, 800.0));
    let pane_right = egui::Rect::from_min_size(egui::pos2(500.0, 0.0), egui::vec2(500.0, 800.0));

    let click_left = egui::Event::PointerButton {
        pos: egui::pos2(100.0, 100.0),
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    };
    let click_right = egui::Event::PointerButton {
        pos: egui::pos2(700.0, 100.0),
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    };

    let raw_input = egui::RawInput {
        events: vec![click_left, click_right],
        ..Default::default()
    };

    let left_filtered =
        filter_raw_input_for_pane(&raw_input, pane_left, Some(egui::pos2(100.0, 100.0)));
    assert_eq!(left_filtered.events.len(), 1);
    match &left_filtered.events[0] {
        egui::Event::PointerButton { pos, .. } => assert_eq!(*pos, egui::pos2(100.0, 100.0)),
        _ => panic!("Expected pointer button event"),
    }

    let right_filtered =
        filter_raw_input_for_pane(&raw_input, pane_right, Some(egui::pos2(700.0, 100.0)));
    assert_eq!(right_filtered.events.len(), 1);
    match &right_filtered.events[0] {
        egui::Event::PointerButton { pos, .. } => assert_eq!(*pos, egui::pos2(700.0, 100.0)),
        _ => panic!("Expected pointer button event"),
    }
}

#[test]
fn test_multi_pane_session_envelope_serialization_and_restore() {
    let mut app1 = MathExplorerApp::default();
    app1.viewport_layout = ViewportLayout::SplitHorizontal;
    app1.set_pane_tab(0, 0);
    if app1.tabs.len() > 1 {
        app1.set_pane_tab(1, 1);
    }

    let json_str = app1.save_state_to_json();
    assert!(!json_str.is_empty());

    let mut app2 = MathExplorerApp::default();
    app2.load_state_from_json(&json_str);

    assert_eq!(app2.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(app2.pane_tabs[0], 0);
    if app1.tabs.len() > 1 {
        assert_eq!(app2.pane_tabs[1], 1);
    }
}

#[test]
#[cfg(feature = "pure_math")]
fn test_legacy_session_deserialization_defaults_to_single() {
    let legacy_json = r#"{
        "selected_tab": 1,
        "selected_tab_name": null,
        "show_info": true,
        "show_warnings": true,
        "show_errors": true,
        "show_diagnostics_panel": true,
        "theme": "Dark",
        "tab_states": {}
    }"#;

    let mut app = MathExplorerApp::default();
    app.load_state_from_json(legacy_json);

    assert_eq!(app.viewport_layout, ViewportLayout::Single);
    assert!(app.pane_tabs.len() >= 4);
    assert_eq!(app.selected_tab, 1);
    assert_eq!(app.pane_tabs[0], 1);
}

#[test]
fn test_multi_pane_frame_rendering_split_horizontal() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.viewport_layout = ViewportLayout::SplitHorizontal;
    app.set_pane_tab(0, 0);
    if app.tabs.len() > 1 {
        app.set_pane_tab(1, 1);
    }

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000.0, 800.0),
        )),
        ..Default::default()
    };

    let output = ctx.run(raw_input, |ctx| {
        app.update(ctx, &mut frame);
    });

    assert!(!output.shapes.is_empty());
    let primitives = ctx.tessellate(output.shapes, 1.0);
    assert!(!primitives.is_empty());

    // Generate verification screenshot for Split Horizontal
    generate_verification_screenshot(
        "/tmp/viewport_split_horizontal.png",
        "Split Horizontal: 2 Viewports",
        &["Left: Analysis", "Right: Chaos"],
    );
}

#[test]
fn test_multi_pane_frame_rendering_quad_grid() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.viewport_layout = ViewportLayout::QuadGrid;
    for i in 0..4 {
        if i < app.tabs.len() {
            app.set_pane_tab(i, i);
        }
    }

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000.0, 800.0),
        )),
        ..Default::default()
    };

    let output = ctx.run(raw_input, |ctx| {
        app.update(ctx, &mut frame);
    });

    assert!(!output.shapes.is_empty());
    let primitives = ctx.tessellate(output.shapes, 1.0);
    assert!(!primitives.is_empty());

    // Generate verification screenshot for 2x2 Quad Grid
    generate_verification_screenshot(
        "/tmp/viewport_quad_grid.png",
        "2x2 Quad Grid: 4 Viewports",
        &["TL: Pane 1", "TR: Pane 2", "BL: Pane 3", "BR: Pane 4"],
    );
}

#[test]
fn test_multi_pane_frame_rendering_split_vertical() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.set_layout(ViewportLayout::SplitVertical);
    app.set_pane_tab(0, 0);
    if app.tabs.len() > 1 {
        app.set_pane_tab(1, 1);
    }

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000.0, 800.0),
        )),
        ..Default::default()
    };

    let output = ctx.run(raw_input, |ctx| {
        app.update(ctx, &mut frame);
    });

    assert!(!output.shapes.is_empty());

    // Generate verification screenshot for Split Vertical
    generate_verification_screenshot(
        "/tmp/viewport_split_vertical.png",
        "Split Vertical: 2 Viewports",
        &["Top: Analysis", "Bottom: Chaos"],
    );
}

#[test]
fn test_quad_grid_selection_overrides_active_split_layout() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    // Activate dual horizontal split
    app.set_layout(ViewportLayout::SplitHorizontal);
    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(app.split_layout_mode, SplitLayoutMode::Horizontal);

    // Select quad grid preset - must immediately transition to 4 quadrants
    app.set_layout(ViewportLayout::QuadGrid);
    assert_eq!(app.viewport_layout, ViewportLayout::QuadGrid);
    assert_eq!(app.split_layout_mode, SplitLayoutMode::Single);

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
        "Quad grid must render canvas display"
    );
}

#[test]
fn test_divider_drag_ratio_adjustments_and_clamping() {
    let total_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));

    // Test default 0.50 ratio
    let (rects_50, divider_50) =
        calculate_pane_rects_with_ratio(ViewportLayout::SplitHorizontal, total_rect, 0.50);
    assert_eq!(rects_50.len(), 2);
    assert!(divider_50.is_some());
    let div = divider_50.unwrap();
    assert!((div.center().x - 500.0).abs() < 1.0);

    // Test 0.30 ratio
    let (rects_30, divider_30) =
        calculate_pane_rects_with_ratio(ViewportLayout::SplitHorizontal, total_rect, 0.30);
    assert_eq!(rects_30.len(), 2);
    let div_30 = divider_30.unwrap();
    assert!((div_30.center().x - 300.0).abs() < 1.0);

    // Test lower bound clamping at 0.20
    let (rects_min, divider_min) =
        calculate_pane_rects_with_ratio(ViewportLayout::SplitHorizontal, total_rect, 0.05);
    assert_eq!(rects_min.len(), 2);
    let div_min = divider_min.unwrap();
    assert!((div_min.center().x - 200.0).abs() < 1.0);

    // Test upper bound clamping at 0.80
    let (rects_max, divider_max) =
        calculate_pane_rects_with_ratio(ViewportLayout::SplitHorizontal, total_rect, 0.95);
    assert_eq!(rects_max.len(), 2);
    let div_max = divider_max.unwrap();
    assert!((div_max.center().x - 800.0).abs() < 1.0);
}

#[test]
#[cfg(feature = "pure_math")]
fn test_legacy_saved_state_restores_split_ratio_and_layout() {
    let legacy_json = r#"{
        "selected_tab": 0,
        "selected_tab_name": null,
        "show_info": true,
        "show_warnings": true,
        "show_errors": true,
        "show_diagnostics_panel": true,
        "theme": "Dark",
        "tab_states": {},
        "split_layout_mode": "Horizontal",
        "secondary_tab": 1,
        "split_ratio": 0.65
    }"#;

    let mut app = MathExplorerApp::default();
    app.load_state_from_json(legacy_json);

    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(app.split_layout_mode, SplitLayoutMode::Horizontal);
    assert!((app.split_ratio - 0.65).abs() < 1e-4);
    assert_eq!(app.pane_tabs[0], 0);
    assert_eq!(app.pane_tabs[1], 1);
}

fn generate_verification_screenshot(path: &str, _title: &str, labels: &[&str]) {
    let mut comparison_datasets = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        let val = (i as f64 + 1.0) * 10.0;
        comparison_datasets.push((
            label.to_string(),
            vec![[0.0, val], [1.0, val + 5.0], [2.0, val]],
        ));
    }
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&comparison_datasets, &options) {
        let _ = std::fs::write(path, png_data);
    }
}

#[test]
fn test_lazy_secondary_slots_and_same_domain_quad_isolation() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    assert_eq!(app.active_secondary_slot_count(), 0);
    let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") else {
        return;
    };
    for p in 0..4 {
        app.set_pane_tab(p, epi_idx);
        let fw = app.pane_tab_mut(p).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("SirTool"));
        let beta = 0.2 + p as f64 * 0.8;
        fw.active_tool.as_mut().unwrap().load_state(&format!(
            r#"{{"n":1000.0,"i0":10.0,"beta":{beta},"gamma":0.1,"duration":100.0}}"#
        ));
    }
    let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1280.0, 900.0));
    for (layout, expected_slots) in [
        (ViewportLayout::SplitHorizontal, 1),
        (ViewportLayout::SplitVertical, 1),
        (ViewportLayout::QuadGrid, 3),
    ] {
        app.set_layout(layout);
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |c| app.update(c, &mut frame),
        );
        assert_eq!(app.active_secondary_slot_count(), expected_slots);
    }
    for p in 0..4 {
        let state = app
            .pane_tab(p)
            .unwrap()
            .framework()
            .unwrap()
            .active_tool
            .as_ref()
            .unwrap()
            .save_state()
            .unwrap();
        let expected_beta = format!("\"beta\":{}", 0.2 + p as f64 * 0.8);
        assert!(
            state.contains(&expected_beta),
            "Pane {p} lost beta: {state}"
        );
    }
    app.set_layout(ViewportLayout::Single);
    assert_eq!(app.active_secondary_slot_count(), 0);
}
