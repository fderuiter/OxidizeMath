#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::viewport::sash::{quad_horizontal_sash_id, quad_vertical_sash_id};
use super::*;
use crate::framework::{
    clear_pane_side_panel_records, pane_side_panel_records, PaneSidePanel, ToolMetadata,
    MAX_PANE_SIDE_PANEL_WIDTH_RATIO,
};
use eframe::egui;
use eframe::App;

fn run_frame_1024(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024.0, 768.0));
    let mut frame = eframe::Frame::_new_kittest();
    ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events,
            ..Default::default()
        },
        |c| app.update(c, &mut frame),
    )
}

fn find_button_bounds_in_x_range(
    out: &egui::FullOutput,
    needle: &str,
    x_range: std::ops::Range<f32>,
) -> Option<egui::Rect> {
    let ak = out.platform_output.accesskit_update.as_ref()?;
    for (_, node) in &ak.nodes {
        let label = node.label().unwrap_or_default();
        let val = node.value().unwrap_or_default();
        if label.contains(needle) || val.contains(needle) {
            if let Some(b) = node.bounds() {
                let rect = egui::Rect::from_min_max(
                    egui::pos2(b.x0 as f32, b.y0 as f32),
                    egui::pos2(b.x1 as f32, b.y1 as f32),
                );
                if x_range.contains(&rect.center().x) {
                    return Some(rect);
                }
            }
        }
    }
    None
}

fn collect_text_galleys(
    shapes: &[egui::epaint::Shape],
    out: &mut Vec<(String, usize, egui::Rect)>,
) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(ts) => {
                let rect = egui::Rect::from_min_size(ts.pos, ts.galley.size());
                out.push((ts.galley.text().to_string(), ts.galley.rows.len(), rect));
            }
            egui::epaint::Shape::Vec(nested) => collect_text_galleys(nested, out),
            _ => {}
        }
    }
}

#[test]
fn test_export_action_bar_1024x768_split_and_quad_no_clipping_and_wrapped_toast() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let toast = "Export status verification toast with a deliberately lengthy diagnostic summary \
        confirming wrapped feedback inside compact 512px split and quad panes without clipping.";

    for layout in [ViewportLayout::SplitHorizontal, ViewportLayout::QuadGrid] {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = MathExplorerApp::default();
        let analysis_idx = app
            .tabs
            .iter()
            .position(|t| t.framework().is_some_and(|fw| fw.domain == "analysis"))
            .unwrap();
        app.set_layout(layout);
        for p in 0..layout.pane_count() {
            app.set_pane_tab(p, analysis_idx);
            let fw = app.pane_tab_mut(p).unwrap().framework_mut().unwrap();
            fw.preset_dialog.toast_info = Some(toast.to_string());
        }

        let out = run_frame_1024(&ctx, &mut app, Vec::new());
        for (pane_x, label_ctx) in [
            (0.0..512.0_f32, "left pane"),
            (512.0..1024.0_f32, "right pane"),
        ] {
            for ctrl in [
                "Export",
                "Starter Gallery",
                "Theory",
                "Presets",
                "Reset Parameters",
            ] {
                let rect = find_button_bounds_in_x_range(&out, ctrl, pane_x.clone())
                    .unwrap_or_else(|| panic!("Missing '{ctrl}' in {label_ctx} for {layout:?}"));
                assert!(
                    rect.min.x >= pane_x.start - 1.0 && rect.max.x <= pane_x.end + 1.0,
                    "Control '{ctrl}' clipped outside {label_ctx} in {layout:?}: {rect:?}"
                );
            }
        }

        let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
        let mut galleys = Vec::new();
        collect_text_galleys(&shapes, &mut galleys);
        let wrapped = galleys
            .iter()
            .find(|(text, _, _)| text.contains("Export status verification toast"))
            .expect("Toast text rendered");
        assert!(
            wrapped.1 >= 2 && wrapped.2.width() <= 512.0,
            "Toast must wrap across >=2 rows within pane width: {wrapped:?}"
        );
    }
}

fn assert_records_valid_and_positive_canvas(ctx: &egui::Context, pane_rect: egui::Rect) {
    let records = pane_side_panel_records(ctx);
    let mut left_sum = 0.0_f32;
    let mut right_sum = 0.0_f32;
    for rec in &records {
        let cap = rec.pane_width * MAX_PANE_SIDE_PANEL_WIDTH_RATIO + 1.5;
        assert!(
            rec.outer_rect.width() <= cap,
            "PaneSidePanel {:?} width {} exceeded 35% cap {} (pane_width={})",
            rec.id,
            rec.outer_rect.width(),
            cap,
            rec.pane_width
        );
        if !rec.collapsed {
            assert!(rec.scroll_viewport_height > 0.0);
        }
        if rec.outer_rect.max.x <= pane_rect.max.x + 2.0 {
            if rec.outer_rect.center().x < pane_rect.center().x {
                left_sum += rec.outer_rect.width();
            } else {
                right_sum += rec.outer_rect.width();
            }
        }
    }
    let remaining_canvas_w = pane_rect.width() - left_sum - right_sum;
    assert!(
        remaining_canvas_w > 0.0,
        "Canvas width must remain positive at split_ratio=0.20, got {remaining_canvas_w} (pane={pane_rect:?})"
    );
}

#[test]
fn test_all_39_pane_side_panels_scroll_collapse_and_keep_positive_canvas_at_twenty_percent_split() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.split_ratio = 0.20;

    let mut distinct_panel_ids = std::collections::HashSet::new();
    let mut observed_vertical_overflow = false;

    for tab_idx in 0..app.tabs.len() {
        app.set_pane_tab(0, tab_idx);
        let tool_count = app
            .pane_tab_mut(0)
            .and_then(|t| t.framework_mut())
            .map_or(1, |fw| {
                fw.show_theory_portal = true;
                fw.available_tools.len().max(1)
            });

        for tool_idx in 0..tool_count {
            if let Some(fw) = app.pane_tab_mut(0).and_then(|t| t.framework_mut()) {
                if tool_idx < fw.available_tools.len() {
                    fw.select_tool(tool_idx);
                }
            }
            clear_pane_side_panel_records(&ctx);
            let _ = run_frame_1024(&ctx, &mut app, Vec::new());
            let records = pane_side_panel_records(&ctx);
            let pane0_rect = ctx
                .data(|d| d.get_temp::<Vec<egui::Rect>>(egui::Id::new("LAST_PANE_RECTS")))
                .and_then(|rects| rects.first().copied())
                .expect("Pane 0 rect recorded in LAST_PANE_RECTS");
            assert_records_valid_and_positive_canvas(&ctx, pane0_rect);

            for rec in &records {
                distinct_panel_ids.insert((tab_idx, tool_idx, rec.id));
                if rec.scroll_content_height > rec.scroll_viewport_height {
                    observed_vertical_overflow = true;
                }
                PaneSidePanel::set_collapsed(&ctx, rec.id, true);
            }
            if !records.is_empty() {
                clear_pane_side_panel_records(&ctx);
                let _ = run_frame_1024(&ctx, &mut app, Vec::new());
                for rec in pane_side_panel_records(&ctx) {
                    assert!(rec.collapsed);
                    assert!(rec.outer_rect.width() <= 32.0);
                    PaneSidePanel::set_collapsed(&ctx, rec.id, false);
                }
            }
        }
    }

    let registered_tools = inventory::iter::<ToolMetadata>.into_iter().count();
    assert!(registered_tools >= 30);
    assert!(
        distinct_panel_ids.len() >= 39,
        "Expected coverage across all >=39 SidePanel sites, exercised {}",
        distinct_panel_ids.len()
    );
    assert!(
        observed_vertical_overflow,
        "Expected tall sidebars to exercise ScrollArea::vertical() overflow at 1024x768"
    );
}

fn click_at_1024(ctx: &egui::Context, app: &mut MathExplorerApp, pos: egui::Pos2) {
    let _ = run_frame_1024(
        ctx,
        app,
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let _ = run_frame_1024(
        ctx,
        app,
        vec![egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
}

fn find_button_in_rect(
    out: &egui::FullOutput,
    needle: &str,
    bounds_rect: egui::Rect,
) -> Option<egui::Rect> {
    let ak = out.platform_output.accesskit_update.as_ref()?;
    for (_, node) in &ak.nodes {
        let label = node.label().unwrap_or_default();
        let val = node.value().unwrap_or_default();
        if label.contains(needle) || val.contains(needle) {
            if let Some(b) = node.bounds() {
                let rect = egui::Rect::from_min_max(
                    egui::pos2(b.x0 as f32, b.y0 as f32),
                    egui::pos2(b.x1 as f32, b.y1 as f32),
                );
                if bounds_rect.contains(rect.center()) {
                    return Some(rect);
                }
            }
        }
    }
    None
}

fn find_topmost_button(out: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    let ak = out.platform_output.accesskit_update.as_ref()?;
    ak.nodes
        .iter()
        .filter_map(|(_, node)| {
            let label = node.label().unwrap_or_default();
            let val = node.value().unwrap_or_default();
            if !(label.contains(needle) || val.contains(needle)) {
                return None;
            }
            let b = node.bounds()?;
            Some(egui::Rect::from_min_max(
                egui::pos2(b.x0 as f32, b.y0 as f32),
                egui::pos2(b.x1 as f32, b.y1 as f32),
            ))
        })
        .min_by(|a, b| a.center().y.total_cmp(&b.center().y))
}

fn verify_pane_starter_gallery_roundtrip(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    source_pane: usize,
    pane_region: egui::Rect,
) {
    let fw = app
        .pane_tab_mut(source_pane)
        .unwrap()
        .framework_mut()
        .expect("SimulationFramework-backed tab");
    fw.baseline_tool_state = Some("__before_gallery_launch__".to_string());

    let out_idle = run_frame_1024(ctx, app, Vec::new());
    let open_btn = find_button_in_rect(&out_idle, "Starter Gallery", pane_region)
        .unwrap_or_else(|| panic!("Missing Starter Gallery button in pane {source_pane}"));
    click_at_1024(ctx, app, open_btn.center());

    let _ = run_frame_1024(ctx, app, Vec::new());
    let out = run_frame_1024(ctx, app, Vec::new());

    let close_rect = find_button_bounds_in_x_range(&out, "Close Gallery", 0.0..1024.0)
        .unwrap_or_else(|| panic!("Missing Close Gallery button for pane {source_pane}"));
    assert!(
        close_rect.min.x >= 0.0
            && close_rect.max.x <= 1024.0
            && close_rect.min.y >= 0.0
            && close_rect.max.y <= 768.0,
        "Close Gallery escaped root window bounds from pane {source_pane}: {close_rect:?}"
    );

    let other_pane = (source_pane + 1) % 4;
    let other_before = app.pane_tab(other_pane).unwrap().save_state();
    let launch_rect = find_topmost_button(&out, "▶ Launch ").unwrap_or_else(|| {
        panic!("Visible Launch button inside root StarterGallery for pane {source_pane}")
    });
    assert!(
        launch_rect.min.x >= 0.0
            && launch_rect.max.x <= 1024.0
            && launch_rect.min.y >= 0.0
            && launch_rect.max.y <= 768.0,
        "Launch button escaped root window bounds from pane {source_pane}: {launch_rect:?}"
    );
    click_at_1024(ctx, app, launch_rect.center());

    assert_eq!(app.pane_tab(other_pane).unwrap().save_state(), other_before);
    let fw_after = app.pane_tab(source_pane).unwrap().framework().unwrap();
    assert!(
        fw_after.baseline_tool_state.as_deref() != Some("__before_gallery_launch__")
            && fw_after.preset_dialog.toast_error.is_none(),
        "Starter preset should load cleanly into source pane {source_pane}"
    );
    let out_closed = run_frame_1024(ctx, app, Vec::new());
    assert!(find_button_bounds_in_x_range(&out_closed, "Close Gallery", 0.0..1024.0).is_none());
}

#[test]
fn test_quad_starter_gallery_hoists_to_root_clamps_close_button_and_loads_into_source_pane() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let analysis_idx = app
        .tabs
        .iter()
        .position(|t| t.framework().is_some_and(|fw| fw.domain == "analysis"))
        .unwrap();
    let chaos_idx = app
        .tabs
        .iter()
        .position(|t| t.framework().is_some_and(|fw| fw.domain == "chaos"))
        .unwrap();
    let morph_idx = app
        .tabs
        .iter()
        .position(|t| t.name().contains("Morphogenesis"))
        .unwrap();
    let epi_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Epidemiology")
        .unwrap();

    app.set_layout(ViewportLayout::QuadGrid);
    app.quad_split_ratios = (0.30, 0.30);
    for (p, idx) in [analysis_idx, chaos_idx, morph_idx, epi_idx]
        .into_iter()
        .enumerate()
    {
        app.set_pane_tab(p, idx);
    }
    app.pane_tab_mut(2)
        .unwrap()
        .load_state(r#"{"dt":0.05,"paused":true}"#);
    let _ = run_frame_1024(&ctx, &mut app, Vec::new());

    let pane_search_rects = ctx
        .data(|d| d.get_temp::<Vec<egui::Rect>>(egui::Id::new("LAST_PANE_RECTS")))
        .expect("QuadGrid pane rects recorded in LAST_PANE_RECTS");
    assert_eq!(pane_search_rects.len(), 4);

    for (source_pane, pane_region) in pane_search_rects.into_iter().enumerate() {
        verify_pane_starter_gallery_roundtrip(&ctx, &mut app, source_pane, pane_region);
    }
}

fn assert_quad_swap_and_rotation_ops(app: &mut MathExplorerApp) {
    app.focus_pane(0);
    let baseline = app.pane_tabs.clone();
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        app.swap_pane_pair(a, b);
        app.swap_pane_pair(a, b);
        assert_eq!(app.pane_tabs, baseline);
    }
    let expected_cw = vec![baseline[2], baseline[0], baseline[3], baseline[1]];
    app.rotate_panes_clockwise();
    assert_eq!(app.pane_tabs, expected_cw);
    assert_eq!(app.focused_pane, 1);
    app.rotate_panes_counter_clockwise();
    assert_eq!(app.pane_tabs, baseline);
    assert_eq!(app.focused_pane, 0);
}

#[test]
fn test_quad_grid_2d_splitters_serde_and_four_pane_swap_and_rotate() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);
    for p in 0..4 {
        app.set_pane_tab(p, p % app.tabs.len());
    }

    let out = run_frame_1024(&ctx, &mut app, Vec::new());
    let splitter_count = out
        .platform_output
        .accesskit_update
        .as_ref()
        .unwrap()
        .nodes
        .iter()
        .filter(|(_, n)| n.role() == egui::accesskit::Role::Splitter)
        .count();
    assert_eq!(splitter_count, 2);

    ctx.memory_mut(|m| m.request_focus(quad_vertical_sash_id()));
    let _ = run_frame_1024(
        &ctx,
        &mut app,
        vec![egui::Event::Key {
            key: egui::Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert!((app.quad_split_ratios.0 - 0.55).abs() < 1e-4);

    ctx.memory_mut(|m| m.request_focus(quad_horizontal_sash_id()));
    let _ = run_frame_1024(
        &ctx,
        &mut app,
        vec![egui::Event::Key {
            key: egui::Key::ArrowDown,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::SHIFT,
        }],
    );
    assert!((app.quad_split_ratios.1 - 0.51).abs() < 1e-4);

    let saved = app.save_state_to_json();
    let mut restored = MathExplorerApp::default();
    restored.load_state_from_json(&saved);
    assert!((restored.quad_split_ratios.0 - 0.55).abs() < 1e-4);
    assert!((restored.quad_split_ratios.1 - 0.51).abs() < 1e-4);

    let mut legacy_val = serde_json::to_value(app.get_app_state()).unwrap();
    legacy_val
        .as_object_mut()
        .unwrap()
        .remove("quad_split_ratios");
    let legacy: AppState = serde_json::from_value(legacy_val).unwrap();
    assert_eq!(legacy.quad_split_ratios, (0.5, 0.5));

    assert_quad_swap_and_rotation_ops(&mut app);
}
