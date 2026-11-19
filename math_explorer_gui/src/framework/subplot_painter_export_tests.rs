#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use std::collections::HashSet;

fn open_context_menu_and_find_save_png(
    ctx: &egui::Context,
    morph: &mut crate::tabs::morphogenesis::MorphogenesisTab,
    screen: egui::Rect,
    plot_center: egui::Pos2,
) -> egui::Pos2 {
    for pressed in [true, false] {
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                events: vec![
                    egui::Event::PointerMoved(plot_center),
                    egui::Event::PointerButton {
                        pos: plot_center,
                        button: egui::PointerButton::Secondary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                ..Default::default()
            },
            |ctx| morph.draw_canvas(ctx),
        );
    }

    let menu_out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| morph.draw_canvas(ctx),
    );
    let mut save_png_center = None;
    let mut disabled_dataset_buttons = 0usize;
    if let Some(update) = menu_out.platform_output.accesskit_update {
        for (_, node) in &update.nodes {
            match node.label() {
                Some("Export CSV" | "Export JSON" | "Table View") => {
                    assert!(
                        node.is_disabled(),
                        "Empty dataset action {:?} must stay disabled on MorphogenesisTab",
                        node.label()
                    );
                    disabled_dataset_buttons += 1;
                }
                Some("Save PNG") if !node.is_disabled() => {
                    if let Some(b) = node.bounds() {
                        save_png_center = Some(egui::pos2(
                            ((b.x0 + b.x1) * 0.5) as f32,
                            ((b.y0 + b.y1) * 0.5) as f32,
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    assert!(disabled_dataset_buttons >= 3);
    save_png_center.expect("Save PNG button present in right-click context menu")
}

fn click_save_png_and_capture(
    ctx: &egui::Context,
    morph: &mut crate::tabs::morphogenesis::MorphogenesisTab,
    screen: egui::Rect,
    btn_pos: egui::Pos2,
) -> (String, Vec<u8>) {
    let (_, captured_save) = egui_plot::file_io::capture_save_dialog_for_test(|| {
        for pressed in [true, false] {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(screen),
                    events: vec![
                        egui::Event::PointerMoved(btn_pos),
                        egui::Event::PointerButton {
                            pos: btn_pos,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                    ..Default::default()
                },
                |ctx| morph.draw_canvas(ctx),
            );
        }
    });
    captured_save.expect("captured Save PNG payload from context menu")
}

#[test]
fn test_morphogenesis_right_click_save_png_and_disabled_menu_gating() {
    let mut morph = crate::tabs::morphogenesis::MorphogenesisTab::default();
    for _ in 0..150 {
        if morph.worker.update().is_some() || morph.worker.latest_snapshot().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(960.0, 720.0));
    let initial_out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| {
            egui_plot::clear_last_plot_datasets();
            morph.draw_canvas(ctx);
        },
    );
    if let Some(update) = initial_out.platform_output.accesskit_update {
        for (_, node) in &update.nodes {
            if matches!(
                node.label(),
                Some("Export CSV" | "Export JSON" | "Table View")
            ) {
                assert!(node.is_disabled());
            } else if node.label() == Some("Save PNG") {
                assert!(!node.is_disabled());
            }
        }
    }

    let subplots = egui_plot::get_last_plot_subplots();
    assert_eq!(subplots.len(), 1);
    assert!(subplots[0].datasets.is_empty());
    let btn_pos = open_context_menu_and_find_save_png(
        &ctx,
        &mut morph,
        screen,
        subplots[0].screen_rect.center(),
    );
    let (filename, saved_png) = click_save_png_and_capture(&ctx, &mut morph, screen, btn_pos);
    assert_eq!(filename, "plot.png");
    let _ = std::fs::write("/tmp/morphogenesis_turing_export.png", &saved_png);
    let (_, _, pixels) =
        egui_plot::export_primitives::decode_png_rgba(&saved_png).expect("decode saved PNG");
    let distinct: HashSet<[u8; 4]> = pixels.into_iter().collect();
    assert!(distinct.len() >= 3);
}

#[test]
fn test_all_seven_painter_simulations_png_svg_gif_csv_json() {
    let painter_ids = [
        ("GraphEditorTool", None),
        ("AlgorithmVisualizerTool", None),
        ("NetworkMetricsTool", None),
        ("NetworkPropagationTool", None),
        (
            "NeuralNetworkVizTool",
            Some("/tmp/neural_network_painter_export.png"),
        ),
        ("GridWorldTool", None),
        (
            "QTableInspectorTool",
            Some("/tmp/q_table_painter_export.png"),
        ),
    ];

    for (id, screenshot_path) in painter_ids {
        let mut tool = build_tool_by_id(id);
        let caps = tool.export_capabilities();
        assert!(caps.dataset && caps.figure);
        if id == "QTableInspectorTool" {
            for _ in 0..25 {
                tool.step_simulation();
            }
        }

        verify_single_figure_tool(id, build_tool_by_id(id), true);

        let mut framework = SimulationFramework::new("__painter_audit__");
        framework.export_options.width = 480;
        framework.export_options.height = 360;
        framework.export_gif_frames = 3;
        framework.active_tool = Some(tool);

        if let Some(path) = screenshot_path {
            run_modal_export_after_clear(&mut framework, ExportFormat::Png);
            if let Some(bytes) = framework.export_queue.last_exported_bytes.as_deref() {
                let _ = std::fs::write(path, bytes);
            }
        }

        run_modal_export_after_clear(&mut framework, ExportFormat::Gif);
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Success
        );
        let gif = framework
            .export_queue
            .last_exported_bytes
            .as_deref()
            .unwrap_or_else(|| panic!("Missing GIF bytes for {id}"));
        assert!(gif.starts_with(b"GIF89a"));
    }
}
