#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::camera::{Camera3D, Plot3DViewport};
use eframe::egui;
use egui_plot::{CanvasCallout, Line, PlotMemory, PlotPoints, SavedCallout};

fn viewport_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn plot_3d_viewport_suppresses_pois_hover_snapping_and_callouts() {
    let ctx = egui::Context::default();
    let mut camera = Camera3D::default();
    let mut plot_id = egui::Id::NULL;
    let mut plot_rect = egui::Rect::NOTHING;

    egui_plot::reset_detect_critical_points_call_count();
    let _ = ctx.run(viewport_input(Vec::new()), |sub_ctx| {
        egui::CentralPanel::default().show(sub_ctx, |ui| {
            let pr = Plot3DViewport::new("optout_3d_vp", 4.0).show(ui, &mut camera, |plot_ui| {
                plot_id = plot_ui.plot_id();
                plot_ui.line(Line::new(
                    "proj_curve",
                    PlotPoints::new(vec![[-2.0, -1.0], [0.0, 2.0], [2.0, -1.0]]),
                ));
            });
            plot_rect = pr.response.rect;
        });
    });

    if let Some(mut mem) = PlotMemory::load(&ctx, plot_id) {
        mem.callouts.push(SavedCallout::new(
            [0.0, 2.0],
            "stale 2d callout",
            [20.0, -20.0],
        ));
        mem.store(&ctx, plot_id);
    }
    egui_plot::push_pending_callout(
        plot_id,
        CanvasCallout {
            target_point: [0.0, 2.0],
            label_offset: [24.0, -24.0],
            text: "queued 2d callout".to_string(),
        },
    );

    let center = plot_rect.center();
    let hover_events = vec![egui::Event::PointerMoved(center)];
    let mut hovered_item = None;
    let _ = ctx.run(viewport_input(hover_events), |sub_ctx| {
        egui::CentralPanel::default().show(sub_ctx, |ui| {
            let pr = Plot3DViewport::new("optout_3d_vp", 4.0).show(ui, &mut camera, |plot_ui| {
                plot_ui.line(Line::new(
                    "proj_curve",
                    PlotPoints::new(vec![[-2.0, -1.0], [0.0, 2.0], [2.0, -1.0]]),
                ));
            });
            hovered_item = pr.hovered_plot_item;
        });
    });

    assert_eq!(egui_plot::detect_critical_points_call_count(), 0);
    assert!(hovered_item.is_none());
    let mem = PlotMemory::load(&ctx, plot_id).expect("3D viewport memory");
    assert!(mem.callouts.is_empty());
}

#[test]
fn plot_3d_viewport_opts_out_of_2d_csv_import_and_paste() {
    let ctx = egui::Context::default();
    let mut camera = Camera3D::default();
    let mut plot_id = egui::Id::NULL;
    let paste_out = ctx.run(
        viewport_input(vec![egui::Event::Paste("x,y\n0,1\n1,2\n".to_string())]),
        |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                Plot3DViewport::new("optout_3d_csv", 4.0).show(ui, &mut camera, |plot_ui| {
                    plot_id = plot_ui.plot_id();
                    plot_ui.line(Line::new(
                        "proj_curve",
                        PlotPoints::new(vec![[-2.0, -1.0], [0.0, 2.0], [2.0, -1.0]]),
                    ));
                });
            });
        },
    );
    let has_import_btn = paste_out
        .shapes
        .iter()
        .any(|s| format!("{:?}", s.shape).contains("Import CSV"));
    assert!(
        !has_import_btn,
        "Plot3DViewport must hide 2D Import CSV toolbar controls"
    );
    let mem_after = PlotMemory::load(&ctx, plot_id).expect("3D viewport memory after paste");
    assert!(mem_after.imported_datasets.is_empty());
}

#[cfg(all(feature = "pure_math", feature = "physics", feature = "ai"))]
#[test]
fn all_six_plot_3d_viewport_tools_emit_zero_2d_pois_across_interactions() {
    let tool_ids: &[&str] = &[
        "CurvatureHeatmap",
        "SurfaceViewer",
        "CrystalViewer",
        "Gaussian Splatting",
        "AttractorPlotter",
        "SpinVisualizer",
        #[cfg(feature = "applied")]
        "ImplicitCadStudio",
    ];
    let hover_pos = egui::pos2(760.0, 420.0);

    for &expected_id in tool_ids {
        let meta = inventory::iter::<crate::framework::ToolMetadata>()
            .find(|m| m.id == expected_id)
            .unwrap_or_else(|| panic!("missing 3D tool metadata for {expected_id}"));
        let mut tool = (meta.build)();
        let ctx = egui::Context::default();
        egui_plot::reset_detect_critical_points_call_count();

        let _ = ctx.run(viewport_input(Vec::new()), |sub_ctx| tool.show(sub_ctx));
        let click_events = vec![
            egui::Event::PointerMoved(hover_pos),
            egui::Event::PointerButton {
                pos: hover_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: hover_pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ];
        let _ = ctx.run(viewport_input(click_events), |sub_ctx| tool.show(sub_ctx));

        assert_eq!(
            egui_plot::detect_critical_points_call_count(),
            0,
            "3D tool '{expected_id}' must never invoke 2D critical-point detection"
        );
    }
}
