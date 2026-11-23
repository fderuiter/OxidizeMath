#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::surfaces::{LandscapeSurface, MAX_SCULPT_BUMPS};
use super::{CanvasInteractionMode, LandscapeViewMode, LossLandscapeTool};
use crate::framework::{InputMode, InteractionContext, InteractiveTool};
use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

fn wasm_raw_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        events,
        ..Default::default()
    }
}

#[wasm_bindgen_test]
fn wasm_loss_landscape_on_brush_and_32_bump_url_budget() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::MonkeySaddle);
    tool.view_mode = LandscapeViewMode::View3D;
    let rect = egui::Rect::from_min_max(egui::pos2(260.0, 60.0), egui::pos2(960.0, 700.0));

    let _ = ctx.run(wasm_raw_input(Vec::new()), |egui_ctx| {
        egui::CentralPanel::default().show(egui_ctx, |ui| {
            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
            for step in 0..40 {
                let frac = step as f32 / 40.0;
                let pos = egui::pos2(rect.min.x + 20.0 + frac * 650.0, rect.min.y + frac * 580.0);
                let brush_ctx = InteractionContext {
                    pointer_pos: Some(pos),
                    delta: egui::vec2(6.0, 6.0),
                    is_dragging: true,
                    is_clicked: false,
                    response: &response,
                    input_mode: InputMode::Mouse,
                    multi_touch: None,
                    keys_down: std::collections::HashSet::new(),
                    modifiers: egui::Modifiers::NONE,
                    brush_stroke: None,
                };
                tool.on_brush(&brush_ctx);
            }
        });
    });

    assert_eq!(tool.sculpt_bumps.len(), MAX_SCULPT_BUMPS);
    let envelope = tool
        .export_preset_json()
        .expect("WASM preset export must succeed");
    assert!(envelope.len() <= crate::url_state::MAX_DECODED_URL_STATE_BYTES);
    crate::url_state::validate_url_preset_json(&envelope)
        .expect("WASM URL state validator must accept 32-bump preset");
}

#[wasm_bindgen_test]
fn wasm_loss_landscape_slingshot_race_and_orbit_suppression() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::Rosenbrock);
    tool.interaction_mode = CanvasInteractionMode::Slingshot;
    let yaw0 = tool.camera.yaw;
    let pitch0 = tool.camera.pitch;
    let p0 = egui::pos2(620.0, 360.0);
    let p1 = egui::pos2(740.0, 440.0);

    let _ = ctx.run(wasm_raw_input(Vec::new()), |c| tool.show(c));
    let _ = ctx.run(
        wasm_raw_input(vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |c| tool.show(c),
    );
    let _ = ctx.run(wasm_raw_input(vec![egui::Event::PointerMoved(p1)]), |c| {
        tool.show(c)
    });
    let _ = ctx.run(
        wasm_raw_input(vec![egui::Event::PointerButton {
            pos: p1,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]),
        |c| tool.show(c),
    );

    assert_eq!(tool.camera.yaw, yaw0);
    assert_eq!(tool.camera.pitch, pitch0);
    for _ in 0..30 {
        tool.step_simulation();
    }
    assert_eq!(tool.arena.step_count, 30);
    assert!(tool.has_finite_parameters());
}

#[wasm_bindgen_test]
fn wasm_batched_tools_headless_png_svg_obj_csv_json_exports() {
    use egui_plot::export::{format_csv, format_json, ExportOptions};
    use egui_plot::export_primitives::{decode_png_rgba, export_primitives_png, export_primitives_svg};
    use egui_plot::{series_share_x, TableSeriesRef};

    let ctx = egui::Context::default();
    let opts = ExportOptions::default();

    let mut loss = LossLandscapeTool {
        view_mode: LandscapeViewMode::Contour2D,
        resolution: 24,
        ..Default::default()
    };
    egui_plot::clear_last_plot_datasets();
    let _ = ctx.run(wasm_raw_input(Vec::new()), |c| loss.show(c));
    assert!(egui_plot::get_last_plot_item_counts().iter().all(|&n| n <= 2));

    let ds = egui_plot::get_last_plot_datasets();
    let refs: Vec<TableSeriesRef<'_>> = ds
        .iter()
        .map(|(n, p)| TableSeriesRef {
            name: n.as_str(),
            points: p.as_slice(),
            color: egui::Color32::WHITE,
        })
        .collect();
    assert!(series_share_x(&refs));
    assert!(format_csv(&ds).contains("Loss (b = "));
    assert!(format_json(&ds).contains("Loss (b = "));

    let prims = egui_plot::get_last_plot_primitives();
    let png = export_primitives_png(&prims, &opts).expect("WASM PNG export");
    let (w, h, _) = decode_png_rgba(&png).expect("decode WASM PNG");
    assert!(w > 0 && h > 0);
    let svg = export_primitives_svg(&prims, &opts).expect("WASM SVG export");
    assert!(svg.contains("<svg") && svg.contains("<image"));
    assert!(loss.build_obj_mesh_string().expect("WASM Loss OBJ").contains("\nv "));

    let surf = crate::tabs::geometry_topology::surface_viewer::SurfaceViewer::default();
    let surf_mesh = surf
        .export_mesh_3d()
        .expect("WASM Surface 3D mesh hook")
        .expect("WASM Surface mesh");
    assert!(oxidize_core::mesh::export_mesh_to_obj_string(&surf_mesh)
        .expect("WASM Surface OBJ")
        .contains("\nf "));
    let curv = crate::tabs::geometry_topology::curvature_heatmap::CurvatureHeatmap::default();
    let curv_mesh = curv
        .export_mesh_3d()
        .expect("WASM Curvature 3D mesh hook")
        .expect("WASM Curvature mesh");
    assert!(oxidize_core::mesh::export_mesh_to_obj_string(&curv_mesh)
        .expect("WASM Curvature OBJ")
        .contains("\nf "));
}
