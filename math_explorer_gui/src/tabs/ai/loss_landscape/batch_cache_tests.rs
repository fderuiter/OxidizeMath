#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::surfaces::LandscapeSurface;
use super::{LandscapeViewMode, LossLandscapeTool};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::export::{format_csv, format_json, ExportOptions};
use egui_plot::export_primitives::{decode_png_rgba, export_primitives_png, export_primitives_svg};
use egui_plot::{series_share_x, TableSeriesRef};

fn run_headless_frame(ctx: &egui::Context, tool: &mut LossLandscapeTool) {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1920.0, 1080.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |c| tool.show(c));
}

#[test]
fn loss_landscape_clamps_high_grid_preset_and_enforces_interactive_cap() {
    let mut tool = LossLandscapeTool::default();
    let base_json = tool.save_state().expect("save_state");
    let mut val: serde_json::Value = serde_json::from_str(&base_json).expect("valid json");
    val["resolution"] = serde_json::json!(1000);
    tool.try_load_state(&val.to_string())
        .expect("resolution 1000 must be accepted up to MAX_GRID_SIZE and clamped");
    assert_eq!(tool.resolution, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
    assert!(tool.has_finite_parameters());

    val["resolution"] = serde_json::json!(1001);
    assert!(
        tool.try_load_state(&val.to_string()).is_err(),
        "resolution above MAX_GRID_SIZE (1000) must be rejected"
    );

    tool.resolution = crate::presets::MAX_INTERACTIVE_GRID_SIZE + 1;
    assert!(!tool.has_finite_parameters());
}

#[test]
fn loss_landscape_skips_eval_on_unchanged_frames_and_3d_camera_orbit() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::MonkeySaddle);
    tool.view_mode = LandscapeViewMode::Contour2D;
    tool.resolution = 40;

    run_headless_frame(&ctx, &mut tool);
    let eval_after_first = tool.cache.eval_count;
    let uploads_after_first = tool.cache.texture_upload_count;
    assert!(eval_after_first >= 1600);
    assert_eq!(uploads_after_first, 1);

    egui_plot::reset_materialized_plot_datasets_call_count();
    run_headless_frame(&ctx, &mut tool);
    assert_eq!(tool.cache.eval_count, eval_after_first);
    assert_eq!(tool.cache.texture_upload_count, uploads_after_first);
    assert_eq!(egui_plot::materialized_plot_datasets_call_count(), 0);

    tool.view_mode = LandscapeViewMode::View3D;
    run_headless_frame(&ctx, &mut tool);
    let eval_in_3d = tool.cache.eval_count;
    let proj_before_orbit = tool.cache.projection_3d_count;

    tool.camera.yaw += 0.42;
    tool.camera.pitch = (tool.camera.pitch - 0.18).clamp(-1.4, 1.4);
    run_headless_frame(&ctx, &mut tool);
    assert_eq!(
        tool.cache.eval_count, eval_in_3d,
        "Camera rotation must reproject cached 3D vertices with zero loss evaluations"
    );
    assert_eq!(tool.cache.projection_3d_count, proj_before_orbit + 1);
}

#[test]
fn loss_landscape_batches_items_shares_x_and_exports_numeric_loss_matrix() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::Rosenbrock);
    tool.resolution = 25;
    tool.view_mode = LandscapeViewMode::Contour2D;

    egui_plot::clear_last_plot_datasets();
    run_headless_frame(&ctx, &mut tool);

    let counts_2d = egui_plot::get_last_plot_item_counts();
    assert_eq!(counts_2d.len(), 1);
    assert!(
        counts_2d[0] <= 2,
        "2D Contour view must submit <= 2 plot items, got {}",
        counts_2d[0]
    );

    let datasets = egui_plot::get_last_plot_datasets();
    assert_eq!(datasets.len(), 25, "Expected one row series per bias slice");
    let refs: Vec<TableSeriesRef<'_>> = datasets
        .iter()
        .map(|(name, pts)| TableSeriesRef {
            name: name.as_str(),
            points: pts.as_slice(),
            color: egui::Color32::WHITE,
        })
        .collect();
    assert!(
        series_share_x(&refs),
        "Loss Landscape rows must share X (weight w) to open a single multi-column Data Table"
    );

    let csv = format_csv(&datasets);
    let mut lines = csv.lines();
    let header = lines.next().expect("CSV header");
    assert_eq!(header, "Dataset,X,Y");
    let first_row: Vec<&str> = lines.next().expect("first data row").split(',').collect();
    assert_eq!(first_row.len(), 3);
    assert!(first_row[0].trim_matches('"').starts_with("Loss (b = "));
    let parsed_w: f64 = first_row[1].parse().expect("numeric weight w");
    let parsed_loss: f64 = first_row[2].parse().expect("numeric loss value");
    let expected_loss = tool
        .surface_evaluator()
        .evaluate(tool.w_min, tool.b_min);
    assert!((parsed_w - tool.w_min).abs() < 1e-6);
    assert!((parsed_loss - expected_loss).abs() < 1e-6);

    let json = format_json(&datasets);
    assert!(json.contains("\"Loss (b = "));

    verify_loss_landscape_raster_and_3d_exports(&ctx, &mut tool);
}

fn verify_loss_landscape_raster_and_3d_exports(
    ctx: &egui::Context,
    tool: &mut LossLandscapeTool,
) {
    let prims = egui_plot::get_last_plot_primitives();
    assert!(!prims.is_empty());
    assert!(prims
        .iter()
        .any(|p| matches!(p, egui_plot::export_primitives::ExportPrimitive::RasterImage { .. })));

    let opts = ExportOptions::default();
    let png = export_primitives_png(&prims, &opts).expect("PNG export");
    let (w, h, rgba) = decode_png_rgba(&png).expect("valid PNG");
    assert!(w > 0 && h > 0 && !rgba.is_empty());
    let svg = export_primitives_svg(&prims, &opts).expect("SVG export");
    assert!(svg.contains("<image") && svg.contains("<polyline"));

    let obj = tool.build_obj_mesh_string().expect("OBJ export");
    assert!(obj.contains("\nv ") && obj.contains("\nf "));

    tool.view_mode = LandscapeViewMode::View3D;
    egui_plot::clear_last_plot_datasets();
    run_headless_frame(ctx, tool);
    let counts_3d = egui_plot::get_last_plot_item_counts();
    assert_eq!(counts_3d.len(), 1);
    assert!(
        counts_3d[0] <= 2,
        "3D Surface view must submit <= 2 plot items, got {}",
        counts_3d[0]
    );
}
