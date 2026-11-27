use super::presets::PredictorMode;
use super::surfaces::{push_sculpt_bump, LandscapeSurface, SculptBump};
use super::{LandscapeViewMode, LossLandscapeTool};
use crate::framework::{ExportFormat, InteractiveTool, SimulationFramework};
use eframe::egui;
use egui_plot::{series_share_x, ExportPrimitive, TableSeriesRef};

fn raw_frame() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

#[test]
fn test_idle_frames_run_zero_surface_grid_evaluations_and_recompute_on_edits() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    let _ = ctx.run(raw_frame(), |c| tool.show(c));
    assert_eq!(tool.cache.grid_eval_count, 1);

    for _ in 0..4 {
        let _ = ctx.run(raw_frame(), |c| tool.show(c));
    }
    assert_eq!(tool.cache.grid_eval_count, 1);

    tool.view_mode = LandscapeViewMode::Contour2D;
    for _ in 0..3 {
        let _ = ctx.run(raw_frame(), |c| tool.show(c));
    }
    assert_eq!(tool.cache.grid_eval_count, 1);

    tool.w_max = 5.0;
    let _ = ctx.run(raw_frame(), |c| tool.show(c));
    assert_eq!(tool.cache.grid_eval_count, 2);

    tool.resolution = 30;
    let _ = ctx.run(raw_frame(), |c| tool.show(c));
    assert_eq!(tool.cache.grid_eval_count, 3);

    tool.set_target_table_text("0.0, 1.5\n1.0, 3.5\n2.0, 5.5");
    let _ = ctx.run(raw_frame(), |c| tool.show(c));
    assert_eq!(tool.cache.grid_eval_count, 4);

    tool.predictor_mode = PredictorMode::Custom;
    tool.math_input.set_expression("w * x * x + b");
    let _ = ctx.run(raw_frame(), |c| tool.show(c));
    assert_eq!(tool.cache.grid_eval_count, 5);

    push_sculpt_bump(&mut tool.sculpt_bumps, SculptBump::new(1.0, 1.0, 3.0, 0.5));
    let _ = ctx.run(raw_frame(), |c| tool.show(c));
    assert_eq!(tool.cache.grid_eval_count, 6);
}

#[test]
fn test_2d_contours_batches_raster_image_eight_contours_and_shared_x_loss_slices() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::Rosenbrock);
    tool.resolution = 50;
    tool.view_mode = LandscapeViewMode::Contour2D;

    egui_plot::clear_last_plot_datasets();
    let _ = ctx.run(raw_frame(), |c| tool.show(c));
    assert!(tool.cache.last_2d_has_plot_image);
    assert_eq!(tool.cache.last_2d_contour_item_count, 8);

    let prims = egui_plot::get_last_plot_primitives();
    let raster_count = prims
        .iter()
        .filter(|p| matches!(p, ExportPrimitive::RasterImage { .. }))
        .count();
    assert_eq!(raster_count, 1);

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 50);
    assert!(tool.export_datasets_3d().is_empty());
    let refs: Vec<TableSeriesRef<'_>> = datasets
        .iter()
        .map(|(name, pts)| TableSeriesRef {
            name,
            points: pts,
            color: egui::Color32::LIGHT_BLUE,
        })
        .collect();
    assert!(series_share_x(&refs));
    assert!(datasets[0].0.starts_with("b = "));
    let expected_loss = tool.surface_evaluator().evaluate(-2.0, -1.0);
    assert!((datasets[0].1[0][0] - (-2.0)).abs() < 1e-9);
    assert!((datasets[0].1[0][1] - expected_loss).abs() < 1e-6);
}

#[test]
fn test_2d_and_3d_csv_json_exports_include_numeric_loss_values() {
    let temp_dir = std::env::temp_dir().join("oxidize_loss_landscape_export_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("temp dir");

    let mut tool_3d = LossLandscapeTool::default();
    tool_3d.apply_surface_preset(LandscapeSurface::Rosenbrock);
    let ctx = egui::Context::default();
    let _ = ctx.run(raw_frame(), |c| tool_3d.show(c));
    let ptr_1 = tool_3d.cache.scene_builder.sidecar().shared_ptr();
    let _ = ctx.run(raw_frame(), |c| tool_3d.show(c));
    assert_eq!(tool_3d.cache.last_3d_plot_item_count, 1);
    assert_eq!(ptr_1, tool_3d.cache.scene_builder.sidecar().shared_ptr());

    let mut fw = SimulationFramework::new("__loss_export_3d__");
    fw.active_tool = Some(Box::new(tool_3d));
    fw.export_queue.destination_dir = Some(temp_dir.clone());
    let _ = ctx.run(raw_frame(), |c| {
        fw.queue_export(ExportFormat::Csv);
        fw.show(c, "fw_loss_3d");
    });
    let csv_3d = fw.export_queue.last_exported_content.clone().unwrap();
    assert!(csv_3d.starts_with("Dataset,X,Y,Z\n"));
    assert!(csv_3d.contains("Loss Surface,-2,-1,2509"));

    let mut tool_2d = LossLandscapeTool::default();
    tool_2d.apply_surface_preset(LandscapeSurface::Rosenbrock);
    tool_2d.view_mode = LandscapeViewMode::Contour2D;
    let mut fw2 = SimulationFramework::new("__loss_export_2d__");
    fw2.active_tool = Some(Box::new(tool_2d));
    fw2.export_queue.destination_dir = Some(temp_dir);
    let _ = ctx.run(raw_frame(), |c| {
        fw2.queue_export(ExportFormat::Json);
        fw2.show(c, "fw_loss_2d");
    });
    let json_2d = fw2.export_queue.last_exported_content.clone().unwrap();
    assert!(json_2d.contains("\"b = -1.00\""));
    assert!(json_2d.contains("2509"));
}

fn collect_shape_texts(out: &egui::FullOutput) -> Vec<(String, egui::Pos2)> {
    let mut texts = Vec::new();
    let mut stack: Vec<&egui::epaint::Shape> = out.shapes.iter().map(|c| &c.shape).collect();
    while let Some(s) = stack.pop() {
        match s {
            egui::epaint::Shape::Text(t) => {
                texts.push((
                    t.galley.job.text.clone(),
                    t.pos + t.galley.rect.center().to_vec2(),
                ));
            }
            egui::epaint::Shape::Vec(v) => stack.extend(v.iter().rev()),
            _ => {}
        }
    }
    texts
}

fn click_text_in_tool(ctx: &egui::Context, tool: &mut LossLandscapeTool, label: &str) {
    let out = ctx.run(raw_frame(), |c| tool.show(c));
    let pos = collect_shape_texts(&out)
        .into_iter()
        .find_map(|(txt, p)| txt.contains(label).then_some(p))
        .expect("target button text");
    let mut click = raw_frame();
    click.events = vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(click, |c| tool.show(c));
    let mut rel = raw_frame();
    rel.events = vec![egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    }];
    let _ = ctx.run(rel, |c| tool.show(c));
}

#[test]
fn test_loss_landscape_2d_and_3d_data_table_modals_render_aligned_loss_values() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::Rosenbrock);
    tool.view_mode = LandscapeViewMode::Contour2D;
    egui_plot::clear_last_plot_datasets();
    click_text_in_tool(&ctx, &mut tool, "Table View");
    let modal_2d = ctx.run(raw_frame(), |c| tool.show(c));
    let texts_2d: Vec<String> = collect_shape_texts(&modal_2d)
        .into_iter()
        .map(|t| t.0)
        .collect();
    assert!(texts_2d.iter().any(|t| t.contains("b = -1.00")));
    assert!(texts_2d.iter().any(|t| t.contains("2509")));

    let ds_2d = tool.export_datasets();
    let refs_2d: Vec<TableSeriesRef<'_>> = ds_2d
        .iter()
        .map(|(name, pts)| TableSeriesRef {
            name,
            points: pts,
            color: egui::Color32::LIGHT_BLUE,
        })
        .collect();
    let tsv_2d = egui_plot::format_tsv(&refs_2d);
    assert!(tsv_2d.starts_with("X\tb = -1.00\t"));
    assert!(tsv_2d.contains("-2\t2509\t"));

    tool.view_mode = LandscapeViewMode::View3D;
    let _ = ctx.run(raw_frame(), |c| tool.show(c));
    let series_3d = tool.cache.scene_builder.sidecar().series().to_vec();
    assert_eq!(series_3d[0].name, "Loss Surface");
    assert_eq!(series_3d[0].world[0], [-2.0, -1.0, 2509.0]);
}
