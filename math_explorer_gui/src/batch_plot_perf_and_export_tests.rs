#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::framework::{InteractiveTool, SimulationFramework};
use crate::tabs::ai::loss_landscape::{LandscapeViewMode, LossLandscapeTool};
use crate::tabs::analysis::complex_mapping::ComplexMappingTool;
use crate::tabs::fluid_dynamics::potential_flow::PotentialFlowTool;
use crate::tabs::geometry_topology::curvature_heatmap::CurvatureHeatmap;
use crate::tabs::geometry_topology::surface_viewer::SurfaceViewer;
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::export::{format_csv, format_csv_3d, format_json, format_json_3d, ExportOptions};
use egui_plot::export_primitives::{decode_png_rgba, export_primitives_png, export_primitives_svg};

fn hd_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1920.0, 1080.0),
        )),
        ..Default::default()
    }
}

fn assert_consolidated_and_lazy(
    ctx: &egui::Context,
    tool: &mut dyn InteractiveTool,
    expected_plots: usize,
    max_items_per_plot: usize,
) {
    let _ = ctx.run(hd_input(), |c| tool.show(c));
    egui_plot::clear_last_plot_datasets();
    egui_plot::reset_materialized_plot_datasets_call_count();
    let _ = ctx.run(hd_input(), |c| tool.show(c));

    let counts = egui_plot::get_last_plot_item_counts();
    assert_eq!(
        counts.len(),
        expected_plots,
        "Plot count mismatch for {}",
        tool.name()
    );
    for &count in &counts {
        assert!(
            count <= max_items_per_plot,
            "{} submitted {count} plot items (expected <= {max_items_per_plot})",
            tool.name()
        );
    }
    assert_eq!(
        egui_plot::materialized_plot_datasets_call_count(),
        0,
        "{} materialized datasets on an idle frame without Data Table or export",
        tool.name()
    );
}

#[test]
fn all_five_tools_consolidate_to_at_most_two_plot_items_and_zero_dataset_cloning() {
    let ctx = egui::Context::default();
    let mut curv = CurvatureHeatmap::default();
    assert_consolidated_and_lazy(&ctx, &mut curv, 1, 2);

    let mut surf = SurfaceViewer::default();
    assert_consolidated_and_lazy(&ctx, &mut surf, 1, 2);

    let mut flow = PotentialFlowTool::default();
    assert_consolidated_and_lazy(&ctx, &mut flow, 1, 3);

    let mut conformal = ComplexMappingTool::default();
    assert_consolidated_and_lazy(&ctx, &mut conformal, 2, 2);

    let mut loss = LossLandscapeTool::default();
    loss.view_mode = LandscapeViewMode::Contour2D;
    assert_consolidated_and_lazy(&ctx, &mut loss, 1, 2);
    loss.view_mode = LandscapeViewMode::View3D;
    assert_consolidated_and_lazy(&ctx, &mut loss, 1, 2);

    // Also verify SimulationFramework skips 2D and 3D dataset materialization on unchanged frames
    let mut fw = SimulationFramework::new("__batch_lazy_fw__");
    fw.active_tool = Some(Box::new(CurvatureHeatmap::default()));
    let _ = ctx.run(hd_input(), |c| fw.show(c, "batch_fw"));
    egui_plot::reset_materialized_plot_datasets_call_count();
    let _ = ctx.run(hd_input(), |c| fw.show(c, "batch_fw"));
    assert_eq!(egui_plot::materialized_plot_datasets_call_count(), 0);
}

#[test]
fn surface_tools_reproject_on_camera_orbit_with_zero_manifold_or_curvature_evals() {
    let ctx = egui::Context::default();

    let mut curv = CurvatureHeatmap::default();
    let _ = ctx.run(hd_input(), |c| curv.show(c));
    let (eval0, proj0) = curv.debug_cache_counts();
    let _ = ctx.run(hd_input(), |c| curv.show(c));
    assert_eq!(curv.debug_cache_counts(), (eval0, proj0));
    curv.camera.yaw += 0.35;
    curv.camera.pitch = (curv.camera.pitch - 0.15).clamp(-1.4, 1.4);
    let _ = ctx.run(hd_input(), |c| curv.show(c));
    assert_eq!(curv.debug_cache_counts(), (eval0, proj0 + 1));

    let mut surf = SurfaceViewer::default();
    let _ = ctx.run(hd_input(), |c| surf.show(c));
    let (seval0, sproj0) = surf.debug_cache_counts();
    let _ = ctx.run(hd_input(), |c| surf.show(c));
    assert_eq!(surf.debug_cache_counts(), (seval0, sproj0));
    surf.camera.yaw -= 0.4;
    surf.camera.pitch = (surf.camera.pitch + 0.2).clamp(-1.4, 1.4);
    let _ = ctx.run(hd_input(), |c| surf.show(c));
    assert_eq!(surf.debug_cache_counts(), (seval0, sproj0 + 1));
}

fn verify_and_save_plot_exports(
    ctx: &egui::Context,
    tool: &mut dyn InteractiveTool,
    expect_3d_csv: bool,
    png_out_path: &str,
) {
    let opts = ExportOptions {
        width: 960,
        height: 640,
        ..ExportOptions::default()
    };
    egui_plot::clear_last_plot_datasets();
    let _ = ctx.run(hd_input(), |c| tool.show(c));

    let prims = egui_plot::get_last_plot_primitives();
    assert!(
        !prims.is_empty(),
        "Missing export primitives for {}",
        tool.name()
    );
    let png = export_primitives_png(&prims, &opts).expect("PNG export");
    let (w, h, rgba) = decode_png_rgba(&png).expect("decode PNG");
    assert_eq!((w, h), (960, 540));
    assert!(rgba
        .iter()
        .any(|px| px[0] != rgba[0][0] || px[1] != rgba[0][1] || px[2] != rgba[0][2]));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = std::fs::write(png_out_path, &png);
    #[cfg(target_arch = "wasm32")]
    assert!(!png_out_path.is_empty());

    let svg = export_primitives_svg(&prims, &opts).expect("SVG export");
    assert!(svg.contains("<svg") && svg.len() > 250);

    if expect_3d_csv {
        let ds3 = egui_plot::get_last_plot_datasets_3d();
        assert!(!ds3.is_empty());
        assert!(format_csv_3d(&ds3).lines().count() > 2);
        assert!(format_json_3d(&ds3).contains("\"z\""));
    } else {
        let ds2 = egui_plot::get_last_plot_datasets();
        assert!(!ds2.is_empty());
        assert!(format_csv(&ds2).lines().count() > 2);
        assert!(format_json(&ds2).len() > 20);
    }
}

#[test]
fn all_five_tools_export_valid_png_svg_obj_csv_and_json_headless() {
    let ctx = egui::Context::default();

    let mut curv = CurvatureHeatmap::default();
    verify_and_save_plot_exports(&ctx, &mut curv, true, "/tmp/curvature_heatmap_batched.png");
    let curv_mesh = curv
        .export_mesh_3d()
        .expect("Curvature 3D mesh hook")
        .expect("Curvature mesh");
    assert!(oxidize_core::mesh::export_mesh_to_obj_string(&curv_mesh)
        .expect("Curvature OBJ")
        .contains("\nf "));

    let mut surf = SurfaceViewer::default();
    verify_and_save_plot_exports(&ctx, &mut surf, true, "/tmp/surface_viewer_torus.png");
    let surf_mesh = surf
        .export_mesh_3d()
        .expect("Surface 3D mesh hook")
        .expect("Surface mesh");
    assert!(oxidize_core::mesh::export_mesh_to_obj_string(&surf_mesh)
        .expect("Surface OBJ")
        .contains("\nf "));

    let mut flow = PotentialFlowTool::default();
    verify_and_save_plot_exports(&ctx, &mut flow, false, "/tmp/potential_flow_arrows.png");

    let mut conformal = ComplexMappingTool::default();
    verify_and_save_plot_exports(
        &ctx,
        &mut conformal,
        false,
        "/tmp/complex_mapping_conformal.png",
    );

    let mut loss = LossLandscapeTool::default();
    loss.view_mode = LandscapeViewMode::Contour2D;
    loss.resolution = 48;
    verify_and_save_plot_exports(&ctx, &mut loss, false, "/tmp/loss_landscape_heatmap.png");
    assert!(loss
        .build_obj_mesh_string()
        .expect("Loss OBJ")
        .contains("\nf "));
}

fn measure_cold_and_warm_ms(ctx: &egui::Context, tool: &mut dyn InteractiveTool) -> (f64, f64) {
    let cold_start = PlatformInstant::now();
    let _ = ctx.run(hd_input(), |c| tool.show(c));
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
    let mut best_warm_ms = f64::MAX;
    for _ in 0..3 {
        let start = PlatformInstant::now();
        let _ = ctx.run(hd_input(), |c| tool.show(c));
        best_warm_ms = best_warm_ms.min(start.elapsed().as_secs_f64() * 1000.0);
    }
    (cold_ms, best_warm_ms)
}

#[test]
fn headless_1920x1080_frames_respect_frame_budget_across_slider_positions() {
    let budget_ms = if cfg!(debug_assertions) { 500.0 } else { 16.67 };
    let ctx = egui::Context::default();
    let _ = ctx.run(hd_input(), |_| {});

    let mut curv = CurvatureHeatmap::default();
    for res in [10, 40, 80, 120] {
        curv.set_grid_resolutions(res, res);
        let (cold_ms, warm_ms) = measure_cold_and_warm_ms(&ctx, &mut curv);
        assert!(
            cold_ms < budget_ms && warm_ms < budget_ms,
            "CurvatureHeatmap res={res}: cold={cold_ms:.2}ms warm={warm_ms:.2}ms"
        );
    }

    let mut surf = SurfaceViewer::default();
    for res in [10, 40, 80, 120] {
        surf.set_grid_resolutions(res, res);
        let (cold_ms, warm_ms) = measure_cold_and_warm_ms(&ctx, &mut surf);
        assert!(
            cold_ms < budget_ms && warm_ms < budget_ms,
            "SurfaceViewer res={res}: cold={cold_ms:.2}ms warm={warm_ms:.2}ms"
        );
    }

    let mut flow = PotentialFlowTool::default();
    for res in [10, 40, 80, 120] {
        flow.grid_size = res;
        let (cold_ms, warm_ms) = measure_cold_and_warm_ms(&ctx, &mut flow);
        assert!(
            cold_ms < budget_ms && warm_ms < budget_ms,
            "PotentialFlowTool res={res}: cold={cold_ms:.2}ms warm={warm_ms:.2}ms"
        );
    }

    let mut conformal = ComplexMappingTool::default();
    for res in [10, 50, 100, 150] {
        conformal.grid_density = res;
        let (cold_ms, warm_ms) = measure_cold_and_warm_ms(&ctx, &mut conformal);
        assert!(
            cold_ms < budget_ms && warm_ms < budget_ms,
            "ComplexMappingTool res={res}: cold={cold_ms:.2}ms warm={warm_ms:.2}ms"
        );
    }

    let mut loss = LossLandscapeTool::default();
    for mode in [LandscapeViewMode::Contour2D, LandscapeViewMode::View3D] {
        loss.view_mode = mode;
        for res in [10, 40, 80, 120] {
            loss.resolution = res;
            let (cold_ms, warm_ms) = measure_cold_and_warm_ms(&ctx, &mut loss);
            assert!(
                cold_ms < budget_ms && warm_ms < budget_ms,
                "LossLandscapeTool {mode:?} res={res}: cold={cold_ms:.2}ms warm={warm_ms:.2}ms"
            );
        }
    }
}
