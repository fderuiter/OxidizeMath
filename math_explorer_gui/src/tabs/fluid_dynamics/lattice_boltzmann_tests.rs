#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::async_sim::unified::UnifiedSimTool;

#[test]
fn test_lbm_unified_calibrated_viscosity_and_single_uniform_inlet() {
    let tool = UnifiedSimTool::<LbmUnified>::new();
    let params = tool.params.read().expect("params lock").clone();
    assert_eq!(params.get("viscosity").copied(), Some(0.02));
    assert_eq!(params.get("inlet_velocity").copied(), Some(LBM_INLET_U));
    assert_eq!(params.get("angle_of_attack").copied(), Some(0.0));
    assert_eq!(params.get("naca_profile").copied(), Some(0.0));
    assert_eq!(params.get("field_view").copied(), Some(0.0));

    let reynolds = 0.1 * 10.0 / params["viscosity"];
    assert!((reynolds - 50.0).abs() < 1e-9);

    let model = LbmUnified::new(&params);
    assert!((model.solver.collision_model.tau - 0.56).abs() < 1e-12);
    for y in 0..50 {
        for x in 0..2 {
            let v = model.solver.get_velocity_magnitude(x, y);
            assert!(
                (v - 0.1).abs() < 1e-6,
                "uniform inlet expected at ({x},{y})"
            );
        }
    }
    // Columns x = 2..5 at y = 25 were previously overwritten by a narrow jet inlet; verify quiescent interior start.
    assert!(model.solver.get_velocity_magnitude(4, 25) < 1e-9);

    let mut evolved = model;
    for _ in 0..250 {
        evolved.step(&params);
    }
    let snap = evolved.get_snapshot();
    let px = snap.pixels.read().expect("pixels lock");
    let (w, h, scale) = (snap.width, snap.height, 6usize);
    let mut img = eframe::egui::ColorImage::new(
        [w * scale, h * scale],
        vec![Color32::BLACK; w * scale * h * scale],
    );
    for y in 0..(h * scale) {
        for x in 0..(w * scale) {
            img.pixels[y * (w * scale) + x] = px[(y / scale) * w + (x / scale)];
        }
    }
    if let Ok(png) = egui_plot::export::color32_image_to_png(&img) {
        let _ = std::fs::write("/tmp/lbm_re50_calibrated_wake.png", png);
    }
}
