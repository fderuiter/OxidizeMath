#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{
    SdsNeRfDistillationTool, SdsOptimizerKind, SdsPromptPreset, MAX_RAY_GRID, MAX_RAY_SAMPLES,
};
use crate::framework::InteractiveTool;

#[test]
fn sds_nerf_distillation_bounds_and_roundtrip() {
    let mut tool = SdsNeRfDistillationTool::default();
    assert!(tool.has_finite_parameters());
    assert!(tool.ray_grid_size <= MAX_RAY_GRID);
    assert!(tool.samples_per_ray <= MAX_RAY_SAMPLES);

    tool.ray_grid_size = 128;
    tool.samples_per_ray = 96;
    tool.clamp_parameters();
    assert_eq!(tool.ray_grid_size, MAX_RAY_GRID);
    assert_eq!(tool.samples_per_ray, MAX_RAY_SAMPLES);

    tool.prompt_preset = SdsPromptPreset::CyanOctahedron;
    tool.optimizer_kind = SdsOptimizerKind::Sgd;
    tool.guidance_scale = 8.5;
    let saved = tool.save_state().expect("serializes");
    assert!(tool.state_bounds().validate_json(&saved).is_ok());

    let mut restored = SdsNeRfDistillationTool::default();
    restored.try_load_state(&saved).expect("loads valid state");
    assert_eq!(restored.prompt_preset, SdsPromptPreset::CyanOctahedron);
    assert_eq!(restored.optimizer_kind, SdsOptimizerKind::Sgd);
    assert!((restored.guidance_scale - 8.5).abs() < 1e-9);

    assert!(restored
        .try_load_state(r#"{"guidance_scale": 6.0, "learning_rate": 0.1, "ray_grid_size": 64, "samples_per_ray": 12}"#)
        .is_err());
    assert!(restored
        .try_load_state(r#"{"guidance_scale": 6.0, "learning_rate": 0.1, "ray_grid_size": 16, "samples_per_ray": 40}"#)
        .is_err());
}

#[test]
fn sds_nerf_distillation_reduces_loss_with_adam_and_sgd_and_exports_raster() {
    let mut adam_tool = SdsNeRfDistillationTool {
        ray_grid_size: 10,
        samples_per_ray: 10,
        optimizer_kind: SdsOptimizerKind::Adam,
        ..SdsNeRfDistillationTool::default()
    };
    adam_tool.reset_distillation();
    let init_adam = adam_tool.current_mse;
    for _ in 0..10 {
        adam_tool.step_simulation();
    }
    assert!(adam_tool.current_mse < init_adam);

    let mut sgd_tool = SdsNeRfDistillationTool {
        ray_grid_size: 10,
        samples_per_ray: 10,
        learning_rate: 0.30,
        optimizer_kind: SdsOptimizerKind::Sgd,
        ..SdsNeRfDistillationTool::default()
    };
    sgd_tool.reset_distillation();
    let init_sgd = sgd_tool.current_mse;
    for _ in 0..10 {
        sgd_tool.step_simulation();
    }
    assert!(sgd_tool.current_mse < init_sgd);

    let export_img = adam_tool.export_raster_image().expect("raster export");
    assert!(export_img.width() > 0 && export_img.height() > 0);
}

#[test]
fn sds_nerf_distillation_renders_headless_viewport() {
    let mut tool = SdsNeRfDistillationTool::default();
    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            tool.show_inside(ui);
        });
    });
}
