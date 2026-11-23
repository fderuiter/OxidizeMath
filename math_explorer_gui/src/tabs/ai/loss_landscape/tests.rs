#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_custom_loss_landscape_mse_and_preset_roundtrip() {
    let mut tool = LossLandscapeTool {
        predictor_mode: PredictorMode::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("w * x + b");
    // At true solution w = 2, b = 1 for y = 2x + 1, MSE must be 0
    assert!((tool.evaluate_mse_at(2.0, 1.0).unwrap() - 0.0).abs() < 1e-12);

    tool.import_preset_json(NONCONVEX_SINUSOIDAL_PRESET_JSON)
        .expect("non-convex preset should load");
    assert!(tool.evaluate_mse_at(1.5, 0.8).unwrap().is_finite());
    assert!(tool.import_preset_json(STARTER_PRESET_JSON).is_ok());
}

#[test]
fn test_loss_landscape_formula_and_table_targets_mse_mae_huber_and_zero_alloc() {
    let mut tool = LossLandscapeTool::default();

    // 1. Table target with an outlier at x = 2: (0, 1), (1, 3), (2, 15), (3, 7)
    tool.set_target_table_text("0.0, 1.0\n1.0, 3.0\n2.0, 15.0\n3.0, 7.0");
    assert!(tool.table_parse_error.is_none());
    // At (w=2, b=1), residuals are [0, 0, -10, 0]
    tool.loss_metric = LossMetric::Mse;
    let mse = tool.evaluate_loss_at(2.0, 1.0).expect("mse");
    assert!((mse - 25.0).abs() < 1e-12); // 100 / 4

    tool.loss_metric = LossMetric::Mae;
    let mae = tool.evaluate_loss_at(2.0, 1.0).expect("mae");
    assert!((mae - 2.5).abs() < 1e-12); // 10 / 4

    tool.loss_metric = LossMetric::Huber;
    tool.huber_delta = 2.0;
    let huber = tool.evaluate_loss_at(2.0, 1.0).expect("huber");
    // delta * (|r| - 0.5*delta) = 2 * (10 - 1) = 18; / 4 = 4.5
    assert!((huber - 4.5).abs() < 1e-12);
    assert!(huber < mse && huber > mae);

    // Transient invalid table edit preserves last valid targets
    tool.set_target_table_text("0.0, 1.0\nbad_row");
    assert!(tool.table_parse_error.is_some());
    assert!((tool.evaluate_loss_at(2.0, 1.0).expect("retained") - 4.5).abs() < 1e-12);

    // 2. Formula target mode + custom predictor zero-allocation hot loop
    tool.target_mode = TargetDataMode::Formula;
    tool.target_formula_bar.set_expression("3 * x - 2");
    tool.target_x_bounds = [-2.0, 2.0];
    tool.target_sample_count = 11;
    tool.refresh_target_samples();

    tool.predictor_mode = PredictorMode::Custom;
    tool.math_input.set_expression("w * x + b");
    assert!(tool.evaluate_loss_at(3.0, -2.0).expect("exact match") < 1e-12);

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (val, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut acc = 0.0;
        for i in 0..32 {
            acc += tool
                .evaluate_loss_at(2.5 + i as f64 * 0.01, -1.5)
                .unwrap_or(0.0);
        }
        acc
    });
    assert!(val.is_finite());

    // 3. Persistence roundtrip for Formula and Table modes
    let json = tool.save_state().expect("save formula state");
    assert!(json.len() < 4096);
    let mut restored = LossLandscapeTool::default();
    restored
        .try_load_state(&json)
        .expect("restore formula state");
    assert_eq!(restored.target_mode, TargetDataMode::Formula);
    assert_eq!(restored.loss_metric, LossMetric::Huber);
    assert!((restored.huber_delta - 2.0).abs() < 1e-12);
    assert!(restored.evaluate_loss_at(3.0, -2.0).expect("restored") < 1e-12);
}
