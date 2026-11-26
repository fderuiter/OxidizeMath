use std::f64::consts::PI;

use approx::assert_relative_eq;
use math_commons::primitives::{PositiveFloat, UnitInterval};
use math_commons::registry::TOLERANCE_STANDARD;

use super::*;

fn ui(v: f64) -> UnitInterval {
    UnitInterval::new(v).expect("valid unit interval")
}

#[test]
#[verified_engine::verified]
fn test_brier_and_murphy_identity_exact_and_binned() {
    let probs = [
        ui(0.2),
        ui(0.2),
        ui(0.2),
        ui(0.2),
        ui(0.8),
        ui(0.8),
        ui(0.8),
        ui(0.8),
    ];
    let outcomes = [false, false, false, true, false, true, true, true];

    let bs = brier_score(&probs, &outcomes).expect("brier");
    // Each group has size 4:
    // Group 0.2: 3 neg (0.04*3 = 0.12) + 1 pos (0.64) = 0.76
    // Group 0.8: 1 neg (0.64) + 3 pos (0.04*3 = 0.12) = 0.76
    // Mean BS = 1.52 / 8 = 0.19.
    assert_relative_eq!(bs, 0.19, epsilon = TOLERANCE_STANDARD);

    let decomp = murphy_decomposition(&probs, &outcomes).expect("murphy");
    // Base rate = 0.5 => UNC = 0.25.
    // Group 0.2 empirical = 0.25 => (0.2 - 0.25)^2 = 0.0025; Group 0.8 empirical = 0.75 => (0.8 - 0.75)^2 = 0.0025 => REL = 0.0025.
    // RES = 0.5*(0.25 - 0.5)^2 + 0.5*(0.75 - 0.5)^2 = 0.0625.
    // REL - RES + UNC = 0.0025 - 0.0625 + 0.25 = 0.19.
    assert_relative_eq!(decomp.brier_score, 0.19, epsilon = TOLERANCE_STANDARD);
    assert_relative_eq!(decomp.reliability, 0.0025, epsilon = TOLERANCE_STANDARD);
    assert_relative_eq!(decomp.resolution, 0.0625, epsilon = TOLERANCE_STANDARD);
    assert_relative_eq!(decomp.uncertainty, 0.25, epsilon = TOLERANCE_STANDARD);
    assert!(decomp.identity_residual() < TOLERANCE_STANDARD);

    // Continuous distinct probabilities also satisfy the Murphy identity exactly and binned.
    let cont_probs: Vec<UnitInterval> = (0..40).map(|i| ui(((i as f64) + 0.5) / 40.0)).collect();
    let cont_outcomes: Vec<bool> = (0..40).map(|i| i % 3 == 0 || i > 24).collect();
    let cont_exact = murphy_decomposition(&cont_probs, &cont_outcomes).expect("cont exact");
    assert!(cont_exact.identity_residual() < TOLERANCE_STANDARD);

    let cont_binned =
        murphy_decomposition_binned(&cont_probs, &cont_outcomes, 5, BinningStrategy::Uniform)
            .expect("cont binned");
    assert!(cont_binned.identity_residual() < TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_binary_and_multiclass_log_loss_reference_and_clipping() {
    let probs = [ui(0.0), ui(1.0), ui(0.8), ui(0.25)];
    let outcomes = [false, true, true, false];
    let nll = binary_log_loss(&probs, &outcomes, DEFAULT_LOG_LOSS_EPSILON).expect("binary nll");
    let expected =
        -(2.0 * (1.0 - DEFAULT_LOG_LOSS_EPSILON).ln() + 0.8_f64.ln() + 0.75_f64.ln()) / 4.0;
    assert_relative_eq!(nll, expected, epsilon = TOLERANCE_STANDARD);

    // Worst-case boundary error remains finite due to symmetric [eps, 1 - eps] clipping.
    let wrong_boundary =
        binary_log_loss(&[ui(1.0), ui(0.0)], &[false, true], 1e-12).expect("clipped boundary");
    assert!(wrong_boundary.is_finite());
    assert_relative_eq!(wrong_boundary, -(1e-12_f64.ln()), epsilon = 1e-7);

    // Multiclass reference value across 3 classes and 2 samples.
    let mc_probs = [ui(0.7), ui(0.2), ui(0.1), ui(0.1), ui(0.3), ui(0.6)];
    let mc_targets = [0_usize, 2_usize];
    let mc_loss = multiclass_log_loss(&mc_probs, 3, &mc_targets, DEFAULT_LOG_LOSS_EPSILON)
        .expect("multiclass");
    let mc_expected = -0.5 * (0.7_f64.ln() + 0.6_f64.ln());
    assert_relative_eq!(mc_loss, mc_expected, epsilon = TOLERANCE_STANDARD);

    assert!(multiclass_log_loss(&mc_probs, 3, &[0, 3], DEFAULT_LOG_LOSS_EPSILON).is_err());
    let unnormalized = [ui(0.5), ui(0.2), ui(0.1)];
    assert!(multiclass_log_loss(&unnormalized, 3, &[0], DEFAULT_LOG_LOSS_EPSILON).is_err());
}

#[test]
#[verified_engine::verified]
fn test_gaussian_and_ensemble_crps_analytical_values() {
    let sigma = PositiveFloat::new(2.0).expect("sigma");
    // At y = mu (z = 0): CRPS(N(mu, sigma^2), mu) = sigma * (sqrt(2) - 1) / sqrt(pi).
    let at_mean = gaussian_crps(5.0, sigma, 5.0).expect("gaussian crps");
    let expected_zero = 2.0 * (2.0_f64.sqrt() - 1.0) / PI.sqrt();
    assert_relative_eq!(at_mean, expected_zero, epsilon = TOLERANCE_STANDARD);

    let batch = dataset_gaussian_crps(&[5.0, 5.0], &[sigma, sigma], &[5.0, 5.0]).expect("batch");
    assert_relative_eq!(batch, expected_zero, epsilon = TOLERANCE_STANDARD);

    // Ensemble M = 2 at [0.0, 2.0], y = 1.0 -> exact integral = 0.5.
    let ens_crps = ensemble_crps_sorted(&[0.0, 2.0], 1.0).expect("ensemble crps");
    assert_relative_eq!(ens_crps, 0.5, epsilon = TOLERANCE_STANDARD);

    // Shift invariance under large 1e8 offset.
    let shifted = ensemble_crps_sorted(&[1.0e8, 1.0e8 + 2.0], 1.0e8 + 1.0).expect("shifted");
    assert_relative_eq!(shifted, 0.5, epsilon = TOLERANCE_STANDARD);

    // Unsorted slice must return Err.
    assert!(ensemble_crps_sorted(&[2.0, 1.0], 1.5).is_err());
}

#[test]
#[verified_engine::verified]
fn test_ece_mce_uniform_and_quantile_reliability() {
    // Bin 0 [0.0, 0.5): probs [0.1, 0.3] => mean_conf = 0.2, outcomes [false, true] => acc = 0.5, gap = 0.3
    // Bin 1 [0.5, 1.0]: probs [0.8, 0.9] => mean_conf = 0.85, outcomes [true, true] => acc = 1.0, gap = 0.15
    // ECE = 0.5 * 0.3 + 0.5 * 0.15 = 0.225, MCE = 0.3.
    let probs = [ui(0.1), ui(0.3), ui(0.8), ui(0.9)];
    let outcomes = [false, true, true, true];

    let ece = expected_calibration_error(&probs, &outcomes, 2, BinningStrategy::Uniform)
        .expect("uniform ece");
    let mce = maximum_calibration_error(&probs, &outcomes, 2, BinningStrategy::Uniform)
        .expect("uniform mce");
    assert_relative_eq!(ece, 0.225, epsilon = TOLERANCE_STANDARD);
    assert_relative_eq!(mce, 0.30, epsilon = TOLERANCE_STANDARD);

    let q_bins =
        reliability_curve(&probs, &outcomes, 2, BinningStrategy::Quantile).expect("quantile curve");
    assert_eq!(q_bins.len(), 2);
    assert_eq!(q_bins[0].count, 2);
    assert_eq!(q_bins[1].count, 2);
}

#[test]
#[verified_engine::verified]
fn test_roc_and_pr_auc_reference_and_ties() {
    // Perfect ranking -> ROC-AUC = 1.0, PR-AUC = 1.0.
    let perfect_scores = [0.1, 0.2, 0.8, 0.9];
    let outcomes = [false, false, true, true];
    assert_relative_eq!(
        roc_auc(&perfect_scores, &outcomes).expect("roc"),
        1.0,
        epsilon = TOLERANCE_STANDARD
    );
    assert_relative_eq!(
        pr_auc(&perfect_scores, &outcomes, PrIntegration::Step).expect("pr step"),
        1.0,
        epsilon = TOLERANCE_STANDARD
    );
    assert_relative_eq!(
        pr_auc(&perfect_scores, &outcomes, PrIntegration::Trapezoidal).expect("pr trap"),
        1.0,
        epsilon = TOLERANCE_STANDARD
    );

    // Tied scores: [0.9 (pos), 0.5 (pos), 0.5 (neg), 0.1 (neg)]
    // Mann-Whitney U: (1 + 1) + (0.5 + 1) = 3.5 out of 4 pairs => ROC-AUC = 0.875.
    let tied_scores = [0.9, 0.5, 0.5, 0.1];
    let tied_outcomes = [true, true, false, false];
    assert_relative_eq!(
        roc_auc(&tied_scores, &tied_outcomes).expect("tied roc"),
        0.875,
        epsilon = TOLERANCE_STANDARD
    );
    // PR points: (0, 1.0) -> score 0.9: tp=1, fp=0 => (0.5, 1.0) -> score 0.5: tp=2, fp=1 => (1.0, 2/3) -> score 0.1: (1.0, 0.5)
    // Step PR-AUC = 0.5 * 1.0 + 0.5 * (2/3) = 5/6.
    assert_relative_eq!(
        pr_auc(&tied_scores, &tied_outcomes, PrIntegration::Step).expect("tied pr step"),
        5.0 / 6.0,
        epsilon = TOLERANCE_STANDARD
    );

    // Degenerate single-class outcomes return Err.
    assert!(roc_auc(&[0.2, 0.8], &[true, true]).is_err());
    assert!(pr_auc(&[0.2, 0.8], &[false, false], PrIntegration::Step).is_err());
}

fn sigmoid_ui(z: f64) -> UnitInterval {
    let p = 1.0 / (1.0 + (-z).exp());
    ui(p.clamp(0.0, 1.0))
}

#[test]
#[verified_engine::verified]
fn test_platt_temperature_and_pava_lower_held_out_ece_and_log_loss() {
    // Generate deterministic miscalibrated calibration + held-out benchmark:
    // True probability p*(x) = sigmoid(z_true), but model emits overconfident shifted logits 3.2 * z_true + 1.1.
    let mut cal_logits = Vec::with_capacity(120);
    let mut cal_outcomes = Vec::with_capacity(120);
    let mut test_logits = Vec::with_capacity(120);
    let mut test_outcomes = Vec::with_capacity(120);

    for i in 0..120 {
        let z_true = -2.2 + 4.4 * ((i as f64) / 119.0);
        let p_true = 1.0 / (1.0 + (-z_true).exp());
        // Deterministic quasi-random uniform threshold via golden-ratio Weyl sequence:
        let u_cal = (((i as f64 + 1.0) * 0.618_033_988_749_895) % 1.0).abs();
        let u_test = (((i as f64 + 37.0) * 0.618_033_988_749_895) % 1.0).abs();
        cal_logits.push(3.2 * z_true + 1.1);
        cal_outcomes.push(u_cal < p_true);
        test_logits.push(3.2 * z_true + 1.1);
        test_outcomes.push(u_test < p_true);
    }

    let raw_probs: Vec<UnitInterval> = test_logits.iter().map(|&z| sigmoid_ui(z)).collect();
    let raw_nll = binary_log_loss(&raw_probs, &test_outcomes, DEFAULT_LOG_LOSS_EPSILON).unwrap();
    let raw_ece =
        expected_calibration_error(&raw_probs, &test_outcomes, 10, BinningStrategy::Uniform)
            .unwrap();

    // 1. Platt scaling
    let platt = PlattScaler::fit(&cal_logits, &cal_outcomes).expect("platt fit");
    let mut platt_probs = vec![ui(0.5); test_logits.len()];
    platt.predict_into(&test_logits, &mut platt_probs).unwrap();
    let platt_nll =
        binary_log_loss(&platt_probs, &test_outcomes, DEFAULT_LOG_LOSS_EPSILON).unwrap();
    let platt_ece =
        expected_calibration_error(&platt_probs, &test_outcomes, 10, BinningStrategy::Uniform)
            .unwrap();
    assert!(
        platt_nll < raw_nll,
        "Platt NLL {platt_nll} >= raw {raw_nll}"
    );
    assert!(
        platt_ece < raw_ece,
        "Platt ECE {platt_ece} >= raw {raw_ece}"
    );

    // 2. Learned Temperature scaling
    let temp = TemperatureScaler::fit_binary(&cal_logits, &cal_outcomes).expect("temp fit");
    assert!(temp.temperature.value() > 1.5);
    let temp_probs: Vec<UnitInterval> = test_logits
        .iter()
        .map(|&z| temp.predict_binary(z).unwrap())
        .collect();
    let temp_nll = binary_log_loss(&temp_probs, &test_outcomes, DEFAULT_LOG_LOSS_EPSILON).unwrap();
    let temp_ece =
        expected_calibration_error(&temp_probs, &test_outcomes, 10, BinningStrategy::Uniform)
            .unwrap();
    assert!(temp_nll < raw_nll, "Temp NLL {temp_nll} >= raw {raw_nll}");
    assert!(temp_ece < raw_ece, "Temp ECE {temp_ece} >= raw {raw_ece}");

    // 3. PAVA Isotonic regression
    let iso = IsotonicCalibrator::fit(&cal_logits, &cal_outcomes).expect("isotonic fit");
    let mut iso_probs = vec![ui(0.5); test_logits.len()];
    iso.predict_into(&test_logits, &mut iso_probs).unwrap();
    let iso_nll = binary_log_loss(&iso_probs, &test_outcomes, DEFAULT_LOG_LOSS_EPSILON).unwrap();
    let iso_ece =
        expected_calibration_error(&iso_probs, &test_outcomes, 10, BinningStrategy::Uniform)
            .unwrap();
    assert!(iso_nll < raw_nll, "Iso NLL {iso_nll} >= raw {raw_nll}");
    assert!(iso_ece < raw_ece, "Iso ECE {iso_ece} >= raw {raw_ece}");
}

#[test]
#[verified_engine::verified]
fn test_pava_piecewise_linear_interpolation_and_invalid_inputs() {
    let scores = [0.0, 1.0, 2.0, 3.0];
    let targets = [0.0, 0.8, 0.2, 1.0];
    let mut blocks = [PavaBlock::default(); 4];
    let m = fit_pava_sorted_into(&scores, &targets, None, &mut blocks).expect("pava fit");
    // Blocks: [0.0..0.0 -> 0.0], [1.0..2.0 -> 0.5], [3.0..3.0 -> 1.0]
    assert_eq!(m, 3);
    assert_relative_eq!(blocks[1].value, 0.5, epsilon = TOLERANCE_STANDARD);

    // Plateau at x = 1.5 -> 0.5; midpoint interpolation at x = 0.5 -> 0.25 and x = 2.5 -> 0.75.
    assert_relative_eq!(
        predict_pava_blocks(&blocks[..m], 1.5).unwrap().value(),
        0.5,
        epsilon = TOLERANCE_STANDARD
    );
    assert_relative_eq!(
        predict_pava_blocks(&blocks[..m], 0.5).unwrap().value(),
        0.25,
        epsilon = TOLERANCE_STANDARD
    );
    assert_relative_eq!(
        predict_pava_blocks(&blocks[..m], 2.5).unwrap().value(),
        0.75,
        epsilon = TOLERANCE_STANDARD
    );

    // Reject empty, mismatched, or NaN inputs cleanly.
    assert!(brier_score(&[], &[]).is_err());
    assert!(brier_score(&[ui(0.5)], &[true, false]).is_err());
    assert!(PlattScaler::fit(&[f64::NAN, 1.0], &[false, true]).is_err());
    assert!(TemperatureScaler::fit_binary(&[0.5, f64::NAN], &[false, true]).is_err());
}
