//! Unit, zero-allocation, and inverse challenge tests for `WinRatioParadoxLab`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::challenge::CHALLENGE_WIN_RATIO_REVERSAL;
use super::model::evaluate_regime_grid_zero_alloc;
use super::{WinRatioParadoxConfig, WinRatioParadoxLab, STARTER_PRESET_JSON};
use crate::framework::InteractiveTool;
use crate::starter_gallery::find_challenge_metadata;

fn assert_close(actual: f64, expected: f64, eps: f64) {
    assert!(
        (actual - expected).abs() <= eps,
        "expected {expected}, got {actual} (eps = {eps})"
    );
}

#[test]
fn test_starter_preset_reproduces_severity_reversal_acceptance_metrics() {
    let mut lab = WinRatioParadoxLab::default();
    lab.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset must load");

    let m = &lab.metrics;
    assert!(m.is_severity_reversal);
    assert_eq!(m.wins, 240);
    assert_eq!(m.losses, 144);
    assert_eq!(m.ties, 16);
    assert_close(m.pooled_hazard_ratio, 3.8623, 1e-4);
    assert_close(m.win_ratio, 1.6667, 1e-4);
    assert_close(m.sample_win_ratio, m.win_ratio, 1e-12);
    assert_close(m.win_ratio_p_value, 4.20e-7, 1e-9);
    assert_close(m.t_statistic, 5.0746, 1e-4);
}

#[test]
fn test_regime_grid_zero_alloc_and_seeded_resample_determinism() {
    let mut lab_a = WinRatioParadoxLab::default();
    let mut lab_b = WinRatioParadoxLab::default();

    let ((), _cert) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        evaluate_regime_grid_zero_alloc(&lab_a.config, &mut lab_a.scratch);
    });

    lab_a.resample_seeded_cohort();
    lab_b.resample_seeded_cohort();
    assert_eq!(lab_a.metrics, lab_b.metrics);

    lab_a.config.matched_pairing = true;
    lab_a.recalculate();
    assert!(lab_a.metrics.wins + lab_a.metrics.losses <= 20);
}

#[test]
fn test_win_ratio_challenge_tamper_clamping_and_solve() {
    let meta = find_challenge_metadata(CHALLENGE_WIN_RATIO_REVERSAL)
        .expect("win_ratio_severity_reversal registered");
    let baseline = meta.generate_baseline_state();

    let mut lab = WinRatioParadoxLab::default();
    lab.apply_challenge(Some(CHALLENGE_WIN_RATIO_REVERSAL))
        .unwrap();
    lab.try_load_state(&baseline).unwrap();
    let base_eval = lab.evaluate_challenge().unwrap();
    assert!(!base_eval.passed);

    let tampered = serde_json::json!({
        "sample_size_per_arm": 38,
        "horizon_months": 55.0,
        "control_mortality_rate": 0.80,
        "mortality_reduction": 0.8333333333333334,
        "early_hosp_rate": 0.90,
        "early_hosp_mean_months": 4.487285,
        "matched_pairing": true,
        "seed_counter": 99
    })
    .to_string();
    let clamped_str = meta.clamp_state_json(&tampered).unwrap();
    let clamped: WinRatioParadoxConfig = serde_json::from_str(&clamped_str).unwrap();
    assert_eq!(clamped.sample_size_per_arm, 20);
    assert_eq!(clamped.horizon_months, 36.0);
    assert_eq!(clamped.control_mortality_rate, 0.60);
    assert!(!clamped.matched_pairing);
    assert_eq!(clamped.seed_counter, 0);

    lab.try_load_state(&clamped_str).unwrap();
    let solved = lab.evaluate_challenge().unwrap();
    assert!(solved.passed);
    assert_close(solved.score, 100.0, 1e-9);
}

// [cite:win_ratio]
