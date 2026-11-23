//! Unit, zero-allocation, and inverse challenge tests for `CopulaTailRiskLab`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::challenge::CHALLENGE_COPULA_TAIL_KELLY;
use super::{CopulaTailRiskConfig, CopulaTailRiskLab, STARTER_PRESET_JSON};
use crate::framework::InteractiveTool;
use crate::starter_gallery::find_challenge_metadata;

fn assert_close(actual: f64, expected: f64, eps: f64) {
    assert!(
        (actual - expected).abs() <= eps,
        "expected {expected}, got {actual} (eps = {eps})"
    );
}

#[test]
fn test_starter_preset_reproduces_sgp_probabilities_and_583x_tail_multiplier() {
    let mut lab = CopulaTailRiskLab::default();
    lab.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset must load");

    let m = &lab.metrics;
    assert_close(m.naive_independent_prob, 0.04, 1e-12);
    // 0.84% at rho = -0.50 and 11.29% at rho = +0.70
    assert_close(m.sgp_neg50_prob * 100.0, 0.8438, 1e-2);
    assert_close(m.sgp_pos70_prob * 100.0, 11.2902, 1e-2);
    // 5.83x Student-t vs Gaussian upper-tail conditional crash ratio at u = 0.999
    assert_close(m.tail_multiplier, 5.8254, 1e-2);

    let (tail_cmp, _cert) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        lab.scratch
            .populate_tail_regime_grid(&lab.rho_axis, &lab.quantile_axis, 3.0);
        lab.scratch
            .evaluate_tail_comparison(0.20, 0.20, 0.999, 0.50, 3.0)
    });
    assert_close(tail_cmp.tail_multiplier, m.tail_multiplier, 1e-12);
}

#[test]
fn test_copula_challenge_tamper_clamping_and_solve() {
    let meta = find_challenge_metadata(CHALLENGE_COPULA_TAIL_KELLY)
        .expect("copula_heavy_tail_kelly registered");
    let baseline = meta.generate_baseline_state();

    let mut lab = CopulaTailRiskLab::default();
    lab.apply_challenge(Some(CHALLENGE_COPULA_TAIL_KELLY))
        .unwrap();
    lab.try_load_state(&baseline).unwrap();
    let base_eval = lab.evaluate_challenge().unwrap();
    assert!(!base_eval.passed);

    let tampered = serde_json::json!({
        "leg1_prob": 0.80,
        "leg2_prob": 0.80,
        "correlation_rho": 0.60,
        "tail_quantile_u": 0.90,
        "degrees_of_freedom": 25.0,
        "offered_decimal_odds": 90.0,
        "stake_multiplier": 0.65
    })
    .to_string();
    let clamped_str = meta.clamp_state_json(&tampered).unwrap();
    let clamped: CopulaTailRiskConfig = serde_json::from_str(&clamped_str).unwrap();
    assert_eq!(clamped.leg1_prob, 0.20);
    assert_eq!(clamped.leg2_prob, 0.20);
    assert_eq!(clamped.tail_quantile_u, 0.999);
    assert_eq!(clamped.degrees_of_freedom, 3.0);
    assert_eq!(clamped.offered_decimal_odds, 18.0);
    assert_close(clamped.correlation_rho, 0.60, 1e-12);
    assert_close(clamped.stake_multiplier, 0.65, 1e-12);

    lab.try_load_state(&clamped_str).unwrap();
    let solved = lab.evaluate_challenge().unwrap();
    assert!(solved.passed);
    assert_close(solved.score, 100.0, 1e-9);
}

// [cite:clinical_trials_statistics]
