//! Unit and zero-allocation tests for [`super::TailRiskDrawdownTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_tail_risk_default_and_cvar_dominance() {
    let tool = TailRiskDrawdownTool::default();
    let sum = tool.summary.expect("default summary populated");
    assert!(sum.historical_cvar + 1e-12 >= sum.historical_var);
    assert!(sum.gaussian_cvar > sum.gaussian_var);
    assert!(sum.cornish_fisher_cvar + 1e-12 >= sum.cornish_fisher_var);
    assert!(sum.max_drawdown >= 0.0 && sum.max_drawdown < 1.0);
    assert_eq!(tool.hist_cvar_curve.len(), CONFIDENCE_GRID_POINTS);
    assert_eq!(tool.wealth_points.len(), tool.horizon_periods + 1);
    assert!(!tool.export_datasets().is_empty());
    assert!(!tool.export_plot_primitives().is_empty());
}

#[test]
fn test_tail_risk_evaluate_in_place_zero_allocations() {
    let mut tool = TailRiskDrawdownTool::default();
    tool.evaluate_in_place();
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.evaluate_in_place();
    });
}

#[test]
fn test_tail_risk_state_roundtrip_and_strict_rejection() {
    let mut tool = TailRiskDrawdownTool::default();
    let saved = tool.save_state().expect("save_state");
    assert!(tool.try_load_state(&saved).is_ok());

    let bad_unknown = r#"{"annual_volatility":0.2,"annual_drift":0.1,"tail_shock_prob":0.05,"tail_shock_scale":0.08,"confidence_level":0.95,"risk_free_rate":0.03,"horizon_periods":252,"seed":1,"extra":true}"#;
    assert!(tool.try_load_state(bad_unknown).is_err());

    let bad_alpha = r#"{"annual_volatility":0.2,"annual_drift":0.1,"tail_shock_prob":0.05,"tail_shock_scale":0.08,"confidence_level":0.40,"risk_free_rate":0.03,"horizon_periods":252,"seed":1}"#;
    assert!(tool.try_load_state(bad_alpha).is_err());
}

// [cite:clinical_trials_statistics]
