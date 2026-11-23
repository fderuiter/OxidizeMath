//! Unit and zero-allocation tests for [`super::MarkowitzFrontierTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_markowitz_default_frontier_and_tangency_invariants() {
    let tool = MarkowitzFrontierTool::default();
    assert!(tool.last_error.is_none());
    let gmv = tool.con_gmv.expect("constrained GMV solved");
    let tan = tool.con_tangency.expect("constrained tangency solved");
    let unc = tool.unc_tangency.expect("unconstrained tangency solved");

    assert!((gmv.weights_slice().iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!((tan.weights_slice().iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(tan.weights_slice().iter().all(|&w| w >= -1e-9));
    assert!(tan.sharpe_ratio + 1e-9 >= gmv.sharpe_ratio);
    assert!(unc.sharpe_ratio + 1e-9 >= tan.sharpe_ratio);
    assert_eq!(tool.con_frontier_pts.len(), FRONTIER_GRID_POINTS);
    assert!(!tool.export_datasets().is_empty());
    assert!(!tool.export_plot_primitives().is_empty());
}

#[test]
fn test_markowitz_evaluate_in_place_zero_allocations_up_to_16_assets() {
    let mut tool = MarkowitzFrontierTool::default();
    while tool.expected_returns.len() < MAX_PORTFOLIO_ASSETS {
        tool.append_asset();
    }
    assert_eq!(tool.expected_returns.len(), MAX_PORTFOLIO_ASSETS);
    tool.evaluate_in_place();
    assert!(tool.last_error.is_none());

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.evaluate_in_place();
    });
}

#[test]
fn test_markowitz_state_roundtrip_and_strict_rejection() {
    let mut tool = MarkowitzFrontierTool::default();
    let saved = tool.save_state().expect("save_state");
    assert!(tool.try_load_state(&saved).is_ok());

    let bad_unknown = r#"{"risk_free_rate":0.03,"max_asset_weight":1.0,"asset_names":["A","B"],"expected_returns":[0.1,0.05],"volatilities":[0.2,0.1],"upper_correlations":[0.1],"extra":1}"#;
    assert!(tool.try_load_state(bad_unknown).is_err());

    let bad_corr = r#"{"risk_free_rate":0.03,"max_asset_weight":1.0,"asset_names":["A","B"],"expected_returns":[0.1,0.05],"volatilities":[0.2,0.1],"upper_correlations":[1.2]}"#;
    assert!(tool.try_load_state(bad_corr).is_err());
}

// [cite:clinical_trials_statistics]
