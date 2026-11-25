//! Zero-allocation integration tests for `pure_math::statistics::risk` under `VerifiedAllocator`.

use pure_math::error::RiskError;
use pure_math::pure_math::statistics::risk::{
    calmar_ratio, calmar_ratio_wealth, cornish_fisher_cvar, cornish_fisher_var,
    drawdown_series_from_returns, drawdown_series_from_wealth, gaussian_cvar, gaussian_var,
    historical_cvar, historical_var, max_drawdown_returns, max_drawdown_wealth,
    max_underwater_duration_returns, max_underwater_duration_wealth, sharpe_ratio, sortino_ratio,
    summarize_return_risk,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_risk_metrics_zero_heap_allocations() {
    let returns = [
        -0.065, 0.012, -0.024, 0.018, 0.009, -0.041, 0.022, 0.015, -0.011, 0.007, -0.032, 0.019,
        0.014, -0.008, 0.021, -0.052, 0.016, 0.011, -0.005, 0.024,
    ];
    let wealth = [
        100.0, 93.5, 94.6, 92.3, 94.0, 94.8, 90.9, 92.9, 94.3, 93.3, 94.0, 91.0, 92.7, 94.0, 93.2,
        95.2, 90.2, 91.6, 92.6, 92.1,
    ];
    let mut scratch = [0.0_f64; 20];
    let mut dd_out = [0.0_f64; 20];

    // Warm up statrs Normal inverse CDF if any lazy static exists, then lock VerifiedAllocator.
    let _ = gaussian_var(&returns, 0.95);

    let (summary, _) = VerifiedEngine::run_verified(|| {
        let h_var = historical_var(&returns, 0.95, &mut scratch).expect("h_var");
        let h_cvar = historical_cvar(&returns, 0.95, &mut scratch).expect("h_cvar");
        let g_var = gaussian_var(&returns, 0.95).expect("g_var");
        let g_cvar = gaussian_cvar(&returns, 0.95).expect("g_cvar");
        let cf_var = cornish_fisher_var(&returns, 0.99).expect("cf_var");
        let cf_cvar = cornish_fisher_cvar(&returns, 0.99).expect("cf_cvar");
        let dd_r = drawdown_series_from_returns(&returns, &mut dd_out).expect("dd_r");
        let dd_w = drawdown_series_from_wealth(&wealth, &mut dd_out).expect("dd_w");
        let mdd_r = max_drawdown_returns(&returns).expect("mdd_r");
        let mdd_w = max_drawdown_wealth(&wealth).expect("mdd_w");
        let uw_r = max_underwater_duration_returns(&returns).expect("uw_r");
        let uw_w = max_underwater_duration_wealth(&wealth).expect("uw_w");
        let sh = sharpe_ratio(&returns, 0.0002, 252.0).expect("sharpe");
        let so = sortino_ratio(&returns, 0.0002, 252.0).expect("sortino");
        let ca = calmar_ratio(&returns, 252.0).expect("calmar");
        let ca_w = calmar_ratio_wealth(&wealth, 252.0).expect("calmar_w");

        // Error path must also remain zero-allocation
        let bad = historical_var(&returns, 0.45, &mut scratch);
        assert_eq!(bad, Err(RiskError::InvalidConfidence { value: 0.45 }));

        assert!(h_cvar >= h_var && g_cvar > g_var && cf_cvar >= cf_var);
        assert!((dd_r - mdd_r).abs() < 1e-12 && (dd_w - mdd_w).abs() < 1e-12);
        assert!(uw_r > 0 && uw_w > 0);
        assert!(sh.is_finite() && so.is_finite() && ca.is_finite() && ca_w.is_finite());

        summarize_return_risk(&returns, 0.95, 0.0002, 252.0, &mut scratch).expect("summary")
    });

    assert!(summary.historical_cvar >= summary.historical_var);
}

// [cite:clinical_trials_statistics]
