//! Zero-allocation integration tests for `domain_applied::applied::portfolio` under `VerifiedAllocator`.

use domain_applied::applied::portfolio::{
    MAX_PORTFOLIO_ASSETS, PortfolioAllocation, PortfolioError, PortfolioSpec,
    constrained_frontier_in_place, constrained_tangency, long_only_frontier_in_place,
    long_only_min_variance, long_only_tangency, long_only_target_return,
    unconstrained_frontier_in_place, unconstrained_min_variance, unconstrained_tangency,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_markowitz_solvers_zero_heap_allocations() {
    let returns = [0.14, 0.11, 0.085, 0.065, 0.05, 0.038];
    let vols = [0.24, 0.18, 0.14, 0.10, 0.07, 0.045];
    let upper_corrs = [
        0.45, 0.30, 0.15, -0.10, 0.05, 0.35, 0.20, -0.05, 0.10, 0.25, 0.05, 0.12, 0.18, 0.22, 0.30,
    ];
    let caps = [0.35_f64; 6];

    let empty_pt = PortfolioAllocation {
        asset_count: 6,
        weights: [0.0; MAX_PORTFOLIO_ASSETS],
        expected_return: 0.0,
        variance: 0.0,
        volatility: 0.0,
        sharpe_ratio: 0.0,
    };
    let mut unc_frontier = [empty_pt; 16];
    let mut lo_frontier = [empty_pt; 16];
    let mut cap_frontier = [empty_pt; 16];

    let ((lo_tan, cap_tan), _) = VerifiedEngine::run_verified(|| {
        let spec =
            PortfolioSpec::from_upper_correlations(&returns, &vols, &upper_corrs).expect("spec");
        let unc_gmv = unconstrained_min_variance(&spec, 0.025).expect("unc_gmv");
        let unc_tan = unconstrained_tangency(&spec, 0.025).expect("unc_tan");
        let lo_gmv = long_only_min_variance(&spec, 0.025).expect("lo_gmv");
        let lo_mid = long_only_target_return(&spec, 0.09, 0.025).expect("lo_mid");
        let lo_tan = long_only_tangency(&spec, 0.025).expect("lo_tan");
        let cap_tan = constrained_tangency(&spec, 0.025, &caps).expect("cap_tan");

        let n_unc =
            unconstrained_frontier_in_place(&spec, 0.04, 0.14, 0.025, &mut unc_frontier).unwrap();
        let n_lo = long_only_frontier_in_place(&spec, 0.025, &mut lo_frontier).unwrap();
        let n_cap = constrained_frontier_in_place(&spec, 0.025, &caps, &mut cap_frontier).unwrap();

        let infeasible = long_only_target_return(&spec, 0.30, 0.025);
        assert!(matches!(
            infeasible,
            Err(PortfolioError::InfeasibleTargetReturn { .. })
        ));

        assert_eq!(n_unc, 16);
        assert_eq!(n_lo, 16);
        assert_eq!(n_cap, 16);
        assert!(lo_gmv.volatility + 1e-10 >= unc_gmv.volatility);
        assert!((lo_mid.expected_return - 0.09).abs() < 1e-8);
        assert!(unc_tan.sharpe_ratio + 1e-9 >= lo_tan.sharpe_ratio);
        (lo_tan, cap_tan)
    });

    assert!(lo_tan.sharpe_ratio >= cap_tan.sharpe_ratio - 1e-8);
}

// [cite:clinical_trials_statistics]
