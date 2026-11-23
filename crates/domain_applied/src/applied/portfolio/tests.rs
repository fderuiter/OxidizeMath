#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use approx::assert_relative_eq;

#[test]
fn test_interior_parity_between_unconstrained_and_long_only_solvers() {
    // Diagonal covariance where both GMV and tangency have strictly positive weights.
    let returns = [0.12, 0.09, 0.06];
    let vols = [0.20, 0.15, 0.10];
    let upper_corrs = [0.10, 0.05, 0.15];
    let spec = PortfolioSpec::from_upper_correlations(&returns, &vols, &upper_corrs).unwrap();

    let unc_gmv = unconstrained_min_variance(&spec, 0.02).unwrap();
    let lo_gmv = long_only_min_variance(&spec, 0.02).unwrap();
    for i in 0..3 {
        assert!(unc_gmv.weights[i] > 0.0);
        assert_relative_eq!(lo_gmv.weights[i], unc_gmv.weights[i], epsilon = 1e-9);
    }

    let unc_tan = unconstrained_tangency(&spec, 0.02).unwrap();
    let lo_tan = long_only_tangency(&spec, 0.02).unwrap();
    for i in 0..3 {
        assert!(unc_tan.weights[i] > 0.0);
        assert_relative_eq!(lo_tan.weights[i], unc_tan.weights[i], epsilon = 1e-6);
    }
    assert_relative_eq!(lo_tan.sharpe_ratio, unc_tan.sharpe_ratio, epsilon = 1e-8);
}

#[test]
fn test_active_zero_weight_clamping_when_unconstrained_shorts_dominated_asset() {
    // Asset 2 has low return (3%) and high positive correlation (0.80) with Asset 0 (14%),
    // so unconstrained tangency shorts Asset 2 while long-only clamps w_2 = 0.
    let returns = [0.14, 0.09, 0.03];
    let vols = [0.18, 0.14, 0.22];
    let upper_corrs = [0.20, 0.80, 0.25];
    let spec = PortfolioSpec::from_upper_correlations(&returns, &vols, &upper_corrs).unwrap();

    let unc_tan = unconstrained_tangency(&spec, 0.025).unwrap();
    assert!(
        unc_tan.weights[2] < -0.05,
        "Expected unconstrained tangency to short dominated asset 2, got {}",
        unc_tan.weights[2]
    );

    let lo_tan = long_only_tangency(&spec, 0.025).unwrap();
    assert_relative_eq!(
        lo_tan.weights_slice().iter().sum::<f64>(),
        1.0,
        epsilon = 1e-10
    );
    for &w in lo_tan.weights_slice() {
        assert!(w >= -1e-12);
    }
    assert_relative_eq!(lo_tan.weights[2], 0.0, epsilon = 1e-9);
    assert!(unc_tan.sharpe_ratio > lo_tan.sharpe_ratio);
}

#[test]
fn test_frontiers_monotone_and_sixteen_asset_upper_bound_support() {
    let n = MAX_PORTFOLIO_ASSETS;
    let mut returns = [0.0_f64; MAX_PORTFOLIO_ASSETS];
    let mut cov = [0.0_f64; MAX_PORTFOLIO_ASSETS * MAX_PORTFOLIO_ASSETS];
    let mut caps = [0.25_f64; MAX_PORTFOLIO_ASSETS];
    for i in 0..n {
        returns[i] = 0.04 + 0.008 * (i as f64);
        caps[i] = 0.20;
        for j in 0..n {
            let rho = if i == j { 1.0 } else { 0.25 };
            let vi = 0.08 + 0.01 * (i as f64);
            let vj = 0.08 + 0.01 * (j as f64);
            cov[math_commons::math_kernel::types::flatten_2d_index(j, i, n)] = rho * vi * vj;
        }
    }
    let spec = PortfolioSpec::from_covariance(&returns, &cov).unwrap();
    let mut lo_pts = [PortfolioAllocation {
        asset_count: n,
        weights: [0.0; MAX_PORTFOLIO_ASSETS],
        expected_return: 0.0,
        variance: 0.0,
        volatility: 0.0,
        sharpe_ratio: 0.0,
    }; 12];
    let written = long_only_frontier_in_place(&spec, 0.02, &mut lo_pts).unwrap();
    assert_eq!(written, 12);
    for k in 1..written {
        assert!(lo_pts[k].expected_return + 1e-10 >= lo_pts[k - 1].expected_return);
        assert!(lo_pts[k].volatility + 1e-9 >= lo_pts[k - 1].volatility);
    }

    let capped_tan = constrained_tangency(&spec, 0.02, &caps).unwrap();
    assert_relative_eq!(
        capped_tan.weights_slice().iter().sum::<f64>(),
        1.0,
        epsilon = 1e-10
    );
    for &w in capped_tan.weights_slice() {
        assert!((-1e-12..=0.20 + 1e-10).contains(&w));
    }
}

#[test]
fn test_portfolio_error_validation() {
    assert_eq!(
        PortfolioSpec::from_covariance(&[0.05], &[0.04]),
        Err(PortfolioError::TooFewAssets { count: 1 })
    );
    assert_eq!(
        PortfolioSpec::from_covariance(&[0.05, 0.08], &[0.04, 0.01, 0.02, 0.09]),
        Err(PortfolioError::NotSymmetric)
    );
    assert_eq!(
        PortfolioSpec::from_covariance(&[0.05, 0.08], &[0.04, 0.06, 0.06, 0.09]),
        Err(PortfolioError::NotPositiveDefinite)
    );
    let spec = PortfolioSpec::from_volatilities_and_correlations(
        &[0.05, 0.10],
        &[0.10, 0.20],
        &[1.0, 0.2, 0.2, 1.0],
    )
    .unwrap();
    assert!(matches!(
        long_only_target_return(&spec, 0.25, 0.02),
        Err(PortfolioError::InfeasibleTargetReturn { .. })
    ));
    assert_eq!(
        long_only_tangency(&spec, 0.15),
        Err(PortfolioError::NoExcessReturnTangency)
    );
}

// [cite:clinical_trials_statistics]
