use super::portfolio::{
    PortfolioKellyWorkspace, evaluate_portfolio_kelly_metrics, independent_kelly_weights,
    long_only_kelly_projected_gradient, solve_long_only_kelly_into, solve_unconstrained_kelly_into,
    unconstrained_covariance_kelly,
};
use super::tail_risk::{
    PortfolioHorizonSpec, empirical_tail_risk, empirical_var_cvar_in_place,
    parametric_lognormal_tail_risk, simulate_portfolio_tail_risk_into,
    simulated_portfolio_tail_risk,
};
use crate::error::KellyError;
use nalgebra::DMatrix;
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

fn three_asset_rho_060_benchmark() -> ([f64; 3], DMatrix<f64>, f64) {
    let mu = [0.12, 0.12, 0.12];
    let sigma = 0.20;
    let rho = 0.60;
    let var = sigma * sigma;
    let cov_off = var * rho;
    let cov = DMatrix::from_row_slice(
        3,
        3,
        &[
            var, cov_off, cov_off, cov_off, var, cov_off, cov_off, cov_off, var,
        ],
    );
    (mu, cov, 0.04)
}

#[test]
#[verified_engine::verified]
fn test_three_asset_rho_060_covariance_vs_independent_kelly_and_cvar() {
    let (mu, cov, r) = three_asset_rho_060_benchmark();

    let cov_sol = unconstrained_covariance_kelly(&mu, &cov, r).unwrap();
    let ind_sol = independent_kelly_weights(&mu, &cov, r).unwrap();

    assert!((cov_sol.metrics.gross_leverage - 30.0 / 11.0).abs() < 1e-10);
    assert!((cov_sol.metrics.gross_leverage - 2.727_272_727).abs() < 1e-6);

    assert!((cov_sol.metrics.excess_log_growth - 0.109_090_909).abs() < 1e-6);

    assert!((ind_sol.metrics.gross_leverage / cov_sol.metrics.gross_leverage - 2.20).abs() < 1e-10);
    assert!((ind_sol.metrics.excess_log_growth - (-0.0480)).abs() < 1e-10);

    let spec = PortfolioHorizonSpec::new(r, 1000.0, 1.0).unwrap();
    let param_risk = parametric_lognormal_tail_risk(&cov_sol.weights, &mu, &cov, spec).unwrap();
    assert!((param_risk.cvar_95 - 550.08).abs() < 2.0);
    assert!(param_risk.cvar_99 > param_risk.cvar_95);
    assert!(param_risk.var_99 > param_risk.var_95);

    let sim_risk =
        simulated_portfolio_tail_risk(&cov_sol.weights, &mu, &cov, spec, 16_384, 42).unwrap();
    assert!((sim_risk.cvar_95 - 550.08).abs() < 6.0);
    assert!(sim_risk.cvar_99 > sim_risk.cvar_95);
}

#[test]
#[verified_engine::verified]
fn test_long_only_projected_gradient_eliminates_shorts_and_respects_leverage_cap() {
    let mu = [0.16, 0.06];
    let r = 0.04;
    let s0: f64 = 0.20;
    let s1: f64 = 0.20;
    let rho = 0.85;
    let cov = DMatrix::from_row_slice(2, 2, &[s0 * s0, rho * s0 * s1, rho * s0 * s1, s1 * s1]);

    let unc = unconstrained_covariance_kelly(&mu, &cov, r).unwrap();
    assert!(
        unc.weights[1] < 0.0,
        "Unconstrained solution must short asset 1"
    );

    let no_short = long_only_kelly_projected_gradient(&mu, &cov, r, None).unwrap();
    assert!((no_short.weights[0] - 3.0).abs() < 1e-6);
    assert!(no_short.weights[1].abs() < 1e-10);

    let simplex = long_only_kelly_projected_gradient(&mu, &cov, r, Some(1.0)).unwrap();
    assert!((simplex.weights[0] - 1.0).abs() < 1e-6);
    assert!(simplex.weights[1].abs() < 1e-10);
    assert!((simplex.metrics.gross_leverage - 1.0).abs() < 1e-6);

    let (mu3, cov3, r3) = three_asset_rho_060_benchmark();
    let sym_simplex = long_only_kelly_projected_gradient(&mu3, &cov3, r3, Some(1.0)).unwrap();
    for &w in &sym_simplex.weights {
        assert!((w - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
#[verified_engine::verified]
fn test_zero_allocation_kelly_and_tail_risk_inside_verified_engine() {
    let (mu, cov, r) = three_asset_rho_060_benchmark();
    let spec = PortfolioHorizonSpec::new(r, 1000.0, 1.0).unwrap();
    let mut ws = PortfolioKellyWorkspace::with_capacity(3);
    let mut weights = [0.0_f64; 3];
    let mut lo_weights = [0.0_f64; 3];
    let mut losses = vec![0.0_f64; 512];
    let mut rng = OxidizeRng::new(2026);

    let (res, _) = VerifiedEngine::run_verified(|| {
        let m_unc = solve_unconstrained_kelly_into(&mu, &cov, r, &mut weights, &mut ws)?;
        let m_lo = solve_long_only_kelly_into(&mu, &cov, r, Some(1.0), &mut lo_weights, &mut ws)?;
        let eval_m = evaluate_portfolio_kelly_metrics(&weights, &mu, &cov, r)?;
        let tail = simulate_portfolio_tail_risk_into(
            &weights,
            &mu,
            &cov,
            spec,
            &mut rng,
            &mut losses,
            &mut ws,
        )?;
        let emp_95 = empirical_var_cvar_in_place(&mut losses, 0.95)?;
        Ok::<_, KellyError>((m_unc, m_lo, eval_m, tail, emp_95))
    });

    let (m_unc, m_lo, eval_m, tail, emp_95) = res.unwrap();
    assert!((m_unc.excess_log_growth - eval_m.excess_log_growth).abs() < 1e-12);
    assert!(m_unc.excess_log_growth > m_lo.excess_log_growth);
    assert!((tail.cvar_95 - emp_95.cvar).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_indefinite_covariance_and_invalid_empirical_slices_rejected() {
    let bad_cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.09, 0.09, 0.04]);
    let err = unconstrained_covariance_kelly(&[0.10, 0.10], &bad_cov, 0.02).unwrap_err();
    assert_eq!(err, KellyError::NotPositiveDefinite);

    assert!(empirical_tail_risk(&[]).is_err());
    assert!(empirical_tail_risk(&[1.0, f64::NAN]).is_err());
}

#[test]
#[verified_engine::verified]
fn test_portfolio_and_copula_workspaces_support_debug_clone_and_eq() {
    use crate::pure_math::statistics::copula::CopulaQmcWorkspace;

    let kelly_ws = PortfolioKellyWorkspace::with_capacity(4);
    assert_eq!(kelly_ws, kelly_ws.clone());
    assert!(!format!("{kelly_ws:?}").is_empty());

    let copula_ws = CopulaQmcWorkspace::with_capacity(6);
    assert_eq!(copula_ws, copula_ws.clone());
    assert!(!format!("{copula_ws:?}").is_empty());
}
