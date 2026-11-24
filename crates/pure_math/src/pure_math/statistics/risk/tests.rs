use super::*;
use approx::assert_relative_eq;

#[test]
fn test_historical_var_and_cvar_coherence_and_monotonicity() {
    let returns = [
        -0.12, -0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.01, -0.02,
        0.015, -0.04, 0.025, 0.005, -0.015, 0.035,
    ];
    let mut scratch = [0.0_f64; 20];

    let mut prev_var = -1.0_f64;
    let mut prev_cvar = -1.0_f64;
    for &alpha in &[0.60, 0.75, 0.85, 0.90, 0.95, 0.99] {
        let var = historical_var(&returns, alpha, &mut scratch).expect("valid var");
        let cvar = historical_cvar(&returns, alpha, &mut scratch).expect("valid cvar");
        assert!(
            cvar + 1e-14 >= var,
            "CVaR ({cvar}) must dominate VaR ({var}) at alpha={alpha}"
        );
        assert!(var + 1e-14 >= prev_var, "VaR must be monotone in alpha");
        assert!(cvar + 1e-14 >= prev_cvar, "CVaR must be monotone in alpha");
        prev_var = var;
        prev_cvar = cvar;
    }

    // At alpha = 0.90 on n = 20, worst 2 losses are 0.12 and 0.08 -> average = 0.10.
    let cvar_90 = historical_cvar(&returns, 0.90, &mut scratch).unwrap();
    assert_relative_eq!(cvar_90, 0.10, epsilon = 1e-12);
}

#[test]
fn test_gaussian_and_cornish_fisher_parity_and_fat_tail_expansion() {
    // Exact 6-point symmetric mesokurtic sample: m2 = 2*a^2/6 = a^2/3, m4 = 2*a^4/6 = a^4/3 => m4/m2^2 == 3.
    let sym_mesokurtic = [-0.02, 0.0, 0.0, 0.0, 0.0, 0.02];
    let g_var = gaussian_var(&sym_mesokurtic, 0.95).unwrap();
    let cf_var = cornish_fisher_var(&sym_mesokurtic, 0.95).unwrap();
    let g_cvar = gaussian_cvar(&sym_mesokurtic, 0.95).unwrap();
    let cf_cvar = cornish_fisher_cvar(&sym_mesokurtic, 0.95).unwrap();
    assert_relative_eq!(cf_var, g_var, epsilon = 1e-10);
    assert_relative_eq!(cf_cvar, g_cvar, epsilon = 1e-10);
    assert!(g_cvar > g_var);

    // Negatively skewed crash-prone series: Cornish-Fisher 99% VaR and CVaR exceed Gaussian.
    let skewed_crash = [
        -0.18, 0.01, 0.012, 0.009, 0.011, 0.008, 0.013, 0.01, 0.007, 0.012, 0.009, 0.011,
    ];
    let g99 = gaussian_var(&skewed_crash, 0.99).unwrap();
    let cf99 = cornish_fisher_var(&skewed_crash, 0.99).unwrap();
    let gc99 = gaussian_cvar(&skewed_crash, 0.99).unwrap();
    let cfc99 = cornish_fisher_cvar(&skewed_crash, 0.99).unwrap();
    assert!(
        cf99 > g99,
        "Negative skewness + fat left tail must raise 99% CF VaR ({cf99}) above Gaussian ({g99})"
    );
    assert!(cfc99 >= cf99);
    assert!(cfc99 > gc99);
}

#[test]
fn test_drawdowns_and_underwater_duration_on_wealth_and_returns() {
    // Wealth: 100 -> 120 (peak) -> 90 (dd 25%) -> 96 (dd 20%) -> 120 (recovered!) -> 84 (dd 30%)
    let wealth = [100.0, 120.0, 90.0, 96.0, 120.0, 84.0];
    let mut dd = [0.0_f64; 6];
    let mdd = drawdown_series_from_wealth(&wealth, &mut dd).unwrap();
    assert_relative_eq!(mdd, 0.30, epsilon = 1e-12);
    assert_relative_eq!(dd[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(dd[2], 0.25, epsilon = 1e-12);
    assert_relative_eq!(dd[4], 0.0, epsilon = 1e-12);
    assert_relative_eq!(dd[5], 0.30, epsilon = 1e-12);
    assert_eq!(max_underwater_duration_wealth(&wealth).unwrap(), 2);

    let returns = [0.20, -0.25, 1.0 / 15.0, 0.25, -0.30];
    let mut r_dd = [0.0_f64; 5];
    let r_mdd = drawdown_series_from_returns(&returns, &mut r_dd).unwrap();
    assert_relative_eq!(r_mdd, 0.30, epsilon = 1e-12);
    assert_eq!(max_underwater_duration_returns(&returns).unwrap(), 2);
}

#[test]
fn test_sharpe_sortino_calmar_and_summary() {
    let returns = [0.02, -0.01, 0.03, -0.015, 0.025, 0.01, -0.005, 0.02];
    let sharpe = sharpe_ratio(&returns, 0.001, 252.0).unwrap();
    let sortino = sortino_ratio(&returns, 0.001, 252.0).unwrap();
    let calmar = calmar_ratio(&returns, 252.0).unwrap();
    assert!(sharpe > 0.0);
    assert!(sortino > sharpe);
    assert!(calmar > 0.0);

    let wealth = [100.0, 105.0, 102.0, 110.0, 108.0, 118.0];
    let calmar_w = calmar_ratio_wealth(&wealth, 12.0).unwrap();
    assert!(calmar_w > 0.0);

    let mut scratch = [0.0_f64; 8];
    let summary = summarize_return_risk(&returns, 0.95, 0.001, 252.0, &mut scratch).unwrap();
    assert_relative_eq!(summary.sharpe, sharpe, epsilon = 1e-12);
    assert_relative_eq!(summary.sortino, sortino, epsilon = 1e-12);
    assert_relative_eq!(summary.calmar, calmar, epsilon = 1e-12);
}

#[test]
fn test_risk_error_validation_boundaries() {
    let mut scratch = [0.0_f64; 4];
    assert_eq!(
        historical_var(&[], 0.95, &mut scratch),
        Err(RiskError::EmptySeries)
    );
    assert_eq!(
        historical_var(&[0.01, -0.02], 0.50, &mut scratch),
        Err(RiskError::InvalidConfidence { value: 0.50 })
    );
    assert_eq!(
        historical_var(&[0.01, -0.02], 1.0, &mut scratch),
        Err(RiskError::InvalidConfidence { value: 1.0 })
    );
    assert_eq!(
        historical_var(&[0.01, -0.02, 0.03], 0.95, &mut scratch[..2]),
        Err(RiskError::ScratchBufferTooSmall {
            required: 3,
            actual: 2
        })
    );
    assert_eq!(
        gaussian_var(&[0.01, 0.01, 0.01], 0.95),
        Err(RiskError::ZeroVolatility)
    );
    assert_eq!(
        calmar_ratio(&[0.01, 0.02, 0.01], 252.0),
        Err(RiskError::ZeroMaxDrawdown)
    );
    assert_eq!(
        max_drawdown_returns(&[0.05, -1.0]),
        Err(RiskError::NonPositiveWealth { value: 0.0 })
    );
    assert_eq!(
        sharpe_ratio(&[0.01, -0.02], 0.0, 0.0),
        Err(RiskError::InvalidAnnualization { value: 0.0 })
    );
}

// [cite:clinical_trials_statistics]
