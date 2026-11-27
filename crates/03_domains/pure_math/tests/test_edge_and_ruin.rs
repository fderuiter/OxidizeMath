//! Integration tests verifying zero heap allocations under `VerifiedAllocator` across
//! `statistics::validation` and `statistics::ruin` (including 10,000-path ensembles).

use oxidize_core::rng::{BASE_SEED, OxidizeRng};
use pure_math::pure_math::statistics::kelly::{Odds, UnitInterval, kelly_fraction};
use pure_math::pure_math::statistics::ruin::{
    BankrollEnsembleWorkspace, PortfolioBet, ProportionalPortfolioBet, gamblers_ruin_with_target,
    kelly_barrier_risk, solve_lundberg_ruin, wilson_interval_95,
};
use pure_math::pure_math::statistics::validation::{
    SettledBet, binary_log_loss, brier_score, compare_calibration, devig_multi_way_into,
    devig_two_way, evaluate_yield_significance, required_bets_for_95_significance, single_bet_clv,
    summarize_clv,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_validation_and_analytical_solvency_zero_allocations() {
    let o1 = Odds::new(2.10).unwrap();
    let o2 = Odds::new(1.91).unwrap();
    let o3 = Odds::new(3.20).unwrap();
    let c1 = UnitInterval::new(0.50).unwrap();
    let c2 = UnitInterval::new(0.54).unwrap();
    let c3 = UnitInterval::new(0.34).unwrap();
    let ledger = [
        SettledBet::new(o1, c1, 100.0, true).unwrap(),
        SettledBet::new(o2, c2, 150.0, false).unwrap(),
        SettledBet::new(o3, c3, 75.0, true).unwrap(),
    ];
    let model_probs = [0.51, 0.55, 0.35];
    let close_probs = [0.50, 0.54, 0.34];
    let outcomes = [true, false, true];
    let odds_slice = [o1, o2, o3];
    let mut devig_out = [c1; 3];
    let portfolio = [
        PortfolioBet::new(c1, o1, 10.0, 0.5).unwrap(),
        PortfolioBet::new(c2, o2, 10.0, 0.3).unwrap(),
        PortfolioBet::new(c3, o3, 5.0, 0.2).unwrap(),
    ];
    let p_kelly = UnitInterval::new(0.55).unwrap();
    let o_even = Odds::new(2.0).unwrap();
    let fk = kelly_fraction(&p_kelly, &o_even).unwrap();

    let ((clv_sum, cal_cmp, yield_sig, n95, lundberg, barrier), metrics) =
        VerifiedEngine::run_verified(|| {
            let _ = devig_two_way(&o1, &o2).unwrap();
            let _ = devig_multi_way_into(&odds_slice, &mut devig_out).unwrap();
            let _ = single_bet_clv(&o1, &c1).unwrap();
            let clv_sum = summarize_clv(&ledger).unwrap();
            let _ = brier_score(&model_probs, &outcomes).unwrap();
            let _ = binary_log_loss(&model_probs, &outcomes).unwrap();
            let cal_cmp = compare_calibration(&model_probs, &close_probs, &outcomes).unwrap();
            let yield_sig = evaluate_yield_significance(&ledger).unwrap();
            let n95 = required_bets_for_95_significance(0.03, &odds_slice, None).unwrap();
            let _ = gamblers_ruin_with_target(40.0, 100.0, 1.0, &p_kelly).unwrap();
            let lundberg = solve_lundberg_ruin(&portfolio, 200.0).unwrap();
            let barrier = kelly_barrier_risk(&p_kelly, &o_even, &fk, 0.50).unwrap();
            (clv_sum, cal_cmp, yield_sig, n95, lundberg, barrier)
        });

    assert!(metrics.function_calls > 0);
    assert!(clv_sum.unweighted_mean_clv > 0.0);
    assert!(cal_cmp.brier_skill_score > 0.0);
    assert!(yield_sig.yield_std_error > 0.0);
    assert!(n95.two_sided_bets > n95.one_sided_bets);
    assert!(lundberg.cramer_lundberg_bound > 0.0 && lundberg.cramer_lundberg_bound < 1.0);
    assert!((barrier.lundberg_barrier_bound - 0.50).abs() < 1e-9);
}

#[test]
fn test_10000_path_ensemble_zero_allocations_and_wilson_precision() {
    let p = UnitInterval::new(0.55).unwrap();
    let odds = Odds::new(2.0).unwrap();
    let fk = kelly_fraction(&p, &odds).unwrap();
    let prop_bet = ProportionalPortfolioBet::unit_weight(p, odds, fk).unwrap();
    let fixed_portfolio = [PortfolioBet::unit_weight(p, odds, 1.0).unwrap()];

    let mut workspace = BankrollEnsembleWorkspace::with_capacity(10_000, 200).unwrap();
    let mut rng = OxidizeRng::from_seed(BASE_SEED);

    let ((prop_report, fixed_report, ci_5pct, q95), metrics) = VerifiedEngine::run_verified(|| {
        let prop_report = workspace
            .simulate_proportional(100.0, 50.0, &prop_bet, 200, &mut rng)
            .unwrap();
        let q95 = workspace.drawdown_quantile(0.95).unwrap();
        let fixed_report = workspace
            .simulate_fixed_stake_portfolio(15.0, 0.0, Some(60.0), &fixed_portfolio, 500, &mut rng)
            .unwrap();
        let ci_5pct = wilson_interval_95(500, 10_000).unwrap();
        (prop_report, fixed_report, ci_5pct, q95)
    });

    assert!(metrics.function_calls > 0);
    assert!(
        (prop_report.ruin_probability - 0.38).abs() < 0.02,
        "Full Kelly 50%-drawdown 200-bet empirical rate should be ~38%, got {}",
        prop_report.ruin_probability
    );
    assert!(
        ci_5pct.width() < 0.01,
        "Wilson 95% width at 5% over 10k paths must be < 1%"
    );
    assert!((prop_report.drawdown_quantiles.p95 - q95).abs() < 1e-12);
    assert!(fixed_report.ruin_probability > 0.0 && fixed_report.ruin_probability < 0.20);
}

// [cite:clinical_trials_statistics]
