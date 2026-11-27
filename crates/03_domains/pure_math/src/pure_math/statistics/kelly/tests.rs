//! Acceptance, KKT optimality, Laplace quadrature, and zero-allocation tests for multi-state Kelly.

use super::*;
use crate::error::KellyError;
use std::sync::Mutex;

static KELLY_ALLOC_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[verified_engine::verified]
fn test_ac1_push_kelly_minus_3_spread_at_minus_110() {
    // -3 spread at -110: p_win = 0.50, p_push = 0.10, p_loss = 0.40 => EV = +5.4545%, f* = 6.6667%
    let odds = Odds::from_american(-110.0).unwrap();
    let wager = PushWager::new(0.50, 0.10, 0.40, odds).unwrap();

    let ev = push_expected_value(&wager);
    let f_star = push_kelly_fraction(&wager).unwrap();
    let g_star = push_expected_growth_rate(&wager, &f_star);

    assert!((ev - 3.0 / 55.0).abs() < 1e-12);
    assert!((ev * 100.0 - 5.4545454545).abs() < 1e-9);
    assert!((f_star.value() - 1.0 / 15.0).abs() < 1e-12);
    assert!((f_star.value() * 100.0 - 6.6666666667).abs() < 1e-9);
    assert!(g_star > 0.0);

    // Binary Kelly erroneously rejects this wager when setting q = 1 - p_win = 0.50.
    let p_bin = UnitInterval::new(0.50).unwrap();
    assert!(matches!(
        kelly_fraction(&p_bin, &odds),
        Err(KellyError::NoEdge { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_ac2_quarter_line_minus_275_at_minus_110() {
    // -2.75 quarter-line at -110: p_fw = 0.44, p_hw = 0.12, p_fl = 0.44 => EV = +1.4545%, f* = 1.7517%
    let odds = Odds::from_american(-110.0).unwrap();
    let ql = QuarterLineWager::new(0.44, 0.12, 0.0, 0.0, 0.44, odds).unwrap();

    let ev = quarter_line_expected_value(&ql);
    let f_star = quarter_line_kelly_fraction(&ql).unwrap();
    let deriv = quarter_line_growth_derivative(&ql, f_star.value());
    let g_star = quarter_line_expected_growth_rate(&ql, &f_star);

    assert!((ev - 0.16 / 11.0).abs() < 1e-12);
    assert!((ev * 100.0 - 1.4545454545).abs() < 1e-9);
    assert!(deriv.abs() < 1e-11);
    assert!((f_star.value() - 0.0175166500852366).abs() < 1e-9);
    assert!((f_star.value() * 100.0 - 1.7517).abs() < 5e-4);
    assert!(g_star > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_ac3_ten_game_slate_exact_and_laplace_agreement() {
    // 10-game slate (p = 0.58, b = 1.0): 9.9672% per game (99.6725% total), Laplace agrees within 1e-7.
    let p = UnitInterval::new(0.58).unwrap();
    let odds = Odds::from_net_odds(1.0).unwrap();
    let wagers = [SlateWager::binary(p, odds); 10];
    let mut scratch = SlateScratch::new();

    let exact = solve_simultaneous_slate(&wagers, &mut scratch).unwrap();
    assert_eq!(exact.eval_mode_used(), SlateEvalMode::ExactConvolution);
    assert!(exact.total_fraction() < 1.0);
    assert!(exact.kkt_residual() < 1e-9);
    for &f_i in exact.stakes() {
        assert!((f_i - 0.0996724797394549).abs() < 1e-9);
    }
    assert!((exact.total_fraction() - 0.996724797394549).abs() < 1e-9);

    let laplace =
        solve_simultaneous_slate_with_mode(&wagers, SlateEvalMode::LaplaceQuadrature, &mut scratch)
            .unwrap();
    assert_eq!(laplace.eval_mode_used(), SlateEvalMode::LaplaceQuadrature);
    assert!(laplace.kkt_residual() < 1e-9);
    for (&f_ex, &f_lap) in exact.stakes().iter().zip(laplace.stakes().iter()) {
        assert!((f_ex - f_lap).abs() < 1e-7);
    }
    assert!((exact.total_fraction() - laplace.total_fraction()).abs() < 1e-7);
}

#[test]
#[verified_engine::verified]
fn test_large_slate_auto_selects_laplace_quadrature_and_satisfies_kkt() {
    let p = UnitInterval::new(0.56).unwrap();
    let odds = Odds::from_net_odds(1.0).unwrap();
    let wagers = [SlateWager::binary(p, odds); 16];
    let mut scratch = SlateScratch::new();

    let sol = solve_simultaneous_slate(&wagers, &mut scratch).unwrap();
    assert_eq!(sol.eval_mode_used(), SlateEvalMode::LaplaceQuadrature);
    assert!(sol.total_fraction() > 0.0 && sol.total_fraction() < 1.0);
    assert!(sol.kkt_residual() < 1e-9);
}

#[test]
#[verified_engine::verified]
fn test_ac4_mutually_exclusive_3_way_market_smoczynski_tomkins() {
    // [0.50, 0.28, 0.22] at [2.20, 4.00, 3.20] => [16.1538%, 9.3846%, 0.0%]
    let outcomes = [
        MarketOutcome::new(UnitInterval::new(0.50).unwrap(), Odds::new(2.20).unwrap()),
        MarketOutcome::new(UnitInterval::new(0.28).unwrap(), Odds::new(4.00).unwrap()),
        MarketOutcome::new(UnitInterval::new(0.22).unwrap(), Odds::new(3.20).unwrap()),
    ];
    let mut scratch = MarketScratch::new();
    let alloc = solve_mutually_exclusive_market(&outcomes, &mut scratch).unwrap();

    assert!((alloc.stakes()[0] - 21.0 / 130.0).abs() < 1e-12);
    assert!((alloc.stakes()[1] - 61.0 / 650.0).abs() < 1e-12);
    assert_eq!(alloc.stakes()[2], 0.0);
    assert!(alloc.total_fraction() < 1.0);
    assert!(alloc.kkt_residual() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_ac5_busseti_ryu_boyd_drawdown_single_and_ten_game_slate() {
    // alpha = 0.70, beta = 0.10 => lambda = 6.4557, single = 2.6900%, 10-game slate = 41.4898%
    let dc = DrawdownConstraint::new(0.70, 0.10).unwrap();
    assert!((dc.lambda() - 6.455696235812881).abs() < 1e-12);
    assert!((dc.gamma() - (1.0 + dc.lambda())).abs() < 1e-12);

    let p_single = UnitInterval::new(0.55).unwrap();
    let o_single = Odds::from_net_odds(1.0).unwrap();
    let f_single = drawdown_constrained_kelly(&p_single, &o_single, &dc).unwrap();
    assert!((f_single.value() - 0.0268997434950584).abs() < 1e-9);
    let single_martingale = 0.55 * (1.0 + f_single.value()).powf(-dc.lambda())
        + 0.45 * (1.0 - f_single.value()).powf(-dc.lambda());
    assert!((single_martingale - 1.0).abs() < 1e-10);

    let p_slate = UnitInterval::new(0.58).unwrap();
    let o_slate = Odds::from_net_odds(1.0).unwrap();
    let wagers = [SlateWager::binary(p_slate, o_slate); 10];
    let mut scratch = SlateScratch::new();
    let sol = solve_drawdown_constrained_slate(&wagers, &dc, &mut scratch).unwrap();
    assert!((sol.total_fraction() - 0.4148978997813092).abs() < 1e-9);
    assert!(sol.drawdown_expectation() <= 1.0 + 1e-9);
    assert!((sol.drawdown_expectation() - 1.0).abs() < 1e-9);
    assert!(sol.kkt_residual() < 1e-9);
}

#[test]
#[verified_engine::verified]
fn test_all_kelly_solvers_run_with_zero_heap_allocations() {
    let odds_110 = Odds::from_american(-110.0).unwrap();
    let push_w = PushWager::new(0.50, 0.10, 0.40, odds_110).unwrap();
    let ql_w = QuarterLineWager::new(0.44, 0.12, 0.0, 0.0, 0.44, odds_110).unwrap();
    let market = [
        MarketOutcome::new(UnitInterval::new(0.50).unwrap(), Odds::new(2.20).unwrap()),
        MarketOutcome::new(UnitInterval::new(0.28).unwrap(), Odds::new(4.00).unwrap()),
        MarketOutcome::new(UnitInterval::new(0.22).unwrap(), Odds::new(3.20).unwrap()),
    ];
    let p_slate = UnitInterval::new(0.58).unwrap();
    let o_slate = Odds::from_net_odds(1.0).unwrap();
    let wagers_10 = [SlateWager::binary(p_slate, o_slate); 10];
    let wagers_14 = [SlateWager::binary(p_slate, o_slate); 14];
    let dc = DrawdownConstraint::new(0.70, 0.10).unwrap();

    let mut m_scratch = MarketScratch::new();
    let mut s_scratch = SlateScratch::new();

    let _guard = KELLY_ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        let f_p = push_kelly_fraction(&push_w).unwrap();
        let f_ql = quarter_line_kelly_fraction(&ql_w).unwrap();
        let m_sol = solve_mutually_exclusive_market(&market, &mut m_scratch).unwrap();
        let s_ex = solve_simultaneous_slate(&wagers_10, &mut s_scratch).unwrap();
        let s_lap = solve_simultaneous_slate(&wagers_14, &mut s_scratch).unwrap();
        let rc_p = drawdown_constrained_push_kelly(&push_w, &dc).unwrap();
        let rc_ql = drawdown_constrained_quarter_line_kelly(&ql_w, &dc).unwrap();
        let rc_s = solve_drawdown_constrained_slate(&wagers_10, &dc, &mut s_scratch).unwrap();
        assert!(
            f_p.value() > 0.0
                && f_ql.value() > 0.0
                && m_sol.total_fraction() > 0.0
                && s_ex.total_fraction() > 0.0
                && s_lap.total_fraction() > 0.0
                && rc_p.value() > 0.0
                && rc_ql.value() > 0.0
                && rc_s.total_fraction() > 0.0
        );
    });
}

#[test]
#[verified_engine::verified]
fn test_validation_and_settlement_helpers() {
    let odds = Odds::from_net_odds(1.0).unwrap();
    assert!(PushWager::new(0.6, 0.3, 0.3, odds).is_err());
    assert!(QuarterLineWager::new(0.5, 0.5, 0.1, 0.0, 0.0, odds).is_err());
    assert!(DrawdownConstraint::new(0.0, 0.1).is_err());
    assert!(DrawdownConstraint::new(0.7, 1.0).is_err());

    let pw = PushWager::from_win_and_push(0.50, 0.10, odds).unwrap();
    let frac = BankrollFraction::new(0.10).unwrap();
    assert_eq!(pw.sample_outcome(0.25), PushOutcome::Win);
    assert_eq!(pw.sample_outcome(0.55), PushOutcome::Push);
    assert_eq!(pw.sample_outcome(0.85), PushOutcome::Loss);
    assert!((pw.settle(1000.0, &frac, PushOutcome::Push) - 1000.0).abs() < 1e-12);

    let ql = QuarterLineWager::new(0.25, 0.25, 0.10, 0.15, 0.25, odds).unwrap();
    assert_eq!(ql.sample_outcome(0.30), QuarterLineOutcome::HalfWin);
    assert!((ql.settle(1000.0, &frac, QuarterLineOutcome::HalfWin) - 1050.0).abs() < 1e-12);
    assert!((ql.settle(1000.0, &frac, QuarterLineOutcome::HalfLoss) - 950.0).abs() < 1e-12);
}
