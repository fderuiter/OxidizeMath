use super::*;
use crate::error::KellyError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};

#[test]
#[verified_engine::verified]
fn test_single_bet_clv_and_devigging_benchmarks_within_1e_9() {
    // Placed 2.20 decimal odds, closing fair probability 0.50 (closing fair odds 2.00):
    // EV fraction = 0.50 * 2.20 - 1 = 0.10 (+10.0%)
    // Implied prob edge = 0.50 - 1/2.20 = 1/22 = 0.045454545454545456
    // Log-odds shift = ln(0.50/0.50) + ln(1.20) = ln(1.2) = 0.1823215567939546
    let placed = Odds::new(2.20).unwrap();
    let close = UnitInterval::new(0.50).unwrap();
    let clv = single_bet_clv(&placed, &close).unwrap();
    assert!((clv.ev_fraction - 0.10).abs() < 1e-9);
    assert!((clv.ev_pct - 10.0).abs() < 1e-9);
    assert!((clv.implied_prob_edge - (0.50 - 1.0 / 2.20)).abs() < 1e-9);
    assert!((clv.log_odds_shift - 1.2_f64.ln()).abs() < 1e-9);

    // Two-way -110 / -110 market devigs to exact (0.5, 0.5)
    let vig_a = Odds::from_american(-110.0).unwrap();
    let vig_b = Odds::from_american(-110.0).unwrap();
    let clv_two = single_bet_clv_two_way(&placed, &vig_a, &vig_b).unwrap();
    assert!((clv_two.ev_pct - 10.0).abs() < 1e-9);

    // Three-way 1X2 soccer closing market devigging
    let market = [
        Odds::new(2.00).unwrap(),
        Odds::new(3.50).unwrap(),
        Odds::new(4.00).unwrap(),
    ];
    let mut probs = [UnitInterval::new(0.5).unwrap(); 3];
    let booksum = devig_multi_way_into(&market, &mut probs).unwrap();
    let expected_sum = 0.5 + 1.0 / 3.5 + 0.25;
    assert!((booksum - expected_sum).abs() < 1e-12);
    let p_sum = probs[0].value() + probs[1].value() + probs[2].value();
    assert!((p_sum - 1.0).abs() < 1e-12);
    assert!((probs[0].value() - 0.5 / expected_sum).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_batch_clv_summary_unweighted_and_stake_weighted_within_1e_9() {
    let b1 = SettledBet::new(
        Odds::new(2.20).unwrap(),
        UnitInterval::new(0.50).unwrap(),
        100.0,
        false,
    )
    .unwrap(); // c1 = +0.10
    let b2 = SettledBet::new(
        Odds::new(3.15).unwrap(),
        UnitInterval::new(1.0 / 3.0).unwrap(),
        300.0,
        true,
    )
    .unwrap(); // c2 = +0.05
    let b3 = SettledBet::new(
        Odds::new(1.96).unwrap(),
        UnitInterval::new(0.50).unwrap(),
        100.0,
        false,
    )
    .unwrap(); // c3 = -0.02

    let summary = summarize_clv(&[b1, b2, b3]).unwrap();
    let expected_unweighted: f64 = (0.10 + 0.05 - 0.02) / 3.0;
    let expected_weighted: f64 = (100.0 * 0.10 + 300.0 * 0.05 + 100.0 * (-0.02)) / 500.0;
    let var: f64 = ((0.10 - expected_unweighted).powi(2)
        + (0.05 - expected_unweighted).powi(2)
        + (-0.02 - expected_unweighted).powi(2))
        / 2.0;
    let expected_se = (var / 3.0).sqrt();

    assert_eq!(summary.num_bets, 3);
    assert!((summary.total_staked - 500.0).abs() < 1e-9);
    assert!((summary.unweighted_mean_clv - expected_unweighted).abs() < 1e-9);
    assert!((summary.stake_weighted_mean_clv - expected_weighted).abs() < 1e-9);
    assert!((summary.std_error - expected_se).abs() < 1e-9);
    assert!((summary.z_score - expected_unweighted / expected_se).abs() < 1e-9);
    assert!(summary.p_value_one_sided > 0.0 && summary.p_value_one_sided < 0.5);
    assert!((summary.p_value_two_sided - 2.0 * summary.p_value_one_sided).abs() < 1e-9);
}

#[test]
#[verified_engine::verified]
fn test_brier_and_domain_guarded_log_loss_and_error_guards() {
    let model = [0.65, 0.30, 0.80, 0.20];
    let close = [0.55, 0.45, 0.70, 0.35];
    let outcomes = [true, false, true, false];

    let cmp = compare_calibration(&model, &close, &outcomes).unwrap();
    assert!(cmp.model_brier < cmp.closing_brier);
    assert!(cmp.brier_skill_score > 0.0);
    assert!(cmp.model_log_loss < cmp.closing_log_loss);
    assert!(cmp.log_loss_edge > 0.0);

    // Domain guards reject empty, mismatched, out-of-bounds, and boundary {0, 1} log-loss inputs
    assert_eq!(
        brier_score(&[], &[]),
        Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0
        })
    );
    assert_eq!(
        brier_score(&[0.5], &[true, false]),
        Err(KellyError::DimensionMismatch {
            expected: 1,
            actual: 2
        })
    );
    assert!(matches!(
        brier_score(&[1.2], &[true]),
        Err(KellyError::InvalidProbability { .. })
    ));
    assert!(matches!(
        binary_log_loss(&[0.0], &[false]),
        Err(KellyError::InvalidProbability { value }) if value == 0.0
    ));
    assert!(matches!(
        binary_log_loss(&[1.0], &[true]),
        Err(KellyError::InvalidProbability { value }) if value == 1.0
    ));
    assert!(matches!(
        binary_log_loss(&[f64::NAN], &[true]),
        Err(KellyError::InvalidProbability { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_yield_significance_heterogeneous_odds_and_250_bet_audit() {
    assert_eq!(
        evaluate_yield_significance(&[]),
        Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0
        })
    );

    // 250-bet heterogeneous ledger where bettor beats closing line (+4% CLV) during a losing run
    let mut ledger = Vec::with_capacity(250);
    for i in 0..250 {
        let (placed, close_p, stake, won) = match i % 5 {
            0 => (2.08, 0.50, 100.0, i % 10 < 4),
            1 => (3.12, 1.0 / 3.0, 75.0, i % 15 < 4),
            2 => (1.60, 0.65, 125.0, i % 10 < 6),
            3 => (2.60, 0.40, 80.0, i % 10 < 3),
            _ => (1.92, 0.54, 100.0, i % 2 == 0),
        };
        ledger.push(
            SettledBet::new(
                Odds::new(placed).unwrap(),
                UnitInterval::new(close_p).unwrap(),
                stake,
                won,
            )
            .unwrap(),
        );
    }

    let clv = summarize_clv(&ledger).unwrap();
    let sig_fair = evaluate_yield_significance(&ledger).unwrap();
    let sig_close = evaluate_yield_against_closing(&ledger).unwrap();
    let (close_bs, close_ll) = settled_closing_calibration(&ledger).unwrap();

    assert!(clv.unweighted_mean_clv > 0.035);
    assert!(clv.stake_weighted_mean_clv > 0.035);
    assert!(sig_fair.realized_yield < 0.0);
    assert!(sig_close.z_score < 0.0);
    assert!(close_bs > 0.0 && close_ll > 0.0);

    // Exact sample size check for +10% yield on even-money odds (Var_0 = 1.0):
    // One-sided 95%: ceil(1.6448536^2 / 0.01) = 271; Two-sided 95%: ceil(1.959964^2 / 0.01) = 385
    let even = [Odds::new(2.0).unwrap()];
    let n95 = required_bets_for_95_significance(0.10, &even, None).unwrap();
    assert_eq!(n95.one_sided_bets, 271);
    assert_eq!(n95.two_sided_bets, 385);
    assert!((n95.mean_bernoulli_variance - 1.0).abs() < 1e-12);
}

// [cite:clinical_trials_statistics]
