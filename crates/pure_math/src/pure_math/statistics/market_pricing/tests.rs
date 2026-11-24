use super::*;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};
use math_commons::registry::TOLERANCE_STANDARD;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_two_way_longshot_bias_and_phantom_ev() {
    let sharp = [
        Odds::from_american(-500.0).unwrap(),
        Odds::from_american(350.0).unwrap(),
    ];
    let ov = market_overround(&sharp).unwrap();
    assert!((ov.booksum - (19.0 / 18.0)).abs() < TOLERANCE_STANDARD);
    assert!((ov.overround - (1.0 / 18.0)).abs() < TOLERANCE_STANDARD);
    assert!((ov.turnover_margin - (1.0 / 19.0)).abs() < TOLERANCE_STANDARD);

    let mut p_mult = [0.0; 2];
    let mut p_add = [0.0; 2];
    let mut p_pow = [0.0; 2];
    let mut p_shin = [0.0; 2];

    devig_multiplicative_into(&sharp, &mut p_mult).unwrap();
    devig_additive_into(&sharp, &mut p_add).unwrap();
    let pow_meta = devig_power_into(&sharp, &mut p_pow).unwrap();
    let shin_meta = devig_shin_into(&sharp, &mut p_shin).unwrap();

    assert!((p_mult[0] - (15.0 / 19.0)).abs() < TOLERANCE_STANDARD);
    assert!((p_mult[1] - (4.0 / 19.0)).abs() < TOLERANCE_STANDARD);
    assert!((p_add[0] - (29.0 / 36.0)).abs() < TOLERANCE_STANDARD);
    assert!((p_add[1] - (7.0 / 36.0)).abs() < TOLERANCE_STANDARD);
    assert!((shin_meta.parameter - 0.057_298_418_459_943).abs() < TOLERANCE_STANDARD);
    assert!((p_shin[0] - (29.0 / 36.0)).abs() < TOLERANCE_STANDARD);
    assert!((p_shin[1] - (7.0 / 36.0)).abs() < TOLERANCE_STANDARD);
    assert!((pow_meta.parameter - 1.121_928_529_667).abs() < TOLERANCE_STANDARD);
    assert!((p_pow[0] - 0.815_012_558_921).abs() < TOLERANCE_STANDARD);
    assert!((p_pow[1] - 0.184_987_441_079).abs() < TOLERANCE_STANDARD);

    let soft = Odds::from_american(400.0).unwrap();
    let cmp = compare_outcome_ev_all_methods(&sharp, 1, soft).unwrap();
    assert!((cmp.multiplicative.ev_per_unit - (1.0 / 19.0)).abs() < TOLERANCE_STANDARD);
    assert!(cmp.multiplicative.has_edge);
    assert!((cmp.shin.ev_per_unit - (-1.0 / 36.0)).abs() < TOLERANCE_STANDARD);
    assert!(!cmp.shin.has_edge);
    assert!(!cmp.power.has_edge);
}

#[test]
#[verified_engine::verified]
fn test_three_way_soccer_devig_and_phantom_ev() {
    let sharp = [
        Odds::new(1.35).unwrap(),
        Odds::new(4.80).unwrap(),
        Odds::new(8.50).unwrap(),
    ];
    let mut p_pow = [0.0; 3];
    let mut p_shin = [0.0; 3];
    let pow_meta = devig_power_into(&sharp, &mut p_pow).unwrap();
    let shin_meta = devig_shin_into(&sharp, &mut p_shin).unwrap();

    assert!((pow_meta.parameter - 1.088_505_595_862).abs() < TOLERANCE_STANDARD);
    assert!((p_pow[0] - 0.721_324_965_530).abs() < TOLERANCE_STANDARD);
    assert!((p_pow[1] - 0.181_328_095_834).abs() < TOLERANCE_STANDARD);
    assert!((p_pow[2] - 0.097_346_938_636).abs() < TOLERANCE_STANDARD);

    assert!((shin_meta.parameter - 0.034_366_923_554).abs() < TOLERANCE_STANDARD);
    assert!((p_shin[0] - 0.712_273_735_617).abs() < TOLERANCE_STANDARD);
    assert!((p_shin[1] - 0.188_245_697_370).abs() < TOLERANCE_STANDARD);
    assert!((p_shin[2] - 0.099_480_567_013).abs() < TOLERANCE_STANDARD);

    let soft_away = Odds::new(9.50).unwrap();
    let cmp = compare_outcome_ev_all_methods(&sharp, 2, soft_away).unwrap();
    assert!((cmp.multiplicative.ev_percent - 4.774_061_782).abs() < TOLERANCE_STANDARD);
    assert!(cmp.multiplicative.has_edge);
    assert!(!cmp.power.has_edge);
    assert!(!cmp.shin.has_edge);
}

#[test]
#[verified_engine::verified]
fn test_six_way_tournament_pod_additive_simplex_violation() {
    let pod = [
        Odds::new(1.90).unwrap(),
        Odds::new(3.25).unwrap(),
        Odds::new(6.50).unwrap(),
        Odds::new(10.0).unwrap(),
        Odds::new(21.0).unwrap(),
        Odds::new(101.0).unwrap(),
    ];
    let mut raw = [0.0; 6];
    let mut clamped = [0.0; 6];
    let meta = devig_additive_detailed_into(&pod, &mut raw, &mut clamped).unwrap();
    let diag = meta
        .simplex_violation
        .expect("101.0 longshot must trigger simplex violation");
    assert_eq!(diag.violated_count, 1);
    assert_eq!(diag.worst_outcome_index, 5);
    assert!((diag.min_raw_probability - (-0.014_328_058_040)).abs() < TOLERANCE_STANDARD);
    assert!((raw[5] - (-0.014_328_058_040)).abs() < TOLERANCE_STANDARD);
    assert_eq!(clamped[5], 0.0);
    let clamped_sum: f64 = clamped.iter().sum();
    assert!((clamped_sum - 1.0).abs() < TOLERANCE_STANDARD);

    let mut p_pow = [0.0; 6];
    let mut p_shin = [0.0; 6];
    devig_power_into(&pod, &mut p_pow).unwrap();
    devig_shin_into(&pod, &mut p_shin).unwrap();
    assert!(p_pow[5] > 0.0 && p_shin[5] > 0.0);
    assert!(((p_pow.iter().sum::<f64>()) - 1.0).abs() < TOLERANCE_STANDARD);
    assert!(((p_shin.iter().sum::<f64>()) - 1.0).abs() < TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_arbitrage_dutching_middle_and_hedge_benchmarks() {
    let (arb, stakes) = evaluate_two_way_arbitrage(
        Odds::new(2.10).unwrap(),
        Odds::new(1.980_392).unwrap(),
        1000.0,
    )
    .unwrap();
    assert!((arb.guaranteed_profit - 19.221_486_975).abs() < 1e-6);
    assert!((stakes[0] - 485.343_565_226).abs() < 1e-6);
    assert!((stakes[1] - 514.656_434_774).abs() < 1e-6);

    let contenders = [
        Odds::new(3.50).unwrap(),
        Odds::new(4.20).unwrap(),
        Odds::new(6.00).unwrap(),
    ];
    let mut dutch_stakes = [0.0; 3];
    let dutch = dutch_outcomes_into(
        &contenders,
        DutchingMode::FixedBudget(100.0),
        &mut dutch_stakes,
    )
    .unwrap();
    assert!((dutch.profit_if_hit - 44.827_586_207).abs() < 1e-6);

    let mut target_stakes = [0.0; 3];
    let dutch_target = dutch_outcomes_into(
        &contenders,
        DutchingMode::TargetProfit(dutch.profit_if_hit),
        &mut target_stakes,
    )
    .unwrap();
    assert!((dutch_target.total_stake - 100.0).abs() < TOLERANCE_STANDARD);

    let minus_110 = Odds::from_american(-110.0).unwrap();
    let mid = evaluate_middle(
        minus_110,
        minus_110,
        220.0,
        Some(UnitInterval::new(0.09).unwrap()),
    )
    .unwrap();
    assert!((mid.break_even_probability - (1.0 / 21.0)).abs() < TOLERANCE_STANDARD);
    assert!((mid.qualifying_loss - 10.0).abs() < TOLERANCE_STANDARD);
    assert!((mid.middle_profit - 200.0).abs() < TOLERANCE_STANDARD);
    assert!(mid.expected_profit.unwrap() > 0.0);

    let orig = Odds::from_american(800.0).unwrap();
    let hedge_line = Odds::from_american(-150.0).unwrap();
    let hedge = evaluate_hedge(100.0, orig, hedge_line, HedgeMode::EqualProfit).unwrap();
    assert!((hedge.hedge_stake - 540.0).abs() < TOLERANCE_STANDARD);
    assert!((hedge.net_profit_if_original_wins - 260.0).abs() < TOLERANCE_STANDARD);
    assert!((hedge.net_profit_if_hedge_wins - 260.0).abs() < TOLERANCE_STANDARD);

    let freeroll = evaluate_hedge(100.0, orig, hedge_line, HedgeMode::TargetFloor(0.0)).unwrap();
    assert!((freeroll.hedge_stake - 150.0).abs() < TOLERANCE_STANDARD);
    assert!(freeroll.net_profit_if_hedge_wins.abs() < TOLERANCE_STANDARD);
    assert!((freeroll.net_profit_if_original_wins - 650.0).abs() < TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_subset_dutching_and_error_guards() {
    let field = [
        Odds::new(2.20).unwrap(),
        Odds::new(4.00).unwrap(),
        Odds::new(6.00).unwrap(),
        Odds::new(12.0).unwrap(),
    ];
    let mut stakes = [0.0; 4];
    let rep = dutch_subset_into(
        &field,
        &[1, 2, 3],
        DutchingMode::FixedBudget(100.0),
        &mut stakes,
    )
    .unwrap();
    assert_eq!(stakes[0], 0.0);
    assert!(stakes[1] > 0.0 && stakes[2] > 0.0 && stakes[3] > 0.0);
    assert!((rep.profit_if_hit - 100.0).abs() < TOLERANCE_STANDARD);

    assert_eq!(
        dutch_subset_into(
            &field,
            &[1, 1],
            DutchingMode::FixedBudget(100.0),
            &mut stakes
        ),
        Err(MarketPricingError::DuplicateOutcomeIndex { index: 1 })
    );
    assert_eq!(
        dutch_subset_into(&field, &[9], DutchingMode::FixedBudget(100.0), &mut stakes),
        Err(MarketPricingError::OutcomeIndexOutOfBounds {
            index: 9,
            outcome_count: 4
        })
    );
    let vig_pair = [
        Odds::from_american(-110.0).unwrap(),
        Odds::from_american(-110.0).unwrap(),
    ];
    let mut arb_stakes = [0.0; 2];
    assert!(matches!(
        evaluate_arbitrage_into(&vig_pair, 100.0, &mut arb_stakes),
        Err(MarketPricingError::NoArbitrage { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_unit_interval_devig_bridges_and_clv_helpers() {
    let line = Odds::from_american(-110.0).unwrap();
    for method in [
        DevigMethod::Multiplicative,
        DevigMethod::Additive,
        DevigMethod::Power,
        DevigMethod::Shin,
    ] {
        let [p1, p2] = devig_two_way(line, line, method).unwrap();
        assert!((p1.value() - 0.5).abs() < TOLERANCE_STANDARD);
        assert!((p2.value() - 0.5).abs() < TOLERANCE_STANDARD);
    }

    let home = Odds::new(1.40).unwrap();
    let draw = Odds::new(4.50).unwrap();
    let away = Odds::new(8.00).unwrap();
    let mult = devig_three_way(home, draw, away, DevigMethod::Multiplicative).unwrap();
    let pow = devig_three_way(home, draw, away, DevigMethod::Power).unwrap();
    let shin = devig_three_way(home, draw, away, DevigMethod::Shin).unwrap();
    assert!(pow[0].value() > mult[0].value() && shin[0].value() > mult[0].value());
    assert!(pow[2].value() < mult[2].value() && shin[2].value() < mult[2].value());

    let vig = book_vigorish(&[Odds::new(1.91).unwrap(), Odds::new(1.91).unwrap()]).unwrap();
    assert!(vig.value() > 0.04 && vig.value() < 0.06);
    assert!((fair_odds(UnitInterval::new(0.4).unwrap()).unwrap().value() - 2.5).abs() < 1e-12);
    assert!(fair_odds(UnitInterval::new(0.0).unwrap()).is_err());
    assert!(fair_odds(UnitInterval::new(1.0).unwrap()).is_err());

    let (clv, fair_close) = closing_line_edge_two_way(
        Odds::new(2.10).unwrap(),
        Odds::new(1.85).unwrap(),
        Odds::new(2.05).unwrap(),
        DevigMethod::Shin,
    )
    .unwrap();
    assert!(fair_close.value() > 0.5 && clv > 0.05);
}

#[test]
#[verified_engine::verified]
fn test_zero_heap_allocations_across_all_market_pricing_routines() {
    let market = [
        Odds::new(1.35).unwrap(),
        Odds::new(4.80).unwrap(),
        Odds::new(8.50).unwrap(),
    ];
    let arb_odds = [Odds::new(2.10).unwrap(), Odds::new(1.980_392).unwrap()];
    let offered = Odds::new(9.50).unwrap();
    let minus_110 = Odds::from_american(-110.0).unwrap();
    let orig = Odds::from_american(800.0).unwrap();
    let hedge_line = Odds::from_american(-150.0).unwrap();

    let (ok, _) = VerifiedEngine::run_verified(|| -> Result<f64, MarketPricingError> {
        let mut probs = [0.0; 3];
        let mut fair = [Odds::new(2.0).unwrap(); 3];
        devig_multiplicative_into(&market, &mut probs)?;
        devig_additive_into(&market, &mut probs)?;
        devig_power_into(&market, &mut probs)?;
        devig_shin_into(&market, &mut probs)?;
        fair_odds_from_probabilities_into(&probs, &mut fair)?;
        let ev = compare_outcome_ev_all_methods(&market, 2, offered)?;
        let mut arb_stakes = [0.0; 2];
        let arb = evaluate_arbitrage_into(&arb_odds, 1000.0, &mut arb_stakes)?;
        let mut sub_stakes = [0.0; 3];
        let dutch = dutch_subset_into(
            &market,
            &[1, 2],
            DutchingMode::FixedBudget(100.0),
            &mut sub_stakes,
        )?;
        let mid = evaluate_middle(minus_110, minus_110, 220.0, None)?;
        let hdg = evaluate_hedge(100.0, orig, hedge_line, HedgeMode::EqualProfit)?;
        let (clv, _) = closing_line_edge_two_way(offered, minus_110, minus_110, DevigMethod::Shin)?;
        Ok(ev.shin.ev_per_unit
            + arb.guaranteed_profit
            + dutch.profit_if_hit
            + mid.middle_profit
            + hdg.hedge_stake
            + clv)
    });
    assert!(ok.unwrap().is_finite());
}

// [cite:sports_quantitative_modeling]
