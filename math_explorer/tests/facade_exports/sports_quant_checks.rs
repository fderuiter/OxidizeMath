use math_explorer::prelude::*;
use math_explorer::sports_quant::*;

fn assert_sgp_copula_and_kelly_composition() {
    let [fair_home, _] = devig_two_way(
        Odds::new(1.80).unwrap(),
        Odds::new(2.10).unwrap(),
        DevigMethod::Shin,
    )
    .unwrap();

    let mut matrix = [0.0_f64; 49];
    dixon_coles_matrix_into(
        PoissonRate::new(1.60).unwrap(),
        PoissonRate::new(1.15).unwrap(),
        DixonColesRho::new(-0.10).unwrap(),
        6,
        &mut matrix,
    )
    .unwrap();
    let outcomes = match_1x2_probabilities(&matrix, 6).unwrap();

    let dc_grid = DixonColesParams::new(1.60, 1.15, -0.10)
        .unwrap()
        .default_grid()
        .unwrap();
    assert!(one_x_two(&dc_grid).home_win > 0.45);

    let zi: ZeroInflation = UnitInterval::new(0.15).unwrap().into();
    let zip = ZipDistribution::new(ZipParams::new(zi, PoissonRate::new(2.0).unwrap()));
    assert!(zip.pmf_unit(Count::new(1)).value() > 0.0);

    let corr = CorrelationMatrix::bivariate(Correlation::new(0.20).unwrap()).unwrap();
    let joint =
        sgp_joint_probability(&[fair_home.into(), outcomes.home_win.into()], &corr).unwrap();
    let joint_unit: UnitInterval = joint.into();

    let stake = kelly_fraction(&joint_unit, &Odds::new(4.50).unwrap()).unwrap();
    let stake_unit: UnitInterval = stake.into();
    assert_eq!(stake, BankrollFraction::from(stake_unit));

    let barrier = kelly_barrier_risk(&joint_unit, &Odds::new(4.50).unwrap(), &stake, 0.5).unwrap();
    assert!((barrier.lundberg_barrier_bound - 0.5).abs() < 1e-6);
    let clv = single_bet_clv(&Odds::new(2.10).unwrap(), &UnitInterval::new(0.50).unwrap()).unwrap();
    assert!((clv.ev_pct - 5.0).abs() < 1e-9);
}

#[test]
fn sports_quant_facade_and_prelude_compose_pricing_into_kelly_without_f64_unwrapping() {
    assert_sgp_copula_and_kelly_composition();

    let glicko_p = head_to_head_win_probability_unit(
        &GlickoPlayer::new(
            Rating::new(1620.0).unwrap(),
            RatingDeviation::new(60.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        &GlickoPlayer::default(),
    );
    let elo_p = elo_expected_score(
        EloRating::new(1620.0).unwrap(),
        EloRating::default(),
        HomeAdvantage::new(40.0).unwrap(),
    );
    let elo_cfg_p = elo_win_probability(
        EloRating::new(1620.0).unwrap(),
        EloRating::default(),
        Venue::Home,
        &EloConfig::default(),
    )
    .unwrap();
    let odds = Odds::new(2.10).unwrap();
    assert!(kelly_fraction(&glicko_p, &odds).unwrap().value() > 0.0);
    assert!(kelly_fraction(&elo_p, &odds).unwrap().value() > 0.0);
    assert!(elo_cfg_p.value() > 0.65);
}
