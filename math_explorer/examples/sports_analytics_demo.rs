//! End-to-end quantitative sports pricing, rating, parlay, execution, and Kelly sizing pipeline.

use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::prelude::*;
use math_explorer::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
use math_explorer::sports_quant::*;

fn stage_devig_and_clv() -> Result<UnitInterval> {
    println!("1. MARKET DEVIGGING & CLV (Shin Insider Equilibrium -> Kelly)");
    let sharp_home = Odds::new(1.80)?;
    let sharp_away = Odds::new(2.10)?;
    let vig = book_vigorish(&[sharp_home, sharp_away])?;
    let [fair_home, _fair_away] = devig_two_way(sharp_home, sharp_away, DevigMethod::Shin)?;

    let soft_home_odds = Odds::new(1.96)?;
    let stake = kelly_fraction(&fair_home, &soft_home_odds)?;
    let clv = closing_line_value(soft_home_odds, fair_home);
    println!(
        "   Sharp vig: {:.2}% | Fair Home: {:.2}% | Soft 1.96 CLV: {:+.2}% | Kelly: {:.2}%",
        vig.value() * 100.0,
        fair_home.value() * 100.0,
        clv * 100.0,
        stake.value() * 100.0
    );

    let sharp_1x2 = [Odds::new(1.35)?, Odds::new(4.80)?, Odds::new(8.50)?];
    let cmp = compare_outcome_ev_all_methods(&sharp_1x2, 2, Odds::new(9.50)?)?;
    println!(
        "   3-Way Away (9.50 vs 8.50): Multiplicative EV = {:+.2}%, Shin EV = {:+.2}% (phantom edge filtered)",
        cmp.for_method(DevigMethod::Multiplicative).ev_percent,
        cmp.for_method(DevigMethod::Shin).ev_percent
    );
    Ok(fair_home)
}

fn stage_score_matrix_pricing() -> Result<UnitInterval> {
    println!("\n2. MATCH SCORE MATRIX (Dixon-Coles 9x9 Grid -> Over 2.5 -> Kelly)");
    let mut grid = [0.0_f64; 81];
    dixon_coles_matrix_into(
        PoissonRate::new(1.65)?,
        PoissonRate::new(1.20)?,
        DixonColesRho::new(-0.12)?,
        8,
        &mut grid,
    )?;
    let outcomes = match_1x2_probabilities(&grid, 8)?;
    let (over_25, _under_25) = over_under_probabilities(&grid, 8, 2.5)?;
    let over_odds = Odds::new(2.02)?;
    let over_stake = kelly_fraction(&over_25, &over_odds)?;
    println!(
        "   1X2 = ({:.1}%, {:.1}%, {:.1}%) | P(Over 2.5) = {:.2}% -> Kelly @ 2.02: {:.2}%",
        outcomes.home_win.value() * 100.0,
        outcomes.draw.value() * 100.0,
        outcomes.away_win.value() * 100.0,
        over_25.value() * 100.0,
        over_stake.value() * 100.0
    );
    Ok(outcomes.home_win)
}

fn stage_zip_and_copula_sgp(home_win_prob: UnitInterval) -> Result<()> {
    println!("\n3. ZIP PLAYER PROP + GENZ GAUSSIAN COPULA SGP -> FRACTIONAL KELLY");
    let rho_zero: ZeroInflation = UnitInterval::new(0.20)?.into();
    let zip = ZipDistribution::new(ZipParams::new(rho_zero, PoissonRate::new(2.20)?));
    let prop_over_15 = zip.over_probability_unit(Count::new(1));

    let corr_2 = CorrelationMatrix::bivariate(Correlation::new(0.28)?)?;
    let sgp_joint = sgp_joint_probability(&[home_win_prob.into(), prop_over_15.into()], &corr_2)?;
    let sgp_unit: UnitInterval = sgp_joint.into();
    let sgp_stake = fractional_kelly(&sgp_unit, &Odds::new(3.90)?, 0.50)?;
    println!(
        "   2-Leg ZIP+Match SGP: {:.2}% -> Half-Kelly @ 3.90: {:.2}%",
        sgp_unit.value() * 100.0,
        sgp_stake.value() * 100.0
    );

    let legs_3 = [
        home_win_prob.into(),
        prop_over_15.into(),
        Probability::new(0.60)?,
    ];
    let corr_3 = CorrelationMatrix::new(DMatrix::from_row_slice(
        3,
        3,
        &[1.0, 0.28, 0.20, 0.28, 1.0, 0.35, 0.20, 0.35, 1.0],
    ))?;
    let priced = price_correlated_parlay(&legs_3, &corr_3)?;
    let smoke = std::env::var("MATH_EXPLORER_SMOKE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let mut rng = OxidizeRng::new(2026);
    let sim = simulate_correlated_parlay(
        &legs_3,
        &corr_3,
        if smoke { 2_500 } else { 25_000 },
        &mut rng,
    )?;
    let sgp3_unit: UnitInterval = priced.joint_probability.into();
    let sgp3_stake = fractional_kelly(&sgp3_unit, &Odds::new(6.25)?, 0.50)?;
    println!(
        "   3-Leg SGP: Naive={:.4}, Genz={:.4} (x{:.2}, Fair={:.2}), MC={:.4} ± {:.4} -> Half-Kelly: {:.2}%",
        priced.naive_probability.value(),
        priced.joint_probability.value(),
        priced.correlation_multiplier,
        priced.fair_decimal_odds,
        sim.joint_probability.value(),
        sim.standard_error,
        sgp3_stake.value() * 100.0
    );
    Ok(())
}

fn stage_live_ou_momentum() -> Result<()> {
    println!("\n4. LIVE ORNSTEIN-UHLENBECK MOMENTUM -> IN-PLAY COMEBACK KELLY");
    let params = OuParams::from_values(2.5, 1.1, 4.5)?;
    let analyzer = OuAnalyzer::new(params, TimeStep::new(0.02)?);
    let mut rng = OxidizeRng::new(2026);
    let live_comeback = analyzer.comeback_probability_unit(-3.0, 0.0, 1.0, 2_000, &mut rng);
    let live_odds = Odds::new(2.15)?;
    let live_stake = kelly_fraction(&live_comeback, &live_odds)?;
    println!(
        "   Down 3 pts (mu=+2.5): Comeback = {:.2}% -> Live Kelly @ 2.15: {:.2}%",
        live_comeback.value() * 100.0,
        live_stake.value() * 100.0
    );
    Ok(())
}

fn stage_ratings_and_ruin(devigged_ref: UnitInterval) -> Result<()> {
    println!("\n5. ELO & GLICKO-2 RATINGS -> KELLY & RISK OF RUIN");
    let elo_prob = elo_expected_score(
        EloRating::new(1615.0)?,
        EloRating::new(1530.0)?,
        HomeAdvantage::new(45.0)?,
    );
    let contender = GlickoPlayer::new(
        Rating::new(1630.0)?,
        RatingDeviation::new(65.0)?,
        Volatility::new(0.06)?,
    );
    let opponent = GlickoPlayer::new(
        Rating::new(1520.0)?,
        RatingDeviation::new(75.0)?,
        Volatility::new(0.06)?,
    );
    let glicko_prob = head_to_head_win_probability_unit(&contender, &opponent);
    let market_odds = Odds::from_american(-135.0)?;
    let full_stake = kelly_fraction(&glicko_prob, &market_odds)?;
    let ror_40u = unit_risk_of_ruin(&glicko_prob, &market_odds, 40.0)?;
    let dd_50 = kelly_drawdown_risk(&glicko_prob, &market_odds, &full_stake, 0.5)?;
    println!(
        "   Elo: {:.2}% | Glicko-2: {:.2}% (Sharp ref: {:.2}%) -> Kelly: {:.2}% | 40u RoR: {:.2}% | 50% DD: {:.1}%",
        elo_prob.value() * 100.0,
        glicko_prob.value() * 100.0,
        devigged_ref.value() * 100.0,
        full_stake.value() * 100.0,
        ror_40u.value() * 100.0,
        dd_50.value() * 100.0
    );
    Ok(())
}

fn stage_markov_and_execution() -> Result<()> {
    println!("\n6. MARKOV POSSESSION EPV & CROSS-BOOK EXECUTION");
    let states = vec![
        StateType::Transient,
        StateType::Transient,
        StateType::Absorbing,
        StateType::Absorbing,
    ];
    let p = DMatrix::from_row_slice(
        4,
        4,
        &[
            0.3, 0.5, 0.1, 0.1, 0.2, 0.4, 0.3, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    let chain = MarkovChain::new(p, states)?;
    let epv = chain.expected_possession_value(&DVector::from_vec(vec![3.0, 0.0]))?;

    let pod = [
        Odds::new(1.90)?,
        Odds::new(3.25)?,
        Odds::new(6.50)?,
        Odds::new(10.0)?,
        Odds::new(21.0)?,
        Odds::new(101.0)?,
    ];
    let (mut raw, mut clamped) = ([0.0; 6], [0.0; 6]);
    let meta = devig_additive_detailed_into(&pod, &mut raw, &mut clamped)?;
    let (arb, _) = evaluate_two_way_arbitrage(Odds::new(2.10)?, Odds::new(1.980_392)?, 1000.0)?;
    let mut dutch_stakes = [0.0; 3];
    let dutch = dutch_outcomes_into(
        &[Odds::new(3.50)?, Odds::new(4.20)?, Odds::new(6.00)?],
        DutchingMode::FixedBudget(100.0),
        &mut dutch_stakes,
    )?;
    let minus_110 = Odds::from_american(-110.0)?;
    let mid = evaluate_middle(minus_110, minus_110, 220.0, Some(UnitInterval::new(0.09)?))?;
    let hedge = evaluate_hedge(
        100.0,
        Odds::from_american(800.0)?,
        Odds::from_american(-150.0)?,
        HedgeMode::EqualProfit,
    )?;
    println!(
        "   EPV=({:.2}, {:.2}) | 6-Way Guard={:?} | Arb +${:.2} | Dutch +${:.2} | Mid BE {:.2}% | Hedge ${:.2}",
        epv[0],
        epv[1],
        meta.simplex_violation.map(|d| d.violated_count),
        arb.guaranteed_profit,
        dutch.profit_if_hit,
        mid.break_even_probability * 100.0,
        hedge.hedge_stake
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("=== OxidizeMath Connected Sports Quant Pipeline ===\n");
    let sharp_home = stage_devig_and_clv()?;
    let dc_home = stage_score_matrix_pricing()?;
    stage_zip_and_copula_sgp(dc_home)?;
    stage_live_ou_momentum()?;
    stage_ratings_and_ruin(sharp_home)?;
    stage_markov_and_execution()?;
    println!("\nConnected pricing, execution, and Kelly pipeline completed cleanly.");
    Ok(())
}
