//! End-to-end quantitative sports pricing, rating, parlay, and Kelly sizing pipeline.

use math_explorer::prelude::*;
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
    println!("\n3. ZIP PLAYER PROP + GAUSSIAN COPULA SGP -> FRACTIONAL KELLY");
    let rho_zero: ZeroInflation = UnitInterval::new(0.20)?.into();
    let zip = ZipDistribution::new(ZipParams::new(rho_zero, PoissonRate::new(2.20)?));
    let prop_over_15 = zip.over_probability_unit(Count::new(1));

    let corr = CorrelationMatrix::bivariate(Correlation::new(0.28)?)?;
    let sgp_joint = sgp_joint_probability(&[home_win_prob.into(), prop_over_15.into()], &corr)?;
    let sgp_unit: UnitInterval = sgp_joint.into();

    let book_sgp_odds = Odds::new(3.90)?;
    let sgp_stake = fractional_kelly(&sgp_unit, &book_sgp_odds, 0.50)?;
    println!(
        "   ZIP Over 1.5: {:.2}% | Copula SGP: {:.2}% -> Half-Kelly @ 3.90: {:.2}%",
        prop_over_15.value() * 100.0,
        sgp_unit.value() * 100.0,
        sgp_stake.value() * 100.0
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

fn main() -> Result<()> {
    println!("=== OxidizeMath Connected Sports Quant Pipeline ===\n");
    let sharp_home = stage_devig_and_clv()?;
    let dc_home = stage_score_matrix_pricing()?;
    stage_zip_and_copula_sgp(dc_home)?;
    stage_live_ou_momentum()?;
    stage_ratings_and_ruin(sharp_home)?;
    println!("\nConnected pricing-to-Kelly pipeline completed without f64 unwrapping.");
    Ok(())
}
