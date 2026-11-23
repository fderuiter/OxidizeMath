//! Comprehensive demonstration of all sports analytics modules.

use math_explorer::error::Result;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::{
    copula::{
        Correlation, CorrelationMatrix, Probability, price_correlated_parlay,
        sgp_joint_probability, simulate_correlated_parlay,
    },
    glicko2::{
        GlickoPlayer, MatchResult, Rating, RatingDeviation, SystemConstant, Volatility,
        update_rating,
    },
    kelly::{Odds, UnitInterval, kelly_fraction},
    markov::dtmc::{MarkovChain, StateType},
    ou_process::{EulerMaruyama, OuParams, TimeStep},
    zip_regression::{Count, ZipDistribution, ZipParams},
};
use math_explorer::rng::OxidizeRng;

fn demo_zip_regression() -> Result<()> {
    println!("1. ZIP REGRESSION - Player Block Modeling");
    let zip_params = ZipParams::from_values(0.25, 1.5)?;
    let zip_dist = ZipDistribution::new(zip_params);
    println!("   Zero-inflation: 25%, Poisson rate: 1.5");
    println!("   P(0 blocks) = {:.3}", zip_dist.pmf(Count::new(0)));
    println!(
        "   Mean: {:.2}, Variance: {:.2} (overdispersed!)",
        zip_dist.mean(),
        zip_dist.variance()
    );
    Ok(())
}

fn demo_ou_process() -> Result<()> {
    println!("\n2. OU PROCESS - Shooting Percentage Momentum");
    let ou_params = OuParams::from_values(0.45, 1.0, 0.15)?;
    let dt = TimeStep::new(0.01)?;
    let solver = EulerMaruyama::new(ou_params, dt);
    let mut rng = OxidizeRng::default();
    let trajectory = solver.simulate(0.60, 100, &mut rng);
    println!("   True skill: 45%, Currently: 60% (hot)");
    println!(
        "   After 100 possessions: {:.1}%",
        trajectory.last().copied().unwrap_or(0.60) * 100.0
    );
    Ok(())
}

fn demo_copula() -> Result<()> {
    println!("\n3. GAUSSIAN COPULA - Same Game Parlay Pricing");
    let p_star_50 = Probability::new(0.01)?;
    let p_win = Probability::new(0.60)?;
    let rho = Correlation::new(-0.15)?;
    let corr_2 = CorrelationMatrix::bivariate(rho)?;
    let joint_2 = sgp_joint_probability(&[p_star_50, p_win], &corr_2)?;
    println!("   2-Leg Hero Ball: Star 50+ pts (1% tail) + Team win (60%)");
    println!(
        "   Naive: {:.4}, Copula: {:.4}",
        0.01 * 0.60,
        joint_2.value()
    );

    let legs_3 = [
        Probability::new(0.55)?,
        Probability::new(0.50)?,
        Probability::new(0.60)?,
    ];
    let corr_3 = CorrelationMatrix::new(DMatrix::from_row_slice(
        3,
        3,
        &[1.0, 0.50, 0.25, 0.50, 1.0, 0.35, 0.25, 0.35, 1.0],
    ))?;
    let priced = price_correlated_parlay(&legs_3, &corr_3)?;
    let mut rng = OxidizeRng::default();
    let sim = simulate_correlated_parlay(&legs_3, &corr_3, 25_000, &mut rng)?;
    println!(
        "   3-Leg SGP: Naive={:.4}, Genz={:.4} (x{:.2}, Fair Odds={:.2}), MC={:.4} ± {:.4}",
        priced.naive_probability.value(),
        priced.joint_probability.value(),
        priced.correlation_multiplier,
        priced.fair_decimal_odds,
        sim.joint_probability.value(),
        sim.standard_error
    );
    Ok(())
}

fn demo_glicko2() -> Result<()> {
    println!("\n4. GLICKO-2 RATING - Team Ranking");
    let mut player = GlickoPlayer::new(
        Rating::new(1500.0)?,
        RatingDeviation::new(200.0)?,
        Volatility::new(0.06)?,
    );
    let opponent_rating = Rating::new(1400.0)?;
    let opponent_rd = RatingDeviation::new(30.0)?;
    let opponent = GlickoPlayer::new(opponent_rating, opponent_rd, Volatility::default());
    let results = vec![MatchResult::new(opponent, 1.0)?];
    let tau = SystemConstant::new(0.5)?;
    player = update_rating(&player, &results, &tau)?;
    println!("   Before: R=1500, RD=200");
    println!(
        "   After win: R={:.0}, RD={:.0}",
        player.rating.value(),
        player.rating_deviation.value()
    );
    Ok(())
}

fn demo_kelly() -> Result<()> {
    println!("\n5. KELLY CRITERION - Bet Sizing");
    let prob = UnitInterval::new(0.55)?;
    let odds = Odds::new(2.0)?;
    let kelly = kelly_fraction(&prob, &odds)?;
    println!("   Win probability: 55%, Odds: 2.0");
    println!("   Full Kelly: {:.1}%", kelly.value() * 100.0);
    println!(
        "   Quarter Kelly: {:.1}% (recommended)",
        kelly.value() * 25.0
    );
    Ok(())
}

fn demo_markov_chain() -> Result<()> {
    println!("\n6. MARKOV CHAIN - Expected Possession Value");
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
    let rewards = DVector::from_vec(vec![3.0, 0.0]);
    let epv = chain.expected_possession_value(&rewards)?;
    println!("   State 0 EPV: {:.2} points", epv[0]);
    println!("   State 1 EPV: {:.2} points", epv[1]);
    Ok(())
}

fn main() -> Result<()> {
    println!("=== Sports Analytics Framework Demo ===\n");
    demo_zip_regression()?;
    demo_ou_process()?;
    demo_copula()?;
    demo_glicko2()?;
    demo_kelly()?;
    demo_markov_chain()?;
    println!("\n✅ All 6 sports analytics modules demonstrated successfully!");
    Ok(())
}
