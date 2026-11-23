//! Comprehensive demonstration of all sports analytics modules.

use math_explorer::error::Result;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::{
    copula::{Correlation, CorrelationMatrix, Probability, sgp_joint_probability},
    glicko2::{
        GlickoPlayer, MatchResult, Rating, RatingDeviation, SystemConstant, Volatility,
        update_rating,
    },
    kelly::{Odds, UnitInterval, kelly_fraction},
    market_pricing::{
        DevigMethod, DutchingMode, HedgeMode, compare_outcome_ev_all_methods,
        devig_additive_detailed_into, dutch_outcomes_into, evaluate_hedge, evaluate_middle,
        evaluate_two_way_arbitrage,
    },
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
    println!("\n3. GAUSSIAN COPULA - Same Game Parlay");
    let p_star_50 = Probability::new(0.99)?;
    let p_win = Probability::new(0.60)?;
    let rho = Correlation::new(-0.15)?;
    let corr_matrix = CorrelationMatrix::bivariate(rho)?;
    let joint = sgp_joint_probability(&[p_star_50, p_win], &corr_matrix)?;
    println!("   Event A: Star scores 50+ (99th percentile)");
    println!("   Event B: Team wins (60% base)");
    println!("   Naive: {:.4}, Copula: {:.4}", 0.99 * 0.60, joint.value());
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

fn demo_market_pricing() -> Result<()> {
    println!("\n7. MARKET PRICING & EXECUTION - Devigging, +EV, Arbitrage, Dutching & Hedge");
    let sharp = [Odds::new(1.35)?, Odds::new(4.80)?, Odds::new(8.50)?];
    let soft_away = Odds::new(9.50)?;
    let cmp = compare_outcome_ev_all_methods(&sharp, 2, soft_away)?;
    println!(
        "   3-Way Away (9.50 vs 8.50): Multiplicative EV = {:+.2}%, Shin EV = {:+.2}% (phantom edge filtered)",
        cmp.for_method(DevigMethod::Multiplicative).ev_percent,
        cmp.for_method(DevigMethod::Shin).ev_percent
    );

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
    if let Some(diag) = meta.simplex_violation {
        println!(
            "   6-Way Additive Guard: outcome #{} raw prob {:.2}% clamped to {:.1}%",
            diag.worst_outcome_index + 1,
            diag.min_raw_probability * 100.0,
            clamped[diag.worst_outcome_index] * 100.0
        );
    }

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
        "   Execution: Arb +${:.2} | Dutching +${:.2} | Middle BE {:.2}% | Hedge Stake ${:.2} (+${:.2})",
        arb.guaranteed_profit,
        dutch.profit_if_hit,
        mid.break_even_probability * 100.0,
        hedge.hedge_stake,
        hedge.guaranteed_floor
    );
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
    demo_market_pricing()?;
    println!("\n✅ All 7 sports analytics modules demonstrated successfully!");
    Ok(())
}
