//! Comprehensive demonstration of all sports analytics modules.

use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::{
    copula::{Correlation, CorrelationMatrix, Probability, sgp_joint_probability},
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

fn demo_zip_regression() {
    println!("1. ZIP REGRESSION - Player Block Modeling");
    let zip_params = ZipParams::from_values(0.25, 1.5).unwrap();
    let zip_dist = ZipDistribution::new(zip_params);
    println!("   Zero-inflation: 25%, Poisson rate: 1.5");
    println!("   P(0 blocks) = {:.3}", zip_dist.pmf(Count::new(0)));
    println!(
        "   Mean: {:.2}, Variance: {:.2} (overdispersed!)",
        zip_dist.mean(),
        zip_dist.variance()
    );
}

fn demo_ou_process() {
    println!("\n2. OU PROCESS - Shooting Percentage Momentum");
    let ou_params = OuParams::from_values(0.45, 1.0, 0.15).unwrap();
    let dt = TimeStep::new(0.01).unwrap();
    let solver = EulerMaruyama::new(ou_params, dt);
    let mut rng = OxidizeRng::default();
    let trajectory = solver.simulate(0.60, 100, &mut rng);
    println!("   True skill: 45%, Currently: 60% (hot)");
    println!(
        "   After 100 possessions: {:.1}%",
        trajectory.last().unwrap() * 100.0
    );
}

fn demo_copula() {
    println!("\n3. GAUSSIAN COPULA - Same Game Parlay");
    let p_star_50 = Probability::new(0.99).unwrap();
    let p_win = Probability::new(0.60).unwrap();
    let rho = Correlation::new(-0.15).unwrap();
    let corr_matrix = CorrelationMatrix::bivariate(rho).unwrap();
    let joint = sgp_joint_probability(&[p_star_50, p_win], &corr_matrix).unwrap();
    println!("   Event A: Star scores 50+ (99th percentile)");
    println!("   Event B: Team wins (60% base)");
    println!("   Naive: {:.4}, Copula: {:.4}", 0.99 * 0.60, joint.value());
}

fn demo_glicko2() {
    println!("\n4. GLICKO-2 RATING - Team Ranking");
    let mut player = GlickoPlayer::new(
        Rating::new(1500.0).unwrap(),
        RatingDeviation::new(200.0).unwrap(),
        Volatility::new(0.06).unwrap(),
    );
    let opponent_rating = Rating::new(1400.0).unwrap();
    let opponent_rd = RatingDeviation::new(30.0).unwrap();
    let opponent = GlickoPlayer::new(opponent_rating, opponent_rd, Volatility::default());
    let results = vec![MatchResult::new(opponent, 1.0).unwrap()];
    let tau = SystemConstant::new(0.5).unwrap();
    player = update_rating(&player, &results, &tau).unwrap();
    println!("   Before: R=1500, RD=200");
    println!(
        "   After win: R={:.0}, RD={:.0}",
        player.rating.value(),
        player.rating_deviation.value()
    );
}

fn demo_kelly() {
    println!("\n5. KELLY CRITERION - Bet Sizing");
    let prob = UnitInterval::new(0.55).unwrap();
    let odds = Odds::new(2.0).unwrap();
    let kelly = kelly_fraction(&prob, &odds).unwrap();
    println!("   Win probability: 55%, Odds: 2.0");
    println!("   Full Kelly: {:.1}%", kelly.value() * 100.0);
    println!(
        "   Quarter Kelly: {:.1}% (recommended)",
        kelly.value() * 25.0
    );
}

fn demo_markov_chain() {
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
    let chain = MarkovChain::new(p, states).unwrap();
    let rewards = DVector::from_vec(vec![3.0, 0.0]);
    let epv = chain.expected_possession_value(&rewards).unwrap();
    println!("   State 0 EPV: {:.2} points", epv[0]);
    println!("   State 1 EPV: {:.2} points", epv[1]);
}

fn main() {
    println!("=== Sports Analytics Framework Demo ===\n");
    demo_zip_regression();
    demo_ou_process();
    demo_copula();
    demo_glicko2();
    demo_kelly();
    demo_markov_chain();
    println!("\n✅ All 6 sports analytics modules demonstrated successfully!");
}
