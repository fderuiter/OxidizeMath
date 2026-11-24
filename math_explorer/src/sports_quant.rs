//! # Curated Quantitative Sports Modeling Facade
//!
//! Ergonomic top-level module unifying market devigging, match score-grid pricing, Gaussian
//! Copula Same-Game Parlays (SGPs), Zero-Inflated Poisson (ZIP) player props, Ornstein-Uhlenbeck
//! live momentum, Elo & Glicko-2 ratings, post-bet CLV validation, and Kelly/Lundberg solvency.
//!
//! Every probability wrapper ([`Probability`], [`BankrollFraction`], [`ZeroInflation`]) converts
//! zero-cost and infallibly to and from [`math_commons::primitives::UnitInterval`], allowing
//! priced outcomes to flow directly into [`kelly_fraction`] and [`fractional_kelly`] without
//! manual `f64` unwrapping.
//!
//! ## Example: Pricing a Correlated SGP into Fractional Kelly Sizing
//!
//! ```rust
//! use math_explorer::prelude::*;
//! use math_explorer::sports_quant::*;
//!
//! // 1. Devig a sharp two-way moneyline via Shin's insider-trading equilibrium
//! let [fair_team_win, _] = devig_two_way(
//!     Odds::new(1.75).unwrap(),
//!     Odds::new(2.15).unwrap(),
//!     DevigMethod::Shin,
//! )
//! .unwrap();
//!
//! // 2. Price a player Over 1.5 blocks prop via Zero-Inflated Poisson
//! let prop_dist = ZipDistribution::from_values(0.20, 2.10).unwrap();
//! let fair_prop_over = prop_dist.over_probability_unit(Count::new(1));
//!
//! // 3. Combine both legs in a Gaussian Copula SGP (positive correlation = +0.25)
//! let corr = CorrelationMatrix::bivariate(Correlation::new(0.25).unwrap()).unwrap();
//! let sgp_prob = sgp_joint_probability(
//!     &[fair_team_win.into(), fair_prop_over.into()],
//!     &corr,
//! )
//! .unwrap();
//!
//! // 4. Pipe the copula Probability directly into Kelly sizing via UnitInterval::from
//! let offered_parlay_odds = Odds::new(4.20).unwrap();
//! let sgp_unit: UnitInterval = sgp_prob.into();
//! let half_kelly = fractional_kelly(&sgp_unit, &offered_parlay_odds, 0.5).unwrap();
//! assert!(half_kelly.value() > 0.0);
//! ```

pub use pure_math::pure_math::statistics::{
    copula, elo, glicko2, kelly, market_pricing, match_models, ou_process, ruin, validation,
    zip_regression,
};

pub use copula::{
    Correlation, CorrelationMatrix, GaussianCopula, MAX_STACK_LEGS, ParlayPricingReport,
    Probability, RHO_SPLIT, bivariate_copula_cdf, bivariate_normal_cdf,
    parlay_simulation_scratch_len, price_correlated_parlay, price_correlated_parlay_with_scratch,
    sgp_joint_probability, sgp_joint_probability_with_scratch, simulate_correlated_parlay,
    simulate_correlated_parlay_into,
};
pub use elo::{
    EloBatchMatch, EloConfig, EloKFactor, EloRating, HomeAdvantage, MarginScaling, Venue,
    elo_expected_score, elo_moneyline_odds, elo_mov_multiplier, elo_point_spread,
    elo_update_rating, elo_update_with_mov, elo_win_probability, evaluate_elo_batch,
    evaluate_elo_sequential, expected_score_with_config, regress_to_mean, update_elo_from_goals,
};
pub use glicko2::{
    GlickoPlayer, MatchResult, Pairing, PeriodMatch, Rating, RatingDeviation, SwissRoundPlan,
    SystemConstant, Volatility, evaluate_rating_period, head_to_head_win_probability,
    head_to_head_win_probability_unit, pairwise_win_matrix, update_rating,
};
pub use kelly::{
    BankrollFraction, DrawdownConstraint, EXACT_CONVOLUTION_MAX_WAGERS, LAPLACE_QUADRATURE_NODES,
    MAX_CONVOLUTION_STATES, MAX_SLATE_WAGERS, MarketAllocation, MarketOutcome, MarketScratch, Odds,
    PROBABILITY_SUM_TOLERANCE, PushOutcome, PushWager, QuarterLineOutcome, QuarterLineWager,
    SlateEvalMode, SlateScratch, SlateSolution, SlateWager, UnitInterval,
    drawdown_constrained_kelly, drawdown_constrained_push_kelly,
    drawdown_constrained_quarter_line_kelly, expected_growth_rate, expected_value,
    fractional_kelly, kelly_drawdown_risk, kelly_fraction, push_expected_growth_rate,
    push_expected_value, push_kelly_fraction, quarter_line_expected_growth_rate,
    quarter_line_expected_value, quarter_line_growth_derivative, quarter_line_kelly_fraction,
    solve_drawdown_constrained_slate, solve_drawdown_constrained_slate_with_mode,
    solve_mutually_exclusive_market, solve_simultaneous_slate, solve_simultaneous_slate_with_mode,
    unit_risk_of_ruin, variants as kelly_variants,
};
pub use market_pricing::{
    AllMethodsEvReport, ArbitrageReport, DevigMetadata, DevigMethod, DutchingMode, DutchingReport,
    EvEvaluation, HedgeMode, HedgeReport, MAX_MARKET_OUTCOMES, MarketOverround, MarketPricingError,
    MiddleReport, SimplexViolationDiagnostic, book_vigorish, booksum, closing_line_edge_two_way,
    closing_line_value, compare_outcome_ev_all_methods, devig_additive_detailed_into,
    devig_additive_into, devig_into, devig_multiplicative_into, devig_power_into, devig_shin_into,
    devig_three_way, devig_two_way, devig_unit_into, dutch_outcomes_into, dutch_subset_into,
    evaluate_arbitrage_into, evaluate_ev_from_probability, evaluate_hedge, evaluate_middle,
    evaluate_middle_custom_stakes, evaluate_outcome_ev, evaluate_two_way_arbitrage,
    fair_decimal_odds_into, fair_odds, fair_odds_from_probabilities_into,
    implied_probabilities_into, is_arbitrage, market_overround, overround, turnover_margin,
};
pub use match_models::{
    AsianHandicapQuote, BivariatePoissonParams, BttsQuote, DixonColesMleConfig, DixonColesParams,
    DixonColesRho, DrawNoBetQuote, FittedDixonColes, HistoricalMatch, IndependentPoissonParams,
    MatchOutcomeProbs, OneXTwoQuote, OverUnderQuote, ScoreGrid, ScoreModelKind, SharedGoalRate,
    asian_handicap, asian_handicap_home_cover, bivariate_poisson_grid,
    bivariate_poisson_matrix_into, bivariate_poisson_pmf, both_teams_to_score_probability, btts,
    dixon_coles_grid, dixon_coles_matrix_into, dixon_coles_rho_bounds, dixon_coles_tau,
    draw_no_bet, fit_dixon_coles_mle, independent_poisson_grid, independent_poisson_matrix_into,
    match_1x2_probabilities, one_x_two, over_under, over_under_probabilities,
};
pub use ou_process::{EulerMaruyama, MomentumType, OuAnalyzer, OuParams, TimeStep};
pub use ruin::{
    BankrollEnsembleWorkspace, DrawdownQuantiles, EnsembleSolvencyReport, GamblersRuinResult,
    KellyBarrierRisk, LundbergSolution, PortfolioBet, ProportionalPortfolioBet, WilsonInterval,
    gamblers_ruin_infinite, gamblers_ruin_with_target, kelly_barrier_risk,
    portfolio_kelly_barrier_risk, solve_lundberg_ruin, solve_single_bet_lundberg,
};
pub use validation::{
    CalibrationComparison, ClvMetrics, ClvSummary, SampleSize95, SettledBet, YieldSignificance,
    binary_log_loss, brier_score, compare_calibration, evaluate_yield_against_closing,
    evaluate_yield_significance, required_bets_for_95_significance, settled_closing_calibration,
    single_bet_clv, single_bet_clv_two_way, summarize_clv,
};
pub use zip_regression::{
    Count, PoissonRate, ZeroInflation, ZipDistribution, ZipParams, simple_zip_fit,
};
