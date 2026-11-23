//! # Curated Quantitative Sports Modeling Facade
//!
//! Ergonomic top-level module unifying market devigging, match score-grid pricing, Gaussian
//! Copula Same-Game Parlays (SGPs), Zero-Inflated Poisson (ZIP) player props, Ornstein-Uhlenbeck
//! live momentum, Elo & Glicko-2 ratings, and Kelly Criterion bankroll sizing.
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
    copula, elo, glicko2, kelly, market_pricing, match_models, ou_process, zip_regression,
};

pub use copula::{
    Correlation, CorrelationMatrix, GaussianCopula, Probability, sgp_joint_probability,
};
pub use elo::{
    EloKFactor, EloRating, HomeAdvantage, elo_expected_score, elo_mov_multiplier,
    elo_update_rating, elo_update_with_mov,
};
pub use glicko2::{
    GlickoPlayer, MatchResult, Pairing, PeriodMatch, Rating, RatingDeviation, SwissRoundPlan,
    SystemConstant, Volatility, evaluate_rating_period, head_to_head_win_probability,
    head_to_head_win_probability_unit, pairwise_win_matrix, update_rating,
};
pub use kelly::{
    BankrollFraction, Odds, UnitInterval, expected_growth_rate, expected_value, fractional_kelly,
    kelly_drawdown_risk, kelly_fraction, unit_risk_of_ruin, variants as kelly_variants,
};
pub use market_pricing::{
    DevigMethod, book_overround, book_vigorish, closing_line_edge_two_way, closing_line_value,
    devig_additive_into, devig_into, devig_multiplicative_into, devig_power_into, devig_shin_into,
    devig_three_way, devig_two_way, fair_odds,
};
pub use match_models::{
    DixonColesRho, MatchOutcomeProbs, ScoreModelKind, SharedGoalRate, asian_handicap_home_cover,
    bivariate_poisson_matrix_into, bivariate_poisson_pmf, both_teams_to_score_probability,
    dixon_coles_matrix_into, dixon_coles_tau, independent_poisson_matrix_into,
    match_1x2_probabilities, over_under_probabilities,
};
pub use ou_process::{EulerMaruyama, MomentumType, OuAnalyzer, OuParams, TimeStep};
pub use zip_regression::{
    Count, PoissonRate, ZeroInflation, ZipDistribution, ZipParams, simple_zip_fit,
};
