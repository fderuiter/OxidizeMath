//! # Sports Elo Rating System & Fair Market Converters
//!
//! Venue-adjusted sports Elo rating system with win-margin multipliers, between-season
//! mean regression, zero-allocation frozen-period batch updates, [`math_commons::primitives::UnitInterval`]
//! expectation wrappers, and fair moneyline/spread converters.
//!
//! ## Mathematical Framework
//!
//! Given club ratings $(r_{\text{home}}, r_{\text{away}})$, home-venue offset $h$, and logistic scale $s$
//! (default $400.0$), the venue-adjusted rating differential is $\Delta r_{\text{adj}} = (r_{\text{home}} + h) - r_{\text{away}}$
//! (with $h = 0$ on [`Venue::Neutral`]) and the expected home match score is:
//!
//! ```text
//! E_home = 1 / (1 + 10^(-delta_r_adj / s))
//! ```
//!
//! Post-match updates conserve rating sum via $\Delta r_{\text{home}} = K \cdot M \cdot (S_{\text{home}} - E_{\text{home}})$,
//! where $M$ is `1.0` ([`MarginScaling::None`]), $\ln(|\Delta g| + 1)$ ([`MarginScaling::Logarithmic`]),
//! or the autocorrelation-adjusted multiplier:
//!
//! ```text
//! M = ln(|delta_goals| + 1) * (2.2 / (|delta_r_adj| * 0.001 + 2.2))
//! ```
//!
//! ## Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::primitives::UnitInterval;
//! use math_explorer::pure_math::statistics::elo::{
//!     EloConfig, EloKFactor, EloRating, HomeAdvantage, MarginScaling, Venue,
//!     elo_expected_score, elo_moneyline_odds, elo_update_rating, expected_score_with_config,
//!     update_elo_from_goals,
//! };
//!
//! let config = EloConfig::default().with_margin_scaling(MarginScaling::Logarithmic);
//! let home = EloRating::new(1500.0).unwrap();
//! let away = EloRating::new(1500.0).unwrap();
//!
//! let expected = expected_score_with_config(home, away, Venue::Home, &config).unwrap();
//! assert!((expected - 0.592466).abs() < 1e-6);
//!
//! let (new_home, new_away) = update_elo_from_goals(home, away, 1, 0, Venue::Home, &config).unwrap();
//! assert!(((new_home.value() - 1500.0) - 5.6496).abs() < 1e-4);
//! assert!(((1500.0 - new_away.value()) - 5.6496).abs() < 1e-4);
//!
//! let (home_odds, away_odds) = elo_moneyline_odds(new_home, new_away, Venue::Home, &config).unwrap();
//! assert!(home_odds.value() < away_odds.value());
//!
//! let hfa = HomeAdvantage::new(55.0).unwrap();
//! let win_prob = elo_expected_score(EloRating::new(1600.0).unwrap(), away, hfa);
//! assert!(win_prob.value() > 0.65);
//! let (updated_h, _) = elo_update_rating(
//!     home,
//!     away,
//!     UnitInterval::new(1.0).unwrap(),
//!     EloKFactor::new(24.0).unwrap(),
//!     hfa,
//! );
//! assert!(updated_h.value() > home.value());
//! ```

pub mod batch;
#[doc(hidden)]
pub mod core;
pub mod markets;
pub mod unit;
pub mod update;

pub use batch::{
    EloBatchMatch, evaluate_elo_batch, evaluate_elo_batch_into, evaluate_elo_sequential,
    evaluate_elo_sequential_into,
};
pub use core::{
    DEFAULT_ELO_K_FACTOR, DEFAULT_ELO_RATING, DEFAULT_ELO_SCALE, DEFAULT_HOME_ADVANTAGE, EloConfig,
    EloRating, MarginScaling, Venue,
};
pub use markets::{
    elo_expected_margin, elo_moneyline_odds, elo_point_spread, elo_win_probability,
    probability_to_fair_odds, spread_to_elo_diff,
};
pub use unit::{
    EloKFactor, HomeAdvantage, elo_expected_score, elo_mov_multiplier, elo_update_rating,
    elo_update_with_mov,
};
pub use update::{
    adjusted_rating_diff, autocorrelated_margin_multiplier, elo_rating_delta, expected_score,
    expected_score_with_config, log_margin_multiplier, margin_multiplier, regress_roster_into,
    regress_to_mean, update_elo_from_goals, update_elo_ratings,
};

// [cite:sports_quantitative_modeling]

#[cfg(test)]
mod tests;
