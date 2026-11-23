//! Converters mapping Elo rating differentials into `UnitInterval` probabilities, fair `Odds`, and point spreads.

use super::core::{EloConfig, EloRating, Venue};
use super::update::{adjusted_rating_diff, expected_score_with_config};
use crate::error::EloError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};

/// Maps two clubs' Elo ratings and venue context to the expected two-way win probability in `[0.0, 1.0]`.
#[verified_engine::verified]
pub fn elo_win_probability(
    home: EloRating,
    away: EloRating,
    venue: Venue,
    config: &EloConfig,
) -> Result<UnitInterval, EloError> {
    let p = expected_score_with_config(home, away, venue, config)?;
    UnitInterval::new(p).map_err(|_| EloError::DegenerateProbability { value: p })
}

/// Converts a single non-degenerate win probability in `(0.0, 1.0)` into fair zero-vig decimal [`Odds`].
#[verified_engine::verified]
pub fn probability_to_fair_odds(probability: UnitInterval) -> Result<Odds, EloError> {
    let p = probability.value();
    if p <= 0.0 || p >= 1.0 || !p.is_finite() {
        return Err(EloError::DegenerateProbability { value: p });
    }
    Odds::new(1.0 / p).map_err(|_| EloError::DegenerateProbability { value: p })
}

/// Maps two clubs' Elo ratings and venue context to fair two-way moneyline `(home_odds, away_odds)`.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::elo::{
///     EloConfig, EloRating, Venue, elo_moneyline_odds,
/// };
///
/// let config = EloConfig::default();
/// let (home_odds, away_odds) = elo_moneyline_odds(
///     EloRating::new(1500.0).unwrap(),
///     EloRating::new(1500.0).unwrap(),
///     Venue::Home,
///     &config,
/// ).unwrap();
/// assert!(home_odds.value() < 2.0);
/// assert!(away_odds.value() > 2.0);
/// ```
#[verified_engine::verified]
pub fn elo_moneyline_odds(
    home: EloRating,
    away: EloRating,
    venue: Venue,
    config: &EloConfig,
) -> Result<(Odds, Odds), EloError> {
    let p_home = elo_win_probability(home, away, venue, config)?;
    let p_away =
        UnitInterval::new(1.0 - p_home.value()).map_err(|_| EloError::DegenerateProbability {
            value: 1.0 - p_home.value(),
        })?;
    Ok((
        probability_to_fair_odds(p_home)?,
        probability_to_fair_odds(p_away)?,
    ))
}

/// Maps the venue-adjusted Elo differential to an expected scoring margin
/// $\Delta g = \Delta r_{\text{adj}} / \text{elo\_points\_per\_point}$.
///
/// Positive values indicate the home team is expected to outscore the away team by `spread` points/goals
/// (corresponding to a sportsbook handicap line of `-spread` on the home side).
#[verified_engine::verified]
pub fn elo_expected_margin(
    home: EloRating,
    away: EloRating,
    venue: Venue,
    config: &EloConfig,
    elo_points_per_point: f64,
) -> Result<f64, EloError> {
    if elo_points_per_point <= 0.0 || !elo_points_per_point.is_finite() {
        return Err(EloError::InvalidPointsPerElo {
            value: elo_points_per_point,
        });
    }
    let delta_r_adj = adjusted_rating_diff(home, away, venue, config.home_advantage());
    Ok(delta_r_adj / elo_points_per_point)
}

/// Maps the venue-adjusted Elo differential to the fair home sportsbook point spread
/// $-\Delta r_{\text{adj}} / \text{elo\_points\_per\_point}$ (negative when home is favored).
#[verified_engine::verified]
pub fn elo_point_spread(
    home: EloRating,
    away: EloRating,
    venue: Venue,
    config: &EloConfig,
    elo_points_per_point: f64,
) -> Result<f64, EloError> {
    Ok(-elo_expected_margin(
        home,
        away,
        venue,
        config,
        elo_points_per_point,
    )?)
}

/// Converts a fair home sportsbook point spread back into the implied venue-adjusted Elo differential.
#[verified_engine::verified]
pub fn spread_to_elo_diff(home_spread: f64, elo_points_per_point: f64) -> Result<f64, EloError> {
    if elo_points_per_point <= 0.0 || !elo_points_per_point.is_finite() {
        return Err(EloError::InvalidPointsPerElo {
            value: elo_points_per_point,
        });
    }
    if !home_spread.is_finite() {
        return Err(EloError::InvalidRating { value: home_spread });
    }
    Ok(-home_spread * elo_points_per_point)
}
