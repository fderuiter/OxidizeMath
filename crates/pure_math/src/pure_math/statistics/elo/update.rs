//! Elo expected score evaluation, win-margin multipliers, single-match updates, and seasonal regression.

use super::core::{EloConfig, EloRating, MarginScaling, Venue};
use crate::error::EloError;

/// Computes the venue-adjusted rating differential `(r_home + offset) - r_away`.
#[verified_engine::verified]
pub fn adjusted_rating_diff(
    home: EloRating,
    away: EloRating,
    venue: Venue,
    home_advantage: f64,
) -> f64 {
    let offset = match venue {
        Venue::Home => home_advantage,
        Venue::Neutral => 0.0,
    };
    (home.value() + offset) - away.value()
}

/// Computes the base-10 logistic Elo expected score
/// $E_{\text{home}} = \frac{1}{1 + 10^{-((r_{\text{home}} + h) - r_{\text{away}}) / s}}$.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::elo::{EloRating, expected_score};
///
/// let r = EloRating::new(1500.0).unwrap();
/// let e = expected_score(r, r, 65.0, 400.0).unwrap();
/// assert!((e - 0.592466).abs() < 1e-6);
/// ```
#[verified_engine::verified]
pub fn expected_score(
    home: EloRating,
    away: EloRating,
    home_advantage: f64,
    scale: f64,
) -> Result<f64, EloError> {
    if scale <= 0.0 || !scale.is_finite() {
        return Err(EloError::InvalidScale { value: scale });
    }
    if !home_advantage.is_finite() {
        return Err(EloError::InvalidHomeAdvantage {
            value: home_advantage,
        });
    }
    let delta_r_adj = (home.value() + home_advantage) - away.value();
    Ok(1.0 / (1.0 + 10.0_f64.powf(-delta_r_adj / scale)))
}

/// Computes expected home score using a validated [`EloConfig`] and [`Venue`].
#[verified_engine::verified]
pub fn expected_score_with_config(
    home: EloRating,
    away: EloRating,
    venue: Venue,
    config: &EloConfig,
) -> Result<f64, EloError> {
    let offset = match venue {
        Venue::Home => config.home_advantage(),
        Venue::Neutral => 0.0,
    };
    expected_score(home, away, offset, config.scale())
}

/// Computes the unadjusted logarithmic win-margin multiplier $\ln(|\Delta g| + 1)$ for $|\Delta g| \ge 1$
/// (returning `1.0` on draws where `goal_diff == 0`).
#[verified_engine::verified]
pub fn log_margin_multiplier(goal_diff: u32) -> f64 {
    if goal_diff == 0 {
        1.0
    } else {
        ((goal_diff as f64) + 1.0).ln()
    }
}

/// Computes the autocorrelation-adjusted win-margin multiplier
/// $\ln(|\Delta g| + 1) \cdot \frac{2.2}{|\Delta r_{\text{adj}}| \cdot 0.001 + 2.2}$
/// for $|\Delta g| \ge 1$ (returning `1.0` on draws where `goal_diff == 0`).
#[verified_engine::verified]
pub fn autocorrelated_margin_multiplier(goal_diff: u32, delta_r_adj: f64) -> f64 {
    if goal_diff == 0 {
        1.0
    } else {
        let damp = 2.2 / (delta_r_adj.abs() * 0.001 + 2.2);
        ((goal_diff as f64) + 1.0).ln() * damp
    }
}

/// Evaluates the configured [`MarginScaling`] multiplier for a fixture.
#[verified_engine::verified]
pub fn margin_multiplier(scaling: MarginScaling, goal_diff: u32, delta_r_adj: f64) -> f64 {
    match scaling {
        MarginScaling::None => 1.0,
        MarginScaling::Logarithmic => log_margin_multiplier(goal_diff),
        MarginScaling::Autocorrelated => autocorrelated_margin_multiplier(goal_diff, delta_r_adj),
    }
}

/// Computes the signed rating transfer $\Delta r_{\text{home}}$ applied to the home club
/// (the away club receives $-\Delta r_{\text{home}}$ so total rating mass is conserved).
#[verified_engine::verified]
pub fn elo_rating_delta(
    home: EloRating,
    away: EloRating,
    score_home: f64,
    goal_diff: u32,
    venue: Venue,
    config: &EloConfig,
) -> Result<f64, EloError> {
    if !(0.0..=1.0).contains(&score_home) || !score_home.is_finite() {
        return Err(EloError::InvalidScore { value: score_home });
    }
    let delta_r_adj = adjusted_rating_diff(home, away, venue, config.home_advantage());
    let expected = 1.0 / (1.0 + 10.0_f64.powf(-delta_r_adj / config.scale()));
    let mult = margin_multiplier(config.margin_scaling(), goal_diff, delta_r_adj);
    Ok(config.k_factor() * mult * (score_home - expected))
}

/// Updates `(home, away)` ratings from an explicit match score `score_home` in `[0.0, 1.0]`
/// and absolute goal/point margin `goal_diff`.
#[verified_engine::verified]
pub fn update_elo_ratings(
    home: EloRating,
    away: EloRating,
    score_home: f64,
    goal_diff: u32,
    venue: Venue,
    config: &EloConfig,
) -> Result<(EloRating, EloRating), EloError> {
    let delta = elo_rating_delta(home, away, score_home, goal_diff, venue, config)?;
    Ok((
        EloRating::new(home.value() + delta)?,
        EloRating::new(away.value() - delta)?,
    ))
}

/// Updates `(home, away)` ratings directly from integer match goals `(home_goals, away_goals)`.
#[verified_engine::verified]
pub fn update_elo_from_goals(
    home: EloRating,
    away: EloRating,
    home_goals: u32,
    away_goals: u32,
    venue: Venue,
    config: &EloConfig,
) -> Result<(EloRating, EloRating), EloError> {
    let score_home = if home_goals > away_goals {
        1.0
    } else if home_goals < away_goals {
        0.0
    } else {
        0.5
    };
    let goal_diff = home_goals.abs_diff(away_goals);
    update_elo_ratings(home, away, score_home, goal_diff, venue, config)
}

/// Applies between-season mean regression $r_{\text{next}} = (1 - \theta)\,r + \theta\,\bar{r}$
/// for $\theta \in [0.0, 1.0]$.
#[verified_engine::verified]
pub fn regress_to_mean(
    rating: EloRating,
    mean_rating: EloRating,
    theta: f64,
) -> Result<EloRating, EloError> {
    if !(0.0..=1.0).contains(&theta) || !theta.is_finite() {
        return Err(EloError::InvalidRegressionFactor { value: theta });
    }
    EloRating::new((1.0 - theta) * rating.value() + theta * mean_rating.value())
}

/// Applies between-season mean regression across a roster slice into `out_roster` with zero heap allocation.
#[verified_engine::verified]
pub fn regress_roster_into(
    roster: &[EloRating],
    mean_rating: EloRating,
    theta: f64,
    out_roster: &mut [EloRating],
) -> Result<(), EloError> {
    if out_roster.len() != roster.len() {
        return Err(EloError::DimensionMismatch {
            expected: roster.len(),
            actual: out_roster.len(),
        });
    }
    if !(0.0..=1.0).contains(&theta) || !theta.is_finite() {
        return Err(EloError::InvalidRegressionFactor { value: theta });
    }
    for (dst, &src) in out_roster.iter_mut().zip(roster.iter()) {
        *dst = EloRating::new((1.0 - theta) * src.value() + theta * mean_rating.value())?;
    }
    Ok(())
}
