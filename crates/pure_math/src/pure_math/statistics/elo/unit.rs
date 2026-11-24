//! Unit-interval and strongly-typed convenience wrappers for Elo rating expectations and MoV updates.

use super::core::{DEFAULT_ELO_SCALE, EloRating};
use super::update::expected_score;
use crate::error::EloError;
use math_commons::primitives::UnitInterval;
use serde::{Deserialize, Serialize};

/// Elo update sensitivity factor $K > 0$ (typical sports range `16.0..=40.0`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EloKFactor(f64);

impl EloKFactor {
    /// Creates a strictly positive finite [`EloKFactor`].
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, EloError> {
        if value <= 0.0 || !value.is_finite() {
            return Err(EloError::InvalidKFactor { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw $K$-factor value.
    #[verified_engine::verified]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for EloKFactor {
    #[verified_engine::verified]
    fn default() -> Self {
        Self(24.0)
    }
}

/// Additive Elo home-field advantage $H$ in Elo points (e.g. `50.0..=70.0` in football/NBA).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HomeAdvantage(f64);

impl HomeAdvantage {
    /// Creates a finite [`HomeAdvantage`] offset in Elo points.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, EloError> {
        if !value.is_finite() {
            return Err(EloError::InvalidHomeAdvantage { value });
        }
        Ok(Self(value))
    }

    /// Zero neutral-site home advantage (`0.0`).
    #[verified_engine::verified]
    pub fn neutral() -> Self {
        Self(0.0)
    }

    /// Returns the raw Elo home-advantage offset.
    #[verified_engine::verified]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for HomeAdvantage {
    #[verified_engine::verified]
    fn default() -> Self {
        Self(0.0)
    }
}

/// Computes the Elo expected score (win probability in binary markets) for side A:
///
/// $$E_A = \frac{1}{1 + 10^{-(R_A + H - R_B)/400}}$$
#[verified_engine::verified]
pub fn elo_expected_score(
    rating_a: EloRating,
    rating_b: EloRating,
    home_advantage: HomeAdvantage,
) -> UnitInterval {
    let prob =
        expected_score(rating_a, rating_b, home_advantage.0, DEFAULT_ELO_SCALE).unwrap_or(0.5);
    UnitInterval::new_clamped(prob)
}

/// Updates `(rating_a, rating_b)` after observing `actual_score_a` (`1.0` = win, `0.5` = draw, `0.0` = loss)
/// while conserving total rating points ($R_A' + R_B' = R_A + R_B$).
#[verified_engine::verified]
pub fn elo_update_rating(
    rating_a: EloRating,
    rating_b: EloRating,
    actual_score_a: UnitInterval,
    k_factor: EloKFactor,
    home_advantage: HomeAdvantage,
) -> (EloRating, EloRating) {
    let expected_a = elo_expected_score(rating_a, rating_b, home_advantage).value();
    let delta = k_factor.0 * (actual_score_a.value() - expected_a);
    let next_a = EloRating::new(rating_a.value() + delta).unwrap_or(rating_a);
    let next_b = EloRating::new(rating_b.value() - delta).unwrap_or(rating_b);
    (next_a, next_b)
}

/// Computes the autocorrelation-corrected logarithmic Margin-of-Victory (MoV) multiplier:
///
/// $$M(\Delta S, \Delta R_{\text{winner}}) = \ln(|\Delta S| + 1) \cdot \frac{2.2}{\max(\Delta R_{\text{winner}}, -1800.0) \cdot 0.001 + 2.2}$$
///
/// Returns `1.0` for draws (`score_diff == 0`).
#[verified_engine::verified]
pub fn elo_mov_multiplier(score_diff: u32, winner_elo_lead: f64) -> f64 {
    if score_diff == 0 {
        return 1.0;
    }
    let clamped_lead = if winner_elo_lead.is_finite() {
        winner_elo_lead.clamp(-1800.0, 4000.0)
    } else {
        0.0
    };
    let autocorr = 2.2 / (clamped_lead * 0.001 + 2.2);
    ((score_diff as f64) + 1.0).ln() * autocorr
}

/// Updates `(rating_a, rating_b)` incorporating Margin-of-Victory scaling `score_diff`.
#[verified_engine::verified]
pub fn elo_update_with_mov(
    rating_a: EloRating,
    rating_b: EloRating,
    actual_score_a: UnitInterval,
    score_diff: u32,
    k_factor: EloKFactor,
    home_advantage: HomeAdvantage,
) -> (EloRating, EloRating) {
    let expected_a = elo_expected_score(rating_a, rating_b, home_advantage).value();
    let eff_diff_a = rating_a.value() + home_advantage.0 - rating_b.value();
    let winner_lead = if actual_score_a.value() > 0.5 {
        eff_diff_a
    } else if actual_score_a.value() < 0.5 {
        -eff_diff_a
    } else {
        0.0
    };
    let mult = elo_mov_multiplier(score_diff, winner_lead);
    let delta = k_factor.0 * mult * (actual_score_a.value() - expected_a);
    let next_a = EloRating::new(rating_a.value() + delta).unwrap_or(rating_a);
    let next_b = EloRating::new(rating_b.value() - delta).unwrap_or(rating_b);
    (next_a, next_b)
}
