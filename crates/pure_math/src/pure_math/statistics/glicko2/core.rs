//! Core types for Glicko-2 rating system.

use crate::error::Glicko2Error;
use std::f64::consts::PI;

/// Player rating (r) on the Glicko-2 scale.
///
/// The rating represents a player's skill level. By convention, new players
/// start at 1500 on the original Glicko scale.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize), serde(transparent))]
pub struct Rating(f64);

impl Rating {
    /// Creates a new rating.
    ///
    /// # Arguments
    ///
    /// * `value` - The rating value (must be finite)
    ///
    /// # Returns
    ///
    /// * `Result<Rating, Glicko2Error>` - The validated rating or an error
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::glicko2::Rating;
    ///
    /// let rating = Rating::new(1500.0).unwrap();
    /// assert_eq!(rating.value(), 1500.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, Glicko2Error> {
        if !value.is_finite() {
            return Err(Glicko2Error::InvalidRating { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw rating value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Converts from Glicko scale to Glicko-2 scale (μ).
    ///
    /// Formula: μ = (r - 1500) / 173.7178
    #[verified_engine::verified]
    pub fn to_glicko2_scale(&self) -> f64 {
        (self.0 - 1500.0) / 173.7178
    }

    /// Creates a rating from Glicko-2 scale (μ).
    ///
    /// Formula: r = 173.7178 * μ + 1500
    #[verified_engine::verified]
    pub fn from_glicko2_scale(mu: f64) -> Result<Self, Glicko2Error> {
        Self::new(173.7178 * mu + 1500.0)
    }
}

impl Default for Rating {
    /// Default rating for new players (1500 on Glicko scale).
    #[verified_engine::verified]
    fn default() -> Self {
        Self(1500.0)
    }
}

/// Rating deviation (RD) - a measure of rating uncertainty.
///
/// Higher RD means more uncertainty about the player's true skill.
/// - New players typically start with RD = 350
/// - Active players converge to RD ≈ 50-100
/// - Inactive players' RD increases over time
///
/// Must be positive.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize), serde(transparent))]
pub struct RatingDeviation(f64);

impl RatingDeviation {
    /// Creates a new rating deviation.
    ///
    /// # Arguments
    ///
    /// * `value` - The RD value (must be positive and finite)
    ///
    /// # Returns
    ///
    /// * `Result<RatingDeviation, Glicko2Error>` - The validated RD or an error
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::glicko2::RatingDeviation;
    ///
    /// let rd = RatingDeviation::new(350.0).unwrap();
    /// assert_eq!(rd.value(), 350.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, Glicko2Error> {
        if value <= 0.0 || !value.is_finite() {
            return Err(Glicko2Error::InvalidRatingDeviation { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw RD value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Converts from Glicko scale to Glicko-2 scale (φ).
    ///
    /// Formula: φ = RD / 173.7178
    #[verified_engine::verified]
    pub fn to_glicko2_scale(&self) -> f64 {
        self.0 / 173.7178
    }

    /// Creates an RD from Glicko-2 scale (φ).
    ///
    /// Formula: RD = 173.7178 * φ
    #[verified_engine::verified]
    pub fn from_glicko2_scale(phi: f64) -> Result<Self, Glicko2Error> {
        Self::new(173.7178 * phi)
    }
}

impl Default for RatingDeviation {
    /// Default RD for new players (350 on Glicko scale).
    #[verified_engine::verified]
    fn default() -> Self {
        Self(350.0)
    }
}

/// Rating volatility (σ) - indicates degree of expected rating fluctuation.
///
/// Volatility measures how consistent a player's performance is:
/// - Low σ (≈ 0.05): Consistent player with predictable performance
/// - High σ (≈ 0.15): Erratic player with variable performance
///
/// Must be positive. Typical values are between 0.3 and 1.2.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize), serde(transparent))]
pub struct Volatility(f64);

impl Volatility {
    /// Creates a new volatility parameter.
    ///
    /// # Arguments
    ///
    /// * `value` - The volatility value (must be positive and finite)
    ///
    /// # Returns
    ///
    /// * `Result<Volatility, Glicko2Error>` - The validated volatility or an error
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::glicko2::Volatility;
    ///
    /// let vol = Volatility::new(0.06).unwrap();
    /// assert_eq!(vol.value(), 0.06);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, Glicko2Error> {
        if value <= 0.0 || !value.is_finite() {
            return Err(Glicko2Error::InvalidVolatility { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw volatility value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Default for Volatility {
    /// Default volatility for new players (0.06).
    #[verified_engine::verified]
    fn default() -> Self {
        Self(0.06)
    }
}

/// System constant (τ) - constrains volatility changes.
///
/// Controls how much the volatility can change in response to unexpected results.
/// - Small τ (0.3-0.5): Conservative, volatility changes slowly
/// - Large τ (0.8-1.2): Aggressive, volatility responds quickly
///
/// Glicko-2 paper recommends τ between 0.3 and 1.2, with 0.5 as reasonable default.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize), serde(transparent))]
pub struct SystemConstant(f64);

impl SystemConstant {
    /// Creates a new system constant.
    ///
    /// # Arguments
    ///
    /// * `value` - The τ value (must be between 0.3 and 1.2)
    ///
    /// # Returns
    ///
    /// * `Result<SystemConstant, Glicko2Error>` - The validated constant or an error
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::glicko2::SystemConstant;
    ///
    /// let tau = SystemConstant::new(0.5).unwrap();
    /// assert_eq!(tau.value(), 0.5);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, Glicko2Error> {
        if !(0.3..=1.2).contains(&value) || !value.is_finite() {
            return Err(Glicko2Error::InvalidSystemConstant { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw τ value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Default for SystemConstant {
    /// Default system constant (0.5) as recommended by Glickman.
    #[verified_engine::verified]
    fn default() -> Self {
        Self(0.5)
    }
}

/// A Glicko-2 player with rating, rating deviation, and volatility.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(deny_unknown_fields)
)]
pub struct GlickoPlayer {
    /// The player's rating.
    pub rating: Rating,
    /// The rating deviation (uncertainty).
    pub rating_deviation: RatingDeviation,
    /// The volatility (consistency measure).
    pub volatility: Volatility,
}

impl GlickoPlayer {
    /// Creates a new Glicko-2 player.
    ///
    /// # Arguments
    ///
    /// * `rating` - Initial rating
    /// * `rating_deviation` - Initial rating deviation
    /// * `volatility` - Initial volatility
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::glicko2::{GlickoPlayer, Rating, RatingDeviation, Volatility};
    ///
    /// let player = GlickoPlayer::new(
    ///     Rating::new(1500.0).unwrap(),
    ///     RatingDeviation::new(200.0).unwrap(),
    ///     Volatility::new(0.06).unwrap(),
    /// );
    /// ```
    #[verified_engine::verified]
    pub fn new(rating: Rating, rating_deviation: RatingDeviation, volatility: Volatility) -> Self {
        Self {
            rating,
            rating_deviation,
            volatility,
        }
    }

    /// Converts player parameters to Glicko-2 scale (μ, φ, σ).
    ///
    /// Returns (mu, phi, sigma) tuple.
    #[verified_engine::verified]
    pub fn to_glicko2_scale(&self) -> (f64, f64, f64) {
        (
            self.rating.to_glicko2_scale(),
            self.rating_deviation.to_glicko2_scale(),
            self.volatility.value(),
        )
    }
}

impl Default for GlickoPlayer {
    /// Creates a default player (new player starting values).
    ///
    /// - Rating: 1500
    /// - RD: 350
    /// - Volatility: 0.06
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            rating: Rating::default(),
            rating_deviation: RatingDeviation::default(),
            volatility: Volatility::default(),
        }
    }
}

/// The outcome of a single match.
///
/// Score must be between 0 and 1:
/// - 1.0 = win
/// - 0.5 = draw
/// - 0.0 = loss
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MatchResult {
    /// The opponent player.
    pub opponent: GlickoPlayer,
    /// The score from this player's perspective (0 = loss, 0.5 = draw, 1 = win).
    pub score: f64,
}

impl MatchResult {
    /// Creates a new match result.
    ///
    /// # Arguments
    ///
    /// * `opponent` - The opponent player
    /// * `score` - The match score (0.0 = loss, 0.5 = draw, 1.0 = win)
    ///
    /// # Returns
    ///
    /// * `Result<MatchResult, Glicko2Error>` - The validated result or an error
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::glicko2::{MatchResult, GlickoPlayer};
    ///
    /// let opponent = GlickoPlayer::default();
    /// let result = MatchResult::new(opponent, 1.0).unwrap(); // Win
    /// ```
    #[verified_engine::verified]
    pub fn new(opponent: GlickoPlayer, score: f64) -> Result<Self, Glicko2Error> {
        if !(0.0..=1.0).contains(&score) || !score.is_finite() {
            return Err(Glicko2Error::InvalidScore { value: score });
        }
        Ok(Self { opponent, score })
    }
}

/// Computes the g(φ) function used in Glicko-2.
///
/// Formula: g(φ) = 1 / √(1 + 3φ²/π²)
///
/// This function reduces the impact of games against opponents with high uncertainty.
#[verified_engine::verified]
pub fn g_function(phi: f64) -> f64 {
    1.0 / (1.0 + 3.0 * phi * phi / (PI * PI)).sqrt()
}

/// Computes the expected outcome E(μ, μⱼ, φⱼ).
///
/// Formula: E = 1 / (1 + exp(-g(φⱼ)(μ - μⱼ)))
///
/// This is the probability that player with rating μ beats opponent with (μⱼ, φⱼ).
#[verified_engine::verified]
pub fn expected_outcome(mu: f64, mu_j: f64, phi_j: f64) -> f64 {
    1.0 / (1.0 + (-g_function(phi_j) * (mu - mu_j)).exp())
}

#[cfg(feature = "serde")]
mod serde_support {
    use super::*;

    impl<'de> serde::Deserialize<'de> for Rating {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let value = f64::deserialize(deserializer)?;
            Self::new(value).map_err(serde::de::Error::custom)
        }
    }

    impl<'de> serde::Deserialize<'de> for RatingDeviation {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let value = f64::deserialize(deserializer)?;
            Self::new(value).map_err(serde::de::Error::custom)
        }
    }

    impl<'de> serde::Deserialize<'de> for Volatility {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let value = f64::deserialize(deserializer)?;
            Self::new(value).map_err(serde::de::Error::custom)
        }
    }

    impl<'de> serde::Deserialize<'de> for SystemConstant {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let value = f64::deserialize(deserializer)?;
            Self::new(value).map_err(serde::de::Error::custom)
        }
    }

    impl<'de> serde::Deserialize<'de> for MatchResult {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct RawMatchResult {
                opponent: GlickoPlayer,
                score: f64,
            }
            let raw = RawMatchResult::deserialize(deserializer)?;
            Self::new(raw.opponent, raw.score).map_err(serde::de::Error::custom)
        }
    }
}
