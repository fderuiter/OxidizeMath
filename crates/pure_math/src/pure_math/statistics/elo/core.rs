//! Core validated primitives and configuration types for the sports Elo rating system.

use crate::error::EloError;
use serde::{Deserialize, Deserializer, Serialize};

/// Default Elo rating for newly initialized clubs or competitors.
pub const DEFAULT_ELO_RATING: f64 = 1500.0;

/// Default base-10 logistic scale parameter (`400.0`).
pub const DEFAULT_ELO_SCALE: f64 = 400.0;

/// Default match update K-factor (`20.0`).
pub const DEFAULT_ELO_K_FACTOR: f64 = 20.0;

/// Default home-venue rating advantage (`65.0` Elo points).
pub const DEFAULT_HOME_ADVANTAGE: f64 = 65.0;

/// Validated finite club or competitor Elo rating.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct EloRating(f64);

impl EloRating {
    /// Creates a validated finite Elo rating.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::elo::EloRating;
    ///
    /// let rating = EloRating::new(1500.0).unwrap();
    /// assert_eq!(rating.value(), 1500.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, EloError> {
        if !value.is_finite() {
            return Err(EloError::InvalidRating { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw floating-point Elo rating.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Default for EloRating {
    #[verified_engine::verified]
    fn default() -> Self {
        Self(DEFAULT_ELO_RATING)
    }
}

impl<'de> Deserialize<'de> for EloRating {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Match venue context controlling whether `home_advantage` is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Venue {
    /// Standard home-and-away fixture where the first team receives `home_advantage`.
    #[default]
    Home,
    /// Neutral-site fixture where no venue offset is applied.
    Neutral,
}

/// Win-margin (Margin of Victory) multiplier rule applied to Elo updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MarginScaling {
    /// Classical binary/ternary Elo update with unit multiplier (`1.0`).
    #[default]
    None,
    /// Unadjusted logarithmic win-margin multiplier `ln(|delta_goals| + 1)` for `|delta_goals| >= 1` (`1.0` on draws).
    Logarithmic,
    /// Autocorrelation-adjusted win-margin multiplier
    /// `ln(|delta_goals| + 1) * (2.2 / (|delta_r_adj| * 0.001 + 2.2))` for `|delta_goals| >= 1` (`1.0` on draws).
    Autocorrelated,
}

/// Validated hyperparameters governing Elo expected scores and post-match updates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EloConfig {
    scale: f64,
    k_factor: f64,
    home_advantage: f64,
    margin_scaling: MarginScaling,
}

impl EloConfig {
    /// Creates a validated [`EloConfig`].
    ///
    /// - `scale` must be finite and strictly positive (typically `400.0`).
    /// - `k_factor` must be finite and strictly positive (typically `20.0` to `32.0`).
    /// - `home_advantage` must be finite (typically `50.0` to `100.0`, or `0.0`).
    #[verified_engine::verified]
    pub fn new(
        scale: f64,
        k_factor: f64,
        home_advantage: f64,
        margin_scaling: MarginScaling,
    ) -> Result<Self, EloError> {
        if scale <= 0.0 || !scale.is_finite() {
            return Err(EloError::InvalidScale { value: scale });
        }
        if k_factor <= 0.0 || !k_factor.is_finite() {
            return Err(EloError::InvalidKFactor { value: k_factor });
        }
        if !home_advantage.is_finite() {
            return Err(EloError::InvalidHomeAdvantage {
                value: home_advantage,
            });
        }
        Ok(Self {
            scale,
            k_factor,
            home_advantage,
            margin_scaling,
        })
    }

    /// Returns a copy of this configuration with a new [`MarginScaling`] mode.
    #[verified_engine::verified]
    pub fn with_margin_scaling(mut self, margin_scaling: MarginScaling) -> Self {
        self.margin_scaling = margin_scaling;
        self
    }

    /// Logistic rating scale (`400.0` by default).
    #[verified_engine::verified]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Update step size $K$ (`20.0` by default).
    #[verified_engine::verified]
    pub fn k_factor(&self) -> f64 {
        self.k_factor
    }

    /// Home-venue rating offset (`65.0` by default).
    #[verified_engine::verified]
    pub fn home_advantage(&self) -> f64 {
        self.home_advantage
    }

    /// Active win-margin multiplier mode.
    #[verified_engine::verified]
    pub fn margin_scaling(&self) -> MarginScaling {
        self.margin_scaling
    }
}

impl Default for EloConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            scale: DEFAULT_ELO_SCALE,
            k_factor: DEFAULT_ELO_K_FACTOR,
            home_advantage: DEFAULT_HOME_ADVANTAGE,
            margin_scaling: MarginScaling::None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEloConfig {
    scale: f64,
    k_factor: f64,
    home_advantage: f64,
    margin_scaling: MarginScaling,
}

impl<'de> Deserialize<'de> for EloConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawEloConfig::deserialize(deserializer)?;
        Self::new(
            raw.scale,
            raw.k_factor,
            raw.home_advantage,
            raw.margin_scaling,
        )
        .map_err(serde::de::Error::custom)
    }
}
