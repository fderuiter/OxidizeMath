//! Bookmaker booksum, overround, and turnover margin metrics over `&[Odds]` slices.

use super::MAX_MARKET_OUTCOMES;
use crate::error::MarketPricingError;
use crate::pure_math::statistics::kelly::Odds;

/// Summary of raw implied probability sum and bookmaker margin for an `n`-way market.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketOverround {
    /// Number of mutually exclusive outcomes in the market.
    pub outcome_count: usize,
    /// Sum of raw implied probabilities `sum_{i=1}^n (1 / O_i)`.
    pub booksum: f64,
    /// Excess implied probability over unity (`booksum - 1.0`).
    pub overround: f64,
    /// Theoretical bookmaker hold per unit wagered (`(booksum - 1.0) / booksum`).
    pub turnover_margin: f64,
}

#[verified_engine::verified]
pub(super) fn validate_odds_slice(
    odds: &[Odds],
    min_required: usize,
) -> Result<(), MarketPricingError> {
    let count = odds.len();
    if count < min_required {
        return Err(MarketPricingError::InsufficientOutcomes {
            count,
            min_required,
        });
    }
    if count > MAX_MARKET_OUTCOMES {
        return Err(MarketPricingError::TooManyOutcomes {
            count,
            max_supported: MAX_MARKET_OUTCOMES,
        });
    }
    let mut i = 0;
    while i < count {
        let val = odds[i].value();
        if !val.is_finite() || val <= 1.0 {
            return Err(MarketPricingError::InvalidOdds { value: val });
        }
        i += 1;
    }
    Ok(())
}

/// Computes the raw implied probability sum (`sum(1.0 / O_i)`) for a multi-outcome market.
#[verified_engine::verified]
pub fn booksum(odds: &[Odds]) -> Result<f64, MarketPricingError> {
    validate_odds_slice(odds, 2)?;
    let mut sum = 0.0;
    let mut i = 0;
    while i < odds.len() {
        sum += odds[i].implied_probability();
        i += 1;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(MarketPricingError::NonPositiveOverround { booksum: sum });
    }
    Ok(sum)
}

/// Computes the excess bookmaker overround (`booksum - 1.0`).
#[verified_engine::verified]
pub fn overround(odds: &[Odds]) -> Result<f64, MarketPricingError> {
    Ok(booksum(odds)? - 1.0)
}

/// Computes the bookmaker turnover margin (`1.0 - 1.0 / booksum`).
#[verified_engine::verified]
pub fn turnover_margin(odds: &[Odds]) -> Result<f64, MarketPricingError> {
    let s = booksum(odds)?;
    Ok((s - 1.0) / s)
}

/// Computes [`MarketOverround`] (`booksum`, `overround`, and `turnover_margin`) in a single pass.
#[verified_engine::verified]
pub fn market_overround(odds: &[Odds]) -> Result<MarketOverround, MarketPricingError> {
    let s = booksum(odds)?;
    Ok(MarketOverround {
        outcome_count: odds.len(),
        booksum: s,
        overround: s - 1.0,
        turnover_margin: (s - 1.0) / s,
    })
}

/// Writes raw implied probabilities (`1.0 / O_i`) into `probs_out` and returns [`MarketOverround`].
#[verified_engine::verified]
pub fn implied_probabilities_into(
    odds: &[Odds],
    probs_out: &mut [f64],
) -> Result<MarketOverround, MarketPricingError> {
    validate_odds_slice(odds, 2)?;
    if probs_out.len() != odds.len() {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: odds.len(),
            actual: probs_out.len(),
        });
    }
    let mut sum = 0.0;
    let mut i = 0;
    while i < odds.len() {
        let pi = odds[i].implied_probability();
        probs_out[i] = pi;
        sum += pi;
        i += 1;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(MarketPricingError::NonPositiveOverround { booksum: sum });
    }
    Ok(MarketOverround {
        outcome_count: odds.len(),
        booksum: sum,
        overround: sum - 1.0,
        turnover_margin: (sum - 1.0) / sum,
    })
}

// [cite:clinical_trials_statistics]
