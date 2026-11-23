//! `UnitInterval`-typed devigging bridges, fair-odds conversion, and Closing Line Value (CLV).

use super::MAX_MARKET_OUTCOMES;
use super::devig::{DevigMetadata, DevigMethod, devig_into};
use super::overround::turnover_margin;
use crate::error::MarketPricingError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};

/// Computes the bookmaker vigorish (hold percentage) $\nu = \max(0, 1 - 1 / S)$ bounded in [`UnitInterval`].
#[verified_engine::verified]
pub fn book_vigorish(odds: &[Odds]) -> Result<UnitInterval, MarketPricingError> {
    let margin = turnover_margin(odds)?;
    if margin <= 0.0 {
        return Ok(UnitInterval::new_clamped(0.0));
    }
    Ok(UnitInterval::new_clamped(margin))
}

/// Dispatches [`DevigMethod`] into a [`UnitInterval`] slice without heap allocation.
#[verified_engine::verified]
pub fn devig_unit_into(
    odds: &[Odds],
    method: DevigMethod,
    out_probs: &mut [UnitInterval],
) -> Result<DevigMetadata, MarketPricingError> {
    let n = odds.len();
    if n > MAX_MARKET_OUTCOMES {
        return Err(MarketPricingError::TooManyOutcomes {
            count: n,
            max_supported: MAX_MARKET_OUTCOMES,
        });
    }
    if out_probs.len() != n {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: n,
            actual: out_probs.len(),
        });
    }
    let mut raw = [0.0_f64; MAX_MARKET_OUTCOMES];
    let meta = devig_into(odds, method, &mut raw[..n])?;
    let mut i = 0;
    while i < n {
        out_probs[i] = UnitInterval::new_clamped(raw[i]);
        i += 1;
    }
    Ok(meta)
}

/// Devigs a two-way (head-to-head / Over-Under / spread) market on the stack.
#[verified_engine::verified]
pub fn devig_two_way(
    home: Odds,
    away: Odds,
    method: DevigMethod,
) -> Result<[UnitInterval; 2], MarketPricingError> {
    let odds = [home, away];
    let z = UnitInterval::new_clamped(0.5);
    let mut probs = [z, z];
    devig_unit_into(&odds, method, &mut probs)?;
    Ok(probs)
}

/// Devigs a three-way (1X2) market on the stack.
#[verified_engine::verified]
pub fn devig_three_way(
    home: Odds,
    draw: Odds,
    away: Odds,
    method: DevigMethod,
) -> Result<[UnitInterval; 3], MarketPricingError> {
    let odds = [home, draw, away];
    let z = UnitInterval::new_clamped(1.0 / 3.0);
    let mut probs = [z, z, z];
    devig_unit_into(&odds, method, &mut probs)?;
    Ok(probs)
}

/// Converts a fair probability $p \in (0, 1)$ into zero-vigorish decimal [`Odds`] ($1 / p$).
#[verified_engine::verified]
pub fn fair_odds(prob: UnitInterval) -> Result<Odds, MarketPricingError> {
    let p = prob.value();
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(MarketPricingError::InvalidProbability { value: p });
    }
    Odds::new(1.0 / p).map_err(|_| MarketPricingError::InvalidProbability { value: p })
}

/// Computes Closing Line Value (CLV) expected return $o_{\text{bet}} \cdot p_{\text{close}} - 1$
/// given placed ticket `bet_odds` and devigged closing probability `closing_fair_prob`.
#[verified_engine::verified]
pub fn closing_line_value(bet_odds: Odds, closing_fair_prob: UnitInterval) -> f64 {
    bet_odds.value() * closing_fair_prob.value() - 1.0
}

/// Computes two-way Closing Line Value directly against closing `closing_side` and `closing_other`
/// bookmaker lines devigged via `method`.
#[verified_engine::verified]
pub fn closing_line_edge_two_way(
    bet_odds: Odds,
    closing_side: Odds,
    closing_other: Odds,
    method: DevigMethod,
) -> Result<(f64, UnitInterval), MarketPricingError> {
    let [fair_side, _] = devig_two_way(closing_side, closing_other, method)?;
    Ok((closing_line_value(bet_odds, fair_side), fair_side))
}

// [cite:sports_quantitative_modeling]
