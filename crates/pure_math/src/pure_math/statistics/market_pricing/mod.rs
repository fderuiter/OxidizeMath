//! # Sports Market Pricing, Devigging & Closing Line Value
//!
//! Zero-allocation vigorish removal (Multiplicative, Additive, Power, and Shin insider-trading
//! equilibrium), fair odds conversion, and Closing Line Value (CLV) edge metrics.
//!
//! ## Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{Odds, kelly_fraction};
//! use math_explorer::pure_math::statistics::market_pricing::{DevigMethod, devig_two_way};
//!
//! let sharp_home = Odds::new(1.85).unwrap();
//! let sharp_away = Odds::new(2.05).unwrap();
//! let [fair_home, _fair_away] = devig_two_way(sharp_home, sharp_away, DevigMethod::Shin).unwrap();
//!
//! let soft_book_home = Odds::new(2.05).unwrap();
//! let stake = kelly_fraction(&fair_home, &soft_book_home).unwrap();
//! assert!(stake.value() > 0.0);
//! ```

pub mod devig;

pub use devig::{
    DevigMethod, book_overround, book_vigorish, devig_additive_into, devig_multiplicative_into,
    devig_power_into, devig_shin_into,
};

use crate::error::KellyError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};

/// Dispatches the requested [`DevigMethod`] into `out_probs` without heap allocation.
#[verified_engine::verified]
pub fn devig_into(
    odds: &[Odds],
    method: DevigMethod,
    out_probs: &mut [UnitInterval],
) -> Result<(), KellyError> {
    match method {
        DevigMethod::Multiplicative => {
            devig_multiplicative_into(odds, out_probs)?;
        }
        DevigMethod::Additive => {
            devig_additive_into(odds, out_probs)?;
        }
        DevigMethod::Power => {
            devig_power_into(odds, out_probs)?;
        }
        DevigMethod::Shin => {
            devig_shin_into(odds, out_probs)?;
        }
    }
    Ok(())
}

/// Devigs a two-way (head-to-head / Over-Under / spread) market on the stack.
#[verified_engine::verified]
pub fn devig_two_way(
    home: Odds,
    away: Odds,
    method: DevigMethod,
) -> Result<[UnitInterval; 2], KellyError> {
    let odds = [home, away];
    let z = UnitInterval::new_clamped(0.5);
    let mut probs = [z, z];
    devig_into(&odds, method, &mut probs)?;
    Ok(probs)
}

/// Devigs a three-way (1X2) market on the stack.
#[verified_engine::verified]
pub fn devig_three_way(
    home: Odds,
    draw: Odds,
    away: Odds,
    method: DevigMethod,
) -> Result<[UnitInterval; 3], KellyError> {
    let odds = [home, draw, away];
    let z = UnitInterval::new_clamped(1.0 / 3.0);
    let mut probs = [z, z, z];
    devig_into(&odds, method, &mut probs)?;
    Ok(probs)
}

/// Converts a fair probability $p \in (0, 1)$ into zero-vigorish decimal [`Odds`] ($1 / p$).
#[verified_engine::verified]
pub fn fair_odds(prob: UnitInterval) -> Result<Odds, KellyError> {
    let p = prob.value();
    if p <= 0.0 || p >= 1.0 || !p.is_finite() {
        return Err(KellyError::InvalidProbability { value: p });
    }
    Odds::new(1.0 / p)
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
) -> Result<(f64, UnitInterval), KellyError> {
    let [fair_side, _] = devig_two_way(closing_side, closing_other, method)?;
    Ok((closing_line_value(bet_odds, fair_side), fair_side))
}

#[cfg(test)]
mod tests;
