//! Core types for Kelly Criterion.

use crate::error::KellyError;
pub use math_commons::primitives::UnitInterval;
use serde::{Deserialize, Deserializer, Serialize};

const MAX_CONTINUED_FRACTION_STEPS: usize = 96;

#[verified_engine::verified]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    let mut steps = 0usize;
    while b != 0 && steps < MAX_CONTINUED_FRACTION_STEPS {
        let r = a % b;
        a = b;
        b = r;
        steps += 1;
    }
    a.max(1)
}

#[verified_engine::verified]
fn select_semiconvergent(
    target: f64,
    bound: u128,
    p0: u128,
    q0: u128,
    p1: u128,
    q1: u128,
) -> (u128, u128) {
    let max_u64 = u128::from(u64::MAX);
    let k_q = (bound - q0).checked_div(q1).unwrap_or(0);
    let k_p = (max_u64 - p0).checked_div(p1).unwrap_or(k_q);
    let k = k_q.min(k_p);
    if k == 0 {
        return (p1, q1);
    }
    let ps = k * p1 + p0;
    let qs = k * q1 + q0;
    if p1 == 0
        || q1 == 0
        || (target - (ps as f64) / (qs as f64)).abs() < (target - (p1 as f64) / (q1 as f64)).abs()
    {
        (ps, qs)
    } else {
        (p1, q1)
    }
}

#[verified_engine::verified]
fn rationalize_positive(target: f64, bound: u64) -> (u64, u64) {
    if target >= u64::MAX as f64 {
        return (u64::MAX, 1);
    }
    if target <= 1.0 / (bound as f64) {
        return (1, bound);
    }

    let bound_u128 = u128::from(bound);
    let max_u64 = u128::from(u64::MAX);
    let tol = 1e-9 * target.max(1.0);
    let (mut p0, mut q0) = (0_u128, 1_u128);
    let (mut p1, mut q1) = (1_u128, 0_u128);
    let mut x = target;

    for _ in 0..MAX_CONTINUED_FRACTION_STEPS {
        let a_f64 = x.floor();
        if !a_f64.is_finite() || a_f64 < 0.0 || a_f64 > (u64::MAX as f64) {
            break;
        }
        let a = a_f64 as u128;
        let p2 = a.saturating_mul(p1).saturating_add(p0);
        let q2 = a.saturating_mul(q1).saturating_add(q0);
        if q2 > bound_u128 || p2 > max_u64 {
            (p1, q1) = select_semiconvergent(target, bound_u128, p0, q0, p1, q1);
            break;
        }
        (p0, q0) = (p1, q1);
        (p1, q1) = (p2, q2);
        if p1 > 0 && q1 > 0 && (target - (p1 as f64) / (q1 as f64)).abs() <= tol {
            break;
        }
        let rem = x - a_f64;
        if rem <= 1e-12 {
            break;
        }
        x = 1.0 / rem;
    }

    let num = (p1 as u64).max(1);
    let den = (q1 as u64).clamp(1, bound);
    let g = gcd_u64(num, den);
    (num / g, den / g)
}

/// Decimal odds `d` (must be finite and `> 1.0`).
///
/// Represents total return per unit staked (including stake):
/// - Odds of `2.0` means winning `$1` net profit on every `$1` bet (`b = 1.0`)
/// - Odds of `3.0` means winning `$2` net profit on every `$1` bet (`b = 2.0`, `2:1`)
///
/// Supports bidirectional conversions across decimal, net, American, fractional,
/// and implied-probability representations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Odds(f64);

impl Odds {
    /// Creates new decimal odds.
    ///
    /// # Arguments
    ///
    /// * `value` - The decimal odds (must be finite and `> 1.0`)
    ///
    /// # Returns
    ///
    /// * `Result<Odds, KellyError>` - The validated odds or an error
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let odds = Odds::new(2.5).unwrap();
    /// assert_eq!(odds.value(), 2.5);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, KellyError> {
        if value <= 1.0 || !value.is_finite() {
            return Err(KellyError::InvalidOdds { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw decimal odds value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }

    /// The net profit multiplier (`decimal - 1.0`).
    ///
    /// This is the amount won per unit bet, excluding the returned stake.
    /// For decimal odds of `2.5`, the net profit multiplier is `1.5`.
    #[verified_engine::verified]
    pub fn net_profit_multiplier(&self) -> f64 {
        self.0 - 1.0
    }

    /// Creates decimal odds from net profit multiplier `b` (`decimal = 1.0 + b`).
    ///
    /// Net odds represent profit won per unit staked (must be finite and `> 0.0`).
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let even_money = Odds::from_net_odds(1.0).unwrap();
    /// assert_eq!(even_money.value(), 2.0);
    /// assert_eq!(even_money.net_profit_multiplier(), 1.0);
    /// ```
    #[verified_engine::verified]
    pub fn from_net_odds(b: f64) -> Result<Self, KellyError> {
        let decimal = 1.0 + b;
        if !b.is_finite() || b <= 0.0 || !decimal.is_finite() || decimal <= 1.0 {
            return Err(KellyError::InvalidNetOdds { value: b });
        }
        Ok(Self(decimal))
    }

    /// Creates odds from American moneyline format (`|american| >= 100.0`).
    ///
    /// American odds use positive numbers for underdogs/even money and negative for favorites:
    /// - Positive (e.g., `+200`): win `$200` on `$100` bet -> `decimal = (200 / 100) + 1 = 3.0`
    /// - Even money (`+100` or `-100`): win `$100` on `$100` bet -> `decimal = 2.0`
    /// - Negative (e.g., `-150`): bet `$150` to win `$100` -> `decimal = (100 / 150) + 1 ≈ 1.667`
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// assert_eq!(Odds::from_american(-100.0).unwrap().value(), 2.0);
    /// assert_eq!(Odds::from_american(100.0).unwrap().value(), 2.0);
    /// ```
    #[verified_engine::verified]
    pub fn from_american(american: f64) -> Result<Self, KellyError> {
        if !american.is_finite() || american.abs() < 100.0 {
            return Err(KellyError::InvalidAmericanOdds { value: american });
        }
        let decimal = if american > 0.0 {
            (american / 100.0) + 1.0
        } else {
            (100.0 / american.abs()) + 1.0
        };
        if !decimal.is_finite() || decimal <= 1.0 {
            return Err(KellyError::InvalidAmericanOdds { value: american });
        }
        Ok(Self(decimal))
    }

    /// Converts decimal odds to American moneyline format (`>= +100.0` or `< -100.0`).
    ///
    /// Even-money decimal odds `2.0` map canonically to `+100.0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// assert!((Odds::new(3.0).unwrap().to_american() - 200.0).abs() < 1e-12);
    /// assert!((Odds::new(1.5).unwrap().to_american() - (-200.0)).abs() < 1e-12);
    /// ```
    #[verified_engine::verified]
    pub fn to_american(&self) -> f64 {
        if self.0 >= 2.0 {
            (self.0 - 1.0) * 100.0
        } else {
            -100.0 / (self.0 - 1.0)
        }
    }

    /// Creates odds from fractional format (`numerator / denominator`, both finite and `> 0.0`).
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let odds = Odds::from_fractional(5.0, 2.0).unwrap();
    /// assert!((odds.value() - 3.5).abs() < 1e-12);
    /// ```
    #[verified_engine::verified]
    pub fn from_fractional(numerator: f64, denominator: f64) -> Result<Self, KellyError> {
        if !numerator.is_finite()
            || !denominator.is_finite()
            || numerator <= 0.0
            || denominator <= 0.0
        {
            return Err(KellyError::InvalidFractionalOdds {
                numerator,
                denominator,
            });
        }
        let decimal = (numerator / denominator) + 1.0;
        if !decimal.is_finite() || decimal <= 1.0 {
            return Err(KellyError::InvalidFractionalOdds {
                numerator,
                denominator,
            });
        }
        Ok(Self(decimal))
    }

    /// Converts net profit multiplier `b = decimal - 1.0` to a coprime `(numerator, denominator)`
    /// fractional pair using bounded continued-fraction reduction with `1 <= denominator <= max_denominator.max(1)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let odds = Odds::new(3.5).unwrap();
    /// assert_eq!(odds.to_fractional(1000), (5, 2));
    /// ```
    #[verified_engine::verified]
    pub fn to_fractional(&self, max_denominator: u64) -> (u64, u64) {
        let bound = max_denominator.max(1);
        rationalize_positive(self.net_profit_multiplier(), bound)
    }

    /// Creates decimal odds from bookmaker implied probability `p` (`0.0 < p < 1.0`), where `decimal = 1.0 / p`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let odds = Odds::from_implied_probability(0.25).unwrap();
    /// assert!((odds.value() - 4.0).abs() < 1e-12);
    /// ```
    #[verified_engine::verified]
    pub fn from_implied_probability(probability: f64) -> Result<Self, KellyError> {
        if !probability.is_finite() || probability <= 0.0 || probability >= 1.0 {
            return Err(KellyError::InvalidImpliedProbability { value: probability });
        }
        let decimal = 1.0 / probability;
        if !decimal.is_finite() || decimal <= 1.0 {
            return Err(KellyError::InvalidImpliedProbability { value: probability });
        }
        Ok(Self(decimal))
    }

    /// Converts to implied probability (`1.0 / decimal_odds`).
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let odds = Odds::new(2.0).unwrap();
    /// assert!((odds.implied_probability() - 0.5).abs() < 1e-12);
    /// ```
    #[verified_engine::verified]
    pub fn implied_probability(&self) -> f64 {
        1.0 / self.0
    }
}

impl<'de> Deserialize<'de> for Odds {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Bankroll fraction to bet (must be finite and in `[0.0, 1.0]`).
///
/// Represents what fraction of the total bankroll to wager.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(transparent)]
pub struct BankrollFraction(f64);

impl BankrollFraction {
    /// Creates a new bankroll fraction.
    ///
    /// # Arguments
    ///
    /// * `value` - The fraction value (must be finite and in `[0.0, 1.0]`)
    ///
    /// # Returns
    ///
    /// * `Result<BankrollFraction, KellyError>` - The validated fraction or an error
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::BankrollFraction;
    ///
    /// let fraction = BankrollFraction::new(0.1).unwrap();
    /// assert_eq!(fraction.value(), 0.1);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, KellyError> {
        if !(0.0..=1.0).contains(&value) || !value.is_finite() {
            return Err(KellyError::InvalidFraction { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw fraction value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Computes the bet amount given a total bankroll.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::BankrollFraction;
    ///
    /// let fraction = BankrollFraction::new(0.1).unwrap();
    /// let bet = fraction.bet_amount(1000.0).unwrap();
    /// assert_eq!(bet, 100.0);
    /// ```
    #[verified_engine::verified]
    pub fn bet_amount(&self, bankroll: f64) -> Result<f64, KellyError> {
        if bankroll <= 0.0 || !bankroll.is_finite() {
            return Err(KellyError::InvalidBankroll { value: bankroll });
        }
        Ok(self.0 * bankroll)
    }
}

impl<'de> Deserialize<'de> for BankrollFraction {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
#[path = "core_tests.rs"]
mod tests;
