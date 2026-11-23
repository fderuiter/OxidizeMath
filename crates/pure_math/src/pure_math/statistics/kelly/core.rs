//! Core types for Kelly Criterion.

use crate::error::KellyError;
pub use math_commons::primitives::UnitInterval;

/// Decimal odds b (must be > 1.0).
///
/// Represents the odds offered. For example:
/// - Odds of 2.0 means you win $1 for every $1 bet (even money)
/// - Odds of 3.0 means you win $2 for every $1 bet (2:1)
///
/// Note: This uses decimal odds format. To convert from:
/// - American odds: Use `Odds::from_american()`
/// - Fractional odds: Use `Odds::from_fractional()`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Odds(f64);

impl Odds {
    /// Creates new decimal odds.
    ///
    /// # Arguments
    ///
    /// * `value` - The decimal odds (must be > 1.0)
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

    /// Returns the raw odds value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }

    /// The net profit multiplier (b - 1).
    ///
    /// This is the amount won per unit bet, excluding the returned stake.
    /// For decimal odds of 2.5, the net profit multiplier is 1.5.
    #[verified_engine::verified]
    pub fn net_profit_multiplier(&self) -> f64 {
        self.0 - 1.0
    }

    /// Creates decimal odds from net profit multiplier `b` (`decimal = 1.0 + b`).
    ///
    /// Net odds represent profit won per unit staked (must be finite and `> 0.0`):
    /// - Even money (`+100`): `b = 1.0` -> decimal odds `2.0`
    /// - `5:2` payoff: `b = 2.5` -> decimal odds `3.5`
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
        if !b.is_finite() || b <= 0.0 {
            return Err(KellyError::InvalidOdds { value: b });
        }
        Self::new(1.0 + b)
    }

    /// Creates odds from American format.
    ///
    /// American odds use positive numbers for underdogs and negative for favorites:
    /// - Positive (e.g., +200): win $200 on $100 bet → decimal = (200/100) + 1 = 3.0
    /// - Negative (e.g., -150): bet $150 to win $100 → decimal = (100/150) + 1 ≈ 1.67
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let underdog = Odds::from_american(200.0).unwrap(); // +200
    /// assert!((underdog.value() - 3.0).abs() < 0.01);
    ///
    /// let favorite = Odds::from_american(-150.0).unwrap(); // -150
    /// assert!((favorite.value() - 1.667).abs() < 0.01);
    /// ```
    #[verified_engine::verified]
    pub fn from_american(american: f64) -> Result<Self, KellyError> {
        if !american.is_finite() || american == 0.0 || (-100.0..100.0).contains(&american) {
            return Err(KellyError::InvalidOdds { value: american });
        }

        let decimal = if american > 0.0 {
            (american / 100.0) + 1.0
        } else {
            (100.0 / american.abs()) + 1.0
        };

        Self::new(decimal)
    }

    /// Creates odds from fractional format.
    ///
    /// Fractional odds represent the profit relative to stake (e.g., 5/2 or "5 to 2").
    /// To convert to decimal: (numerator / denominator) + 1
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let odds = Odds::from_fractional(5.0, 2.0).unwrap(); // 5/2
    /// assert!((odds.value() - 3.5).abs() < 0.01);
    /// ```
    #[verified_engine::verified]
    pub fn from_fractional(numerator: f64, denominator: f64) -> Result<Self, KellyError> {
        if denominator <= 0.0
            || numerator < 0.0
            || !numerator.is_finite()
            || !denominator.is_finite()
        {
            return Err(KellyError::InvalidOdds {
                value: numerator / denominator,
            });
        }

        Self::new((numerator / denominator) + 1.0)
    }

    /// Converts to implied probability (the bookmaker's edge).
    ///
    /// Implied probability = 1 / decimal_odds
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::Odds;
    ///
    /// let odds = Odds::new(2.0).unwrap();
    /// assert!((odds.implied_probability() - 0.5).abs() < 0.01);
    /// ```
    #[verified_engine::verified]
    pub fn implied_probability(&self) -> f64 {
        1.0 / self.0
    }
}

/// Bankroll fraction to bet (must be between 0 and 1).
///
/// Represents what fraction of the total bankroll to wager.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BankrollFraction(f64);

impl BankrollFraction {
    /// Creates a new bankroll fraction.
    ///
    /// # Arguments
    ///
    /// * `value` - The fraction value (must be between 0 and 1)
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

/// Maximum number of outcomes supported in a zero-allocation mutually exclusive market.
pub const MAX_MARKET_OUTCOMES: usize = 32;

/// Maximum number of concurrent independent wagers supported in a zero-allocation slate.
pub const MAX_SLATE_WAGERS: usize = 64;

/// Maximum wager count evaluated via exact state convolution before switching to Laplace quadrature.
pub const EXACT_CONVOLUTION_MAX_WAGERS: usize = 10;

/// Number of log-spaced nodes used by 1D Laplace / Frullani quadrature for large slates.
pub const LAPLACE_QUADRATURE_NODES: usize = 128;

/// Tolerance used when validating complete discrete probability distributions.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

/// Busseti, Ryu, and Boyd (2016) martingale drawdown risk constraint.
///
/// Bounds the infinite-horizon probability that wealth falls below fraction `alpha`
/// of initial bankroll to at most `beta`:
/// `P(inf_{t >= 1} W_t / W_0 <= alpha) <= beta`
/// via the convex supermartingale inequality `E[R^(-lambda)] <= 1`, where
/// `lambda = ln(beta) / ln(alpha) > 0` (equivalently `E[R^(1 - gamma)] <= 1` with `gamma = 1 + lambda`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawdownConstraint {
    alpha: f64,
    beta: f64,
    lambda: f64,
}

impl DrawdownConstraint {
    /// Constructs a validated Busseti-Ryu-Boyd drawdown constraint for `alpha, beta in (0, 1)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::kelly::DrawdownConstraint;
    ///
    /// let dc = DrawdownConstraint::new(0.70, 0.10).unwrap();
    /// assert!((dc.lambda() - 6.455696).abs() < 1e-5);
    /// assert!((dc.gamma() - (1.0 + dc.lambda())).abs() < 1e-12);
    /// ```
    #[verified_engine::verified]
    pub fn new(alpha: f64, beta: f64) -> Result<Self, KellyError> {
        if !alpha.is_finite()
            || !beta.is_finite()
            || alpha <= 0.0
            || alpha >= 1.0
            || beta <= 0.0
            || beta >= 1.0
        {
            return Err(KellyError::InvalidDrawdownBounds { alpha, beta });
        }
        let lambda = beta.ln() / alpha.ln();
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(KellyError::InvalidDrawdownBounds { alpha, beta });
        }
        Ok(Self {
            alpha,
            beta,
            lambda,
        })
    }

    /// Returns the wealth floor fraction `alpha in (0, 1)`.
    #[verified_engine::verified]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the maximum allowable drawdown probability `beta in (0, 1)`.
    #[verified_engine::verified]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Returns the martingale exponent `lambda = ln(beta) / ln(alpha) > 0`.
    #[verified_engine::verified]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the equivalent CRRA relative risk-aversion coefficient `gamma = 1 + lambda`.
    #[verified_engine::verified]
    pub fn gamma(&self) -> f64 {
        1.0 + self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_odds_valid() {
        let odds = Odds::new(2.5).unwrap();
        assert_eq!(odds.value(), 2.5);
        assert_eq!(odds.net_profit_multiplier(), 1.5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_odds_invalid() {
        assert!(Odds::new(1.0).is_err()); // Must be > 1.0
        assert!(Odds::new(0.5).is_err());
        assert!(Odds::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_odds_from_american() {
        // Underdog (+200)
        let odds = Odds::from_american(200.0).unwrap();
        assert!((odds.value() - 3.0).abs() < 0.01);

        // Favorite (-150)
        let odds = Odds::from_american(-150.0).unwrap();
        assert!((odds.value() - 1.667).abs() < 0.01);
    }

    #[test]
    #[verified_engine::verified]
    fn test_odds_from_fractional() {
        let odds = Odds::from_fractional(5.0, 2.0).unwrap(); // 5/2
        assert!((odds.value() - 3.5).abs() < 0.01);

        let odds = Odds::from_fractional(1.0, 1.0).unwrap(); // Even money
        assert!((odds.value() - 2.0).abs() < 0.01);
    }

    #[test]
    #[verified_engine::verified]
    fn test_odds_implied_probability() {
        let odds = Odds::new(2.0).unwrap();
        assert!((odds.implied_probability() - 0.5).abs() < math_commons::registry::TOLERANCE_FAST);

        let odds = Odds::new(4.0).unwrap();
        assert!((odds.implied_probability() - 0.25).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_bankroll_fraction_valid() {
        let fraction = BankrollFraction::new(0.1).unwrap();
        assert_eq!(fraction.value(), 0.1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_bankroll_fraction_invalid() {
        assert!(BankrollFraction::new(-0.1).is_err());
        assert!(BankrollFraction::new(1.1).is_err());
        assert!(BankrollFraction::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_bankroll_fraction_bet_amount() {
        let fraction = BankrollFraction::new(0.1).unwrap();
        let bet = fraction.bet_amount(1000.0).unwrap();
        assert_eq!(bet, 100.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_bankroll_fraction_boundaries() {
        assert!(BankrollFraction::new(0.0).is_ok());
        assert!(BankrollFraction::new(1.0).is_ok());
    }

    #[test]
    #[verified_engine::verified]
    fn test_odds_from_net_odds() {
        let even = Odds::from_net_odds(1.0).unwrap();
        assert!((even.value() - 2.0).abs() < math_commons::registry::TOLERANCE_HIGH);
        assert!(
            (even.net_profit_multiplier() - 1.0).abs() < math_commons::registry::TOLERANCE_HIGH
        );

        let sub_even = Odds::from_net_odds(0.5).unwrap();
        assert!((sub_even.value() - 1.5).abs() < math_commons::registry::TOLERANCE_HIGH);
        assert!(
            (sub_even.net_profit_multiplier() - 0.5).abs() < math_commons::registry::TOLERANCE_HIGH
        );

        assert!(Odds::from_net_odds(0.0).is_err());
        assert!(Odds::from_net_odds(-0.5).is_err());
        assert!(Odds::from_net_odds(f64::NAN).is_err());
        assert!(Odds::from_net_odds(f64::INFINITY).is_err());
    }
}
