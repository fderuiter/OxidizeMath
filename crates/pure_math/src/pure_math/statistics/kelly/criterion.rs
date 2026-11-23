//! Kelly Criterion calculations for optimal bet sizing.

use super::core::{BankrollFraction, Odds, UnitInterval};
use crate::error::KellyError;

/// Computes the optimal Kelly fraction for a bet.
///
/// The Kelly criterion maximizes the expected logarithmic growth rate of wealth.
///
/// Formula: f* = (bp - q) / b
///
/// where:
/// - b = net profit multiplier (odds - 1)
/// - p = win probability
/// - q = loss probability (1 - p)
///
/// # Arguments
///
/// * `probability` - Win probability
/// * `odds` - Decimal odds
///
/// # Returns
///
/// * `Result<BankrollFraction, KellyError>` - The optimal fraction or an error
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{
///     kelly_fraction, UnitInterval, Odds
/// };
///
/// let p = UnitInterval::new(0.55).unwrap();
/// let odds = Odds::new(2.0).unwrap();
/// let kelly = kelly_fraction(&p, &odds).unwrap();
///
/// println!("Optimal Kelly fraction: {}", kelly.value());
/// ```
///
/// # References
///
/// Kelly, J. L. (1956). "A New Interpretation of Information Rate."
/// Bell System Technical Journal, 35(4), 917–926.
#[verified_engine::verified]
pub fn kelly_fraction(
    probability: &UnitInterval,
    odds: &Odds,
) -> Result<BankrollFraction, KellyError> {
    let p = probability.value();
    let q = probability.complement();
    let b = odds.net_profit_multiplier();

    // Kelly formula: f* = (bp - q) / b
    let f = (b * p - q) / b;

    // Check for no edge (negative expectation)
    if f <= 0.0 {
        return Err(KellyError::NoEdge {
            probability: p,
            odds: odds.value(),
        });
    }

    // Cap at 100% (though this rarely happens with valid odds)
    let f_capped = f.min(1.0);

    BankrollFraction::new(f_capped)
}

/// Computes a fractional Kelly bet (reduced Kelly for risk management).
///
/// Many practitioners use fractional Kelly (e.g., half-Kelly or quarter-Kelly)
/// to reduce volatility while still capturing most of the growth.
///
/// Formula: f*_fractional = fraction × f*
///
/// Common fractions:
/// - Quarter-Kelly: 0.25 (very conservative)
/// - Half-Kelly: 0.50 (balanced risk/reward)
/// - Full Kelly: 1.00 (maximum growth, high volatility)
///
/// # Arguments
///
/// * `probability` - Win probability
/// * `odds` - Decimal odds
/// * `fraction` - Fraction of full Kelly to use (typically 0.25 or 0.5)
///
/// # Returns
///
/// * `Result<BankrollFraction, KellyError>` - The fractional Kelly or an error
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{
///     fractional_kelly, UnitInterval, Odds
/// };
///
/// let p = UnitInterval::new(0.55).unwrap();
/// let odds = Odds::new(2.0).unwrap();
///
/// let half_kelly = fractional_kelly(&p, &odds, 0.5).unwrap();
/// println!("Half-Kelly fraction: {}", half_kelly.value());
/// ```
#[verified_engine::verified]
pub fn fractional_kelly(
    probability: &UnitInterval,
    odds: &Odds,
    fraction: f64,
) -> Result<BankrollFraction, KellyError> {
    if !(0.0..=1.0).contains(&fraction) || !fraction.is_finite() {
        return Err(KellyError::InvalidFraction { value: fraction });
    }

    let full_kelly = kelly_fraction(probability, odds)?;
    BankrollFraction::new(full_kelly.value() * fraction)
}

/// Computes the expected growth rate for a given bet fraction.
///
/// The growth rate g is the expected logarithmic return per bet:
///
/// Formula: g = p ln(1 + bf) + q ln(1 - f)
///
/// where:
/// - p = win probability
/// - q = 1 - p
/// - b = net profit multiplier
/// - f = bet fraction
///
/// A positive growth rate indicates profitable betting over time.
///
/// # Arguments
///
/// * `probability` - Win probability
/// * `odds` - Decimal odds
/// * `fraction` - Bet fraction
///
/// # Returns
///
/// * `f64` - The expected growth rate
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{
///     expected_growth_rate, kelly_fraction, UnitInterval, Odds
/// };
///
/// let p = UnitInterval::new(0.55).unwrap();
/// let odds = Odds::new(2.0).unwrap();
/// let kelly = kelly_fraction(&p, &odds).unwrap();
///
/// let growth = expected_growth_rate(&p, &odds, &kelly);
/// println!("Expected growth rate: {:.4}", growth);
/// ```
#[verified_engine::verified]
pub fn expected_growth_rate(
    probability: &UnitInterval,
    odds: &Odds,
    fraction: &BankrollFraction,
) -> f64 {
    let p = probability.value();
    let q = probability.complement();
    let b = odds.net_profit_multiplier();
    let f = fraction.value();

    // Avoid log of non-positive numbers
    if f >= 1.0 || f < 0.0 || 1.0 + b * f <= 0.0 {
        return f64::NEG_INFINITY;
    }

    p * (1.0 + b * f).ln() + q * (1.0 - f).ln()
}

/// Computes the expected value (EV) of a bet.
///
/// Formula: EV = p × (win_amount) - q × (loss_amount)
///         = p × (b × stake) - q × stake
///         = stake × (bp - q)
///
/// For a unit stake:
/// EV = bp - q
///
/// # Arguments
///
/// * `probability` - Win probability
/// * `odds` - Decimal odds
///
/// # Returns
///
/// * `f64` - The expected value per unit staked
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{
///     expected_value, UnitInterval, Odds
/// };
///
/// let p = UnitInterval::new(0.55).unwrap();
/// let odds = Odds::new(2.0).unwrap();
///
/// let ev = expected_value(&p, &odds);
/// if ev > 0.0 {
///     println!("Positive EV: {:.4} per unit staked", ev);
/// }
/// ```
#[verified_engine::verified]
pub fn expected_value(probability: &UnitInterval, odds: &Odds) -> f64 {
    let p = probability.value();
    let q = probability.complement();
    let b = odds.net_profit_multiplier();

    b * p - q
}

/// Common Kelly variants for convenience.
pub mod variants {
    use super::*;

    /// Computes quarter-Kelly (25% of full Kelly).
    ///
    /// Very conservative approach with minimal drawdowns.
    #[verified_engine::verified]
    pub fn quarter_kelly(
        probability: &UnitInterval,
        odds: &Odds,
    ) -> Result<BankrollFraction, KellyError> {
        fractional_kelly(probability, odds, 0.25)
    }

    /// Computes half-Kelly (50% of full Kelly).
    ///
    /// Balanced approach: 75% of full Kelly growth with 50% of variance.
    #[verified_engine::verified]
    pub fn half_kelly(
        probability: &UnitInterval,
        odds: &Odds,
    ) -> Result<BankrollFraction, KellyError> {
        fractional_kelly(probability, odds, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_kelly_fraction_positive_edge() {
        // 55% win probability, 2:1 odds (even money)
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();
        let kelly = kelly_fraction(&p, &odds).unwrap();

        // f* = (1.0 * 0.55 - 0.45) / 1.0 = 0.10
        assert!((kelly.value() - 0.10).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_kelly_fraction_no_edge() {
        // Fair coin, even money odds (no edge)
        let p = UnitInterval::new(0.5).unwrap();
        let odds = Odds::new(2.0).unwrap();

        // Should return error (no positive edge)
        assert!(kelly_fraction(&p, &odds).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_kelly_fraction_large_edge() {
        // 70% win probability, 2:1 odds
        let p = UnitInterval::new(0.70).unwrap();
        let odds = Odds::new(2.0).unwrap();
        let kelly = kelly_fraction(&p, &odds).unwrap();

        // f* = (1.0 * 0.70 - 0.30) / 1.0 = 0.40
        assert!((kelly.value() - 0.40).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_fractional_kelly_half() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();
        let half = fractional_kelly(&p, &odds, 0.5).unwrap();

        let full = kelly_fraction(&p, &odds).unwrap();
        assert!((half.value() - full.value() * 0.5).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_fractional_kelly_quarter() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();
        let quarter = fractional_kelly(&p, &odds, 0.25).unwrap();

        let full = kelly_fraction(&p, &odds).unwrap();
        assert!(
            (quarter.value() - full.value() * 0.25).abs() < math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_expected_growth_rate_full_kelly() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();
        let kelly = kelly_fraction(&p, &odds).unwrap();

        let growth = expected_growth_rate(&p, &odds, &kelly);

        // Should be positive for positive edge
        assert!(growth > 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_expected_growth_rate_zero_bet() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();
        let zero = BankrollFraction::new(0.0).unwrap();

        let growth = expected_growth_rate(&p, &odds, &zero);

        // Zero bet means zero growth
        assert!((growth - 0.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    }

    #[test]
    #[verified_engine::verified]
    fn test_expected_growth_rate_overbetting() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();

        // Bet 100% of bankroll (too much)
        let full = BankrollFraction::new(1.0).unwrap();
        let growth = expected_growth_rate(&p, &odds, &full);

        // Should be negative (risk of ruin)
        assert!(growth < 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_expected_value_positive_edge() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();

        let ev = expected_value(&p, &odds);

        // EV = 1.0 * 0.55 - 0.45 = 0.10
        assert!((ev - 0.10).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_expected_value_no_edge() {
        let p = UnitInterval::new(0.5).unwrap();
        let odds = Odds::new(2.0).unwrap();

        let ev = expected_value(&p, &odds);

        // Fair bet: EV = 0
        assert!(ev.abs() < math_commons::registry::TOLERANCE_HIGH);
    }

    #[test]
    #[verified_engine::verified]
    fn test_expected_value_negative_edge() {
        let p = UnitInterval::new(0.45).unwrap();
        let odds = Odds::new(2.0).unwrap();

        let ev = expected_value(&p, &odds);

        // Negative edge
        assert!(ev < 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_variants_quarter_kelly() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();

        let quarter = variants::quarter_kelly(&p, &odds).unwrap();
        let full = kelly_fraction(&p, &odds).unwrap();

        assert!(
            (quarter.value() - full.value() * 0.25).abs() < math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_variants_half_kelly() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();

        let half = variants::half_kelly(&p, &odds).unwrap();
        let full = kelly_fraction(&p, &odds).unwrap();

        assert!((half.value() - full.value() * 0.5).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_realistic_sports_betting() {
        // Realistic scenario: 53% win rate with -110 odds (American)
        // -110 converts to approximately 1.909 decimal odds
        let p = UnitInterval::new(0.53).unwrap();
        let odds = Odds::from_american(-110.0).unwrap();

        let ev = expected_value(&p, &odds);

        // Check if there's an edge
        if ev > 0.0 {
            let kelly = kelly_fraction(&p, &odds).unwrap();
            assert!(kelly.value() > 0.0);
            assert!(kelly.value() < 0.1); // Typically small for sports betting
        } else {
            // If no edge, kelly_fraction should return error
            assert!(kelly_fraction(&p, &odds).is_err());
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_high_odds_scenario() {
        // Long shot: 20% win probability, 6:1 odds
        let p = UnitInterval::new(0.20).unwrap();
        let odds = Odds::new(6.0).unwrap();

        let kelly = kelly_fraction(&p, &odds).unwrap();

        // f* = (5.0 * 0.20 - 0.80) / 5.0 = 0.04
        assert!((kelly.value() - 0.04).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_expected_growth_rate_above_reciprocal_net_odds_is_finite() {
        // With net odds b = 2.0 (decimal 3.0), 1 / b = 0.5, yet f = 0.60 is strictly < 1.0.
        let p = UnitInterval::new(0.60).unwrap();
        let odds = Odds::from_net_odds(2.0).unwrap();
        let frac = BankrollFraction::new(0.60).unwrap();
        let g = expected_growth_rate(&p, &odds, &frac);
        let expected = 0.60 * (1.0_f64 + 1.20).ln() + 0.40 * (0.40_f64).ln();
        assert!(g.is_finite());
        assert!((g - expected).abs() < math_commons::registry::TOLERANCE_HIGH);
    }

    #[test]
    #[verified_engine::verified]
    fn test_expected_growth_rate_volatility_drag_above_double_kelly() {
        // p = 0.55, b = 1.0 => f* = 0.10, 2*f* = 0.20; at f = 0.30, g(0.30) < 0 while EV > 0.
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::from_net_odds(1.0).unwrap();
        let overbet = BankrollFraction::new(0.30).unwrap();
        let g = expected_growth_rate(&p, &odds, &overbet);
        let w_median_200 = 100.0 * (200.0 * g).exp();
        assert!(g < 0.0);
        assert!((w_median_200 - 3.913748).abs() < 1e-4);
    }
}
