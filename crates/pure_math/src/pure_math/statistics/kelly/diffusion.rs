//! Martingale drawdown constraints and Brownian diffusion / fixed-unit risk-of-ruin bounds.

use super::core::{BankrollFraction, Odds, UnitInterval};
use crate::error::KellyError;

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

/// Computes the single-wager logarithmic drift $\mu$ and variance $\sigma^2$ for a proportional
/// wager $f \in [0, 1)$ at decimal `odds` with win probability `win_prob`.
#[verified_engine::verified]
fn log_wealth_moments(
    win_prob: &UnitInterval,
    odds: &Odds,
    fraction: &BankrollFraction,
) -> (f64, f64) {
    let p = win_prob.value();
    let q = 1.0 - p;
    let b = odds.net_profit_multiplier();
    let f = fraction.value();
    if f <= 0.0 {
        return (0.0, 0.0);
    }
    if f >= 1.0 {
        return if q > 0.0 {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            ((1.0 + b).ln(), 0.0)
        };
    }
    let up = (1.0 + b * f).ln();
    let down = (1.0 - f).ln();
    let mu = p * up + q * down;
    let var = p * q * (up - down) * (up - down);
    (mu, var)
}

/// Estimates the continuous-diffusion probability of bankroll drawdown to `drawdown_ratio`
/// ($0 < \alpha < 1$, e.g. $\alpha = 0.5$ for a $50\%$ drawdown) under repeated proportional
/// bets of `fraction` at `odds`:
///
/// $$P\left(\inf_{n \ge 0} \frac{W_n}{W_0} \le \alpha\right) \approx \alpha^{2 \mu / \sigma^2}$$
///
/// When $\mu \le 0$ (non-positive log-growth rate, e.g. over-betting past ~2x Kelly or negative EV)
/// and $\sigma^2 > 0$, eventual breach occurs almost surely (`1.0`).
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{
///     Odds, UnitInterval, kelly_drawdown_risk, kelly_fraction,
/// };
///
/// let p = UnitInterval::new(0.55).unwrap();
/// let odds = Odds::new(2.0).unwrap();
/// let f_star = kelly_fraction(&p, &odds).unwrap();
/// let half_drawdown = kelly_drawdown_risk(&p, &odds, &f_star, 0.5).unwrap();
/// assert!(half_drawdown.value() > 0.45 && half_drawdown.value() < 0.55);
/// ```
#[verified_engine::verified]
pub fn kelly_drawdown_risk(
    win_prob: &UnitInterval,
    odds: &Odds,
    fraction: &BankrollFraction,
    drawdown_ratio: f64,
) -> Result<UnitInterval, KellyError> {
    if drawdown_ratio <= 0.0 || drawdown_ratio >= 1.0 || !drawdown_ratio.is_finite() {
        return Err(KellyError::InvalidFraction {
            value: drawdown_ratio,
        });
    }
    if fraction.value() <= 0.0 {
        return Ok(UnitInterval::new_clamped(0.0));
    }
    if win_prob.value() >= 1.0 {
        return Ok(UnitInterval::new_clamped(0.0));
    }
    if fraction.value() >= 1.0 {
        return Ok(UnitInterval::new_clamped(1.0));
    }
    let (mu, var) = log_wealth_moments(win_prob, odds, fraction);
    if mu <= 0.0 {
        return Ok(UnitInterval::new_clamped(if var > 0.0 { 1.0 } else { 0.0 }));
    }
    if var <= 0.0 {
        return Ok(UnitInterval::new_clamped(0.0));
    }
    let exponent = (2.0 * mu / var) * drawdown_ratio.ln();
    Ok(UnitInterval::new_clamped(exponent.exp()))
}

/// Estimates fixed-unit gambler's Risk of Ruin for a bankroll of `bankroll_units > 0`
/// betting 1 unit per trial at decimal `odds` with win probability `win_prob`, using
/// Thorp's Brownian drift-variance approximation:
///
/// $$R(B) = \exp\!\left(-\frac{2 \mu_{\text{unit}} B}{\sigma_{\text{unit}}^2}\right)$$
///
/// where $\mu_{\text{unit}} = p b - q$ and $\sigma_{\text{unit}}^2 = p q (b + 1)^2$.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::kelly::{Odds, UnitInterval, unit_risk_of_ruin};
///
/// let p = UnitInterval::new(0.54).unwrap();
/// let odds = Odds::from_american(-110.0).unwrap();
/// let ror = unit_risk_of_ruin(&p, &odds, 50.0).unwrap();
/// assert!(ror.value() > 0.0 && ror.value() < 1.0);
/// ```
#[verified_engine::verified]
pub fn unit_risk_of_ruin(
    win_prob: &UnitInterval,
    odds: &Odds,
    bankroll_units: f64,
) -> Result<UnitInterval, KellyError> {
    if bankroll_units <= 0.0 || !bankroll_units.is_finite() {
        return Err(KellyError::InvalidBankroll {
            value: bankroll_units,
        });
    }
    let p = win_prob.value();
    let q = 1.0 - p;
    if p <= 0.0 {
        return Ok(UnitInterval::new_clamped(1.0));
    }
    if q <= 0.0 {
        return Ok(UnitInterval::new_clamped(0.0));
    }
    let b = odds.net_profit_multiplier();
    let drift = p * b - q;
    if drift <= 0.0 {
        return Ok(UnitInterval::new_clamped(1.0));
    }
    let variance = p * q * (b + 1.0) * (b + 1.0);
    if variance <= 0.0 {
        return Ok(UnitInterval::new_clamped(0.0));
    }
    let exponent = -2.0 * drift * bankroll_units / variance;
    Ok(UnitInterval::new_clamped(exponent.exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::statistics::kelly::{kelly_fraction, variants};

    #[test]
    #[verified_engine::verified]
    fn test_kelly_drawdown_risk_monotonic_in_fraction() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();
        let full = kelly_fraction(&p, &odds).unwrap();
        let half = variants::half_kelly(&p, &odds).unwrap();

        let risk_full = kelly_drawdown_risk(&p, &odds, &full, 0.5).unwrap();
        let risk_half = kelly_drawdown_risk(&p, &odds, &half, 0.5).unwrap();
        assert!((risk_full.value() - 0.5).abs() < 0.02);
        assert!(risk_half.value() < risk_full.value());
        assert!(kelly_drawdown_risk(&p, &odds, &full, 0.0).is_err());
        assert!(kelly_drawdown_risk(&p, &odds, &full, 1.0).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_unit_risk_of_ruin_decays_with_bankroll() {
        let p = UnitInterval::new(0.55).unwrap();
        let odds = Odds::new(2.0).unwrap();
        let r10 = unit_risk_of_ruin(&p, &odds, 10.0).unwrap();
        let r50 = unit_risk_of_ruin(&p, &odds, 50.0).unwrap();
        assert!(r50.value() < r10.value());

        let neg_ev = UnitInterval::new(0.45).unwrap();
        assert_eq!(
            unit_risk_of_ruin(&neg_ev, &odds, 50.0).unwrap().value(),
            1.0
        );
        assert!(unit_risk_of_ruin(&p, &odds, 0.0).is_err());
    }
}

// [cite:sports_quantitative_modeling]
