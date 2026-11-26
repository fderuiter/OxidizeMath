//! Closed-form Black-Scholes-Merton European option pricing and analytical Greeks.
//!
//! Computes exact European call and put option values alongside first- and second-order
//! sensitivities (`Delta`, `Gamma`, `Vega`, `Theta`, `Rho`) with continuous dividend yield $q$.

use std::f64::consts::{FRAC_1_SQRT_2, PI};
use thiserror::Error;

/// Errors returned when validating Black-Scholes-Merton option parameters.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum OptionPricingError {
    /// Underlying spot price must be strictly positive and finite.
    #[error("Invalid spot price: {value} (must be finite and > 0)")]
    InvalidSpot {
        /// Supplied spot price $S$.
        value: f64,
    },
    /// Option strike price must be strictly positive and finite.
    #[error("Invalid strike price: {value} (must be finite and > 0)")]
    InvalidStrike {
        /// Supplied strike price $K$.
        value: f64,
    },
    /// Annualized implied volatility must be strictly positive and finite.
    #[error("Invalid volatility: {value} (must be finite and > 0)")]
    InvalidVolatility {
        /// Supplied volatility $\sigma$.
        value: f64,
    },
    /// Time to expiration (in years) must be strictly positive and finite.
    #[error("Invalid time to expiry: {value} (must be finite and > 0)")]
    InvalidTimeToExpiry {
        /// Supplied time to expiration $T$.
        value: f64,
    },
    /// Continuously compounded risk-free rate must be finite.
    #[error("Invalid risk-free rate: {value} (must be finite)")]
    InvalidRate {
        /// Supplied risk-free interest rate $r$.
        value: f64,
    },
    /// Continuous dividend yield must be finite and non-negative.
    #[error("Invalid dividend yield: {value} (must be finite and >= 0)")]
    InvalidDividendYield {
        /// Supplied continuous dividend yield $q$.
        value: f64,
    },
}

/// Payoff direction for a European option contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OptionType {
    /// European Call option ($\max(S_T - K, 0)$).
    Call,
    /// European Put option ($\max(K - S_T, 0)$).
    Put,
}

/// Validated input parameters for the Black-Scholes-Merton European option model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlackScholesParams {
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
}

impl BlackScholesParams {
    /// Constructs validated Black-Scholes parameters with zero dividend yield ($q = 0$).
    ///
    /// # Example
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::black_scholes::BlackScholesParams;
    ///
    /// let params = BlackScholesParams::new(100.0, 100.0, 0.05, 0.20, 1.0).unwrap();
    /// assert_eq!(params.spot(), 100.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(
        spot: f64,
        strike: f64,
        rate: f64,
        volatility: f64,
        time_to_expiry: f64,
    ) -> Result<Self, OptionPricingError> {
        Self::with_dividend_yield(spot, strike, rate, 0.0, volatility, time_to_expiry)
    }

    /// Constructs validated Black-Scholes-Merton parameters with continuous dividend yield `q`.
    #[verified_engine::verified]
    pub fn with_dividend_yield(
        spot: f64,
        strike: f64,
        rate: f64,
        dividend_yield: f64,
        volatility: f64,
        time_to_expiry: f64,
    ) -> Result<Self, OptionPricingError> {
        if !spot.is_finite() || spot <= 0.0 {
            return Err(OptionPricingError::InvalidSpot { value: spot });
        }
        if !strike.is_finite() || strike <= 0.0 {
            return Err(OptionPricingError::InvalidStrike { value: strike });
        }
        if !rate.is_finite() {
            return Err(OptionPricingError::InvalidRate { value: rate });
        }
        if !dividend_yield.is_finite() || dividend_yield < 0.0 {
            return Err(OptionPricingError::InvalidDividendYield {
                value: dividend_yield,
            });
        }
        if !volatility.is_finite() || volatility <= 0.0 {
            return Err(OptionPricingError::InvalidVolatility { value: volatility });
        }
        if !time_to_expiry.is_finite() || time_to_expiry <= 0.0 {
            return Err(OptionPricingError::InvalidTimeToExpiry {
                value: time_to_expiry,
            });
        }
        Ok(Self {
            spot,
            strike,
            rate,
            dividend_yield,
            volatility,
            time_to_expiry,
        })
    }

    /// Underlying spot price $S$.
    #[verified_engine::verified]
    pub fn spot(&self) -> f64 {
        self.spot
    }

    /// Strike price $K$.
    #[verified_engine::verified]
    pub fn strike(&self) -> f64 {
        self.strike
    }

    /// Continuously compounded risk-free rate $r$.
    #[verified_engine::verified]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Continuous dividend yield $q$.
    #[verified_engine::verified]
    pub fn dividend_yield(&self) -> f64 {
        self.dividend_yield
    }

    /// Annualized volatility $\sigma$.
    #[verified_engine::verified]
    pub fn volatility(&self) -> f64 {
        self.volatility
    }

    /// Time to expiration $T$ in years.
    #[verified_engine::verified]
    pub fn time_to_expiry(&self) -> f64 {
        self.time_to_expiry
    }
}

/// Closed-form option value and analytical Black-Scholes-Merton Greeks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionGreeks {
    /// Fair theoretical option value.
    pub price: f64,
    /// $\Delta = \partial V / \partial S$.
    pub delta: f64,
    /// $\Gamma = \partial^2 V / \partial S^2$.
    pub gamma: f64,
    /// $\mathcal{V} = \partial V / \partial \sigma$ (per `1.0` unit of volatility).
    pub vega: f64,
    /// $\Theta = \partial V / \partial t = -\partial V / \partial T$ (annualized time decay).
    pub theta: f64,
    /// $\rho = \partial V / \partial r$ (per `1.0` unit of interest rate).
    pub rho: f64,
    /// Standardized moneyness score $d_1$.
    pub d1: f64,
    /// Risk-neutral exercise score $d_2 = d_1 - \sigma\sqrt{T}$.
    pub d2: f64,
}

/// Simultaneous Call and Put Black-Scholes-Merton valuation and Greeks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlackScholesPair {
    /// European Call valuation and Greeks.
    pub call: OptionGreeks,
    /// European Put valuation and Greeks.
    pub put: OptionGreeks,
}

#[inline]
fn std_norm_cdf(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x * FRAC_1_SQRT_2)
}

#[inline]
fn std_norm_pdf(x: f64) -> f64 {
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
    inv_sqrt_2pi * (-0.5 * x * x).exp()
}

/// Computes closed-form Black-Scholes-Merton European Call and Put prices and analytical Greeks
/// in a single zero-allocation pass:
/// $$C = S e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2), \quad P = K e^{-rT}\Phi(-d_2) - S e^{-qT}\Phi(-d_1), \quad d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}.$$
///
/// Reference: Black & Scholes (1973), *J. Polit. Econ.* 81(3), DOI: 10.1086/260062;
/// Merton (1973), *Bell J. Econ. Manag. Sci.* 4(1), DOI: 10.2307/3003143;
/// [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Evaluates $\Phi(\pm d_{1,2}) = \tfrac{1}{2}\operatorname{erfc}(\mp d_{1,2}/\sqrt{2})$ directly
/// via complementary error functions to avoid catastrophic $1 - \Phi(x)$ tail cancellation for
/// deep out-of-the-money contracts, preserving put-call parity $C - P = S e^{-qT} - K e^{-rT}$ to
/// machine precision.
///
/// # Domain of Validity
/// Validated via [`BlackScholesParams`]: requires strictly positive finite spot $S > 0$, strike
/// $K > 0$, volatility $\sigma > 0$, and time to expiry $T > 0$, finite risk-free rate $r$, and
/// finite non-negative continuous dividend yield $q \ge 0$ under constant-parameter geometric
/// Brownian motion without early exercise.
///
/// # Example
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::black_scholes::{
///     black_scholes_merton, BlackScholesParams,
/// };
///
/// let params = BlackScholesParams::new(100.0, 100.0, 0.05, 0.20, 1.0).unwrap();
/// let pair = black_scholes_merton(&params);
/// assert!((pair.call.price - 10.45058).abs() < 1e-4);
/// assert!((pair.put.price - 5.57353).abs() < 1e-4);
/// ```
#[verified_engine::verified]
pub fn black_scholes_merton(params: &BlackScholesParams) -> BlackScholesPair {
    let s = params.spot;
    let k = params.strike;
    let r = params.rate;
    let q = params.dividend_yield;
    let sigma = params.volatility;
    let t = params.time_to_expiry;

    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;

    let disc_r = (-r * t).exp();
    let disc_q = (-q * t).exp();
    let phi_d1 = std_norm_pdf(d1);
    let n_d1 = std_norm_cdf(d1);
    let n_d2 = std_norm_cdf(d2);
    let n_neg_d1 = std_norm_cdf(-d1);
    let n_neg_d2 = std_norm_cdf(-d2);

    let gamma = (disc_q * phi_d1) / (s * vol_sqrt_t);
    let vega = s * disc_q * phi_d1 * sqrt_t;
    let common_theta = -(s * disc_q * phi_d1 * sigma) / (2.0 * sqrt_t);

    let call = OptionGreeks {
        price: s * disc_q * n_d1 - k * disc_r * n_d2,
        delta: disc_q * n_d1,
        gamma,
        vega,
        theta: common_theta - r * k * disc_r * n_d2 + q * s * disc_q * n_d1,
        rho: k * t * disc_r * n_d2,
        d1,
        d2,
    };

    let put = OptionGreeks {
        price: k * disc_r * n_neg_d2 - s * disc_q * n_neg_d1,
        delta: -disc_q * n_neg_d1,
        gamma,
        vega,
        theta: common_theta + r * k * disc_r * n_neg_d2 - q * s * disc_q * n_neg_d1,
        rho: -k * t * disc_r * n_neg_d2,
        d1,
        d2,
    };

    BlackScholesPair { call, put }
}

/// Computes the analytical price and Greeks for a single [`OptionType`].
#[verified_engine::verified]
pub fn price_and_greeks(params: &BlackScholesParams, option_type: OptionType) -> OptionGreeks {
    let pair = black_scholes_merton(params);
    match option_type {
        OptionType::Call => pair.call,
        OptionType::Put => pair.put,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_hull_european_call_put_and_parity() {
        let params = BlackScholesParams::new(100.0, 100.0, 0.05, 0.20, 1.0).unwrap();
        let (pair, _) = VerifiedEngine::run_verified(|| black_scholes_merton(&params));

        assert!((pair.call.price - 10.450_583_57).abs() < 1e-6);
        assert!((pair.put.price - 5.573_526_02).abs() < 1e-6);
        assert!((pair.call.delta - 0.636_830_65).abs() < 1e-6);
        assert!((pair.put.delta - (-0.363_169_35)).abs() < 1e-6);
        assert!((pair.call.gamma - 0.018_762_02).abs() < 1e-6);
        assert!((pair.call.vega - 37.524_034_69).abs() < 1e-5);
        assert!((pair.call.theta - (-6.414_027_55)).abs() < 1e-5);
        assert!((pair.call.rho - 53.232_481_55).abs() < 1e-5);

        let parity_lhs = pair.call.price - pair.put.price;
        let parity_rhs = 100.0 - 100.0 * (-0.05_f64).exp();
        assert!((parity_lhs - parity_rhs).abs() < 1e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_greeks_match_central_finite_differences() {
        let base =
            BlackScholesParams::with_dividend_yield(105.0, 100.0, 0.04, 0.015, 0.25, 0.75).unwrap();
        let g = price_and_greeks(&base, OptionType::Put);
        let eps = 1e-4;

        let s_up =
            BlackScholesParams::with_dividend_yield(105.0 + eps, 100.0, 0.04, 0.015, 0.25, 0.75)
                .unwrap();
        let s_dn =
            BlackScholesParams::with_dividend_yield(105.0 - eps, 100.0, 0.04, 0.015, 0.25, 0.75)
                .unwrap();
        let p_up = price_and_greeks(&s_up, OptionType::Put).price;
        let p_dn = price_and_greeks(&s_dn, OptionType::Put).price;
        assert!(((p_up - p_dn) / (2.0 * eps) - g.delta).abs() < 1e-6);
        assert!(((p_up - 2.0 * g.price + p_dn) / (eps * eps) - g.gamma).abs() < 1e-4);

        let v_up =
            BlackScholesParams::with_dividend_yield(105.0, 100.0, 0.04, 0.015, 0.25 + eps, 0.75)
                .unwrap();
        let v_dn =
            BlackScholesParams::with_dividend_yield(105.0, 100.0, 0.04, 0.015, 0.25 - eps, 0.75)
                .unwrap();
        let vega_fd = (price_and_greeks(&v_up, OptionType::Put).price
            - price_and_greeks(&v_dn, OptionType::Put).price)
            / (2.0 * eps);
        assert!((vega_fd - g.vega).abs() < 1e-4);

        let r_up =
            BlackScholesParams::with_dividend_yield(105.0, 100.0, 0.04 + eps, 0.015, 0.25, 0.75)
                .unwrap();
        let r_dn =
            BlackScholesParams::with_dividend_yield(105.0, 100.0, 0.04 - eps, 0.015, 0.25, 0.75)
                .unwrap();
        let rho_fd = (price_and_greeks(&r_up, OptionType::Put).price
            - price_and_greeks(&r_dn, OptionType::Put).price)
            / (2.0 * eps);
        assert!((rho_fd - g.rho).abs() < 1e-4);

        let t_up =
            BlackScholesParams::with_dividend_yield(105.0, 100.0, 0.04, 0.015, 0.25, 0.75 + eps)
                .unwrap();
        let t_dn =
            BlackScholesParams::with_dividend_yield(105.0, 100.0, 0.04, 0.015, 0.25, 0.75 - eps)
                .unwrap();
        let theta_fd = -(price_and_greeks(&t_up, OptionType::Put).price
            - price_and_greeks(&t_dn, OptionType::Put).price)
            / (2.0 * eps);
        assert!((theta_fd - g.theta).abs() < 1e-4);
    }

    #[test]
    #[verified_engine::verified]
    fn test_invalid_bsm_parameters_rejected() {
        assert!(BlackScholesParams::new(0.0, 100.0, 0.05, 0.2, 1.0).is_err());
        assert!(BlackScholesParams::new(100.0, -1.0, 0.05, 0.2, 1.0).is_err());
        assert!(BlackScholesParams::new(100.0, 100.0, f64::NAN, 0.2, 1.0).is_err());
        assert!(BlackScholesParams::new(100.0, 100.0, 0.05, 0.0, 1.0).is_err());
        assert!(BlackScholesParams::new(100.0, 100.0, 0.05, 0.2, 0.0).is_err());
        assert!(
            BlackScholesParams::with_dividend_yield(100.0, 100.0, 0.05, -0.01, 0.2, 1.0).is_err()
        );
    }
}

// [cite:clinical_trials_statistics]
