//! Link functions for ZIP regression.
//!
//! This module implements the link functions used in ZIP regression to
//! transform the linear predictors into parameter values.

use nalgebra::DVector;

/// Log link function for modeling the Poisson rate λ.
///
/// Maps the linear predictor η to the rate parameter:
/// λ = exp(η) = exp(β₀ + β₁x₁ + ... + βₚxₚ)
///
/// This ensures λ > 0, which is required for the Poisson distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LogLink;

impl LogLink {
    /// Applies the log link function: λ = exp(η).
    ///
    /// # Arguments
    ///
    /// * `linear_predictor` - The linear predictor η = X'β
    ///
    /// # Returns
    ///
    /// The rate parameter λ = exp(η)
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::LogLink;
    ///
    /// let eta = 1.0;
    /// let lambda = LogLink::link(eta);
    /// assert!((lambda - 2.718281828).abs() < math_explorer::registry::TOLERANCE_FAST);  // e^1
    /// ```
    #[verified_engine::verified]
    pub fn link(linear_predictor: f64) -> f64 {
        linear_predictor.exp()
    }

    /// Applies the inverse log link function: η = log(λ).
    ///
    /// # Arguments
    ///
    /// * `rate` - The rate parameter λ (must be positive)
    ///
    /// # Returns
    ///
    /// The linear predictor η = log(λ)
    #[verified_engine::verified]
    pub fn inverse_link(rate: f64) -> f64 {
        rate.ln()
    }

    /// Computes the derivative of the link function.
    ///
    /// d(λ)/d(η) = exp(η) = λ
    #[verified_engine::verified]
    pub fn derivative(linear_predictor: f64) -> f64 {
        linear_predictor.exp()
    }

    /// Applies the log link to a vector of linear predictors.
    #[verified_engine::verified]
    pub fn link_vector(linear_predictors: &DVector<f64>) -> DVector<f64> {
        linear_predictors.map(Self::link)
    }
}

/// Logit link function for modeling the zero-inflation probability ρ.
///
/// Maps the linear predictor γ to the zero-inflation probability:
/// ρ = 1/(1 + exp(-γ)) = expit(γ)
///
/// This ensures ρ ∈ [0, 1], which is required for a probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LogitLink;

impl LogitLink {
    /// Applies the logit link function: ρ = 1/(1 + exp(-γ)).
    ///
    /// # Arguments
    ///
    /// * `linear_predictor` - The linear predictor γ = Z'α
    ///
    /// # Returns
    ///
    /// The zero-inflation probability ρ = expit(γ)
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::LogitLink;
    ///
    /// let gamma = 0.0;
    /// let rho = LogitLink::link(gamma);
    /// assert!((rho - 0.5).abs() < math_explorer::registry::TOLERANCE_STANDARD);  // expit(0) = 0.5
    /// ```
    #[verified_engine::verified]
    pub fn link(linear_predictor: f64) -> f64 {
        // expit(γ) = 1/(1 + exp(-γ))
        // Numerically stable version
        if linear_predictor >= 0.0 {
            1.0 / (1.0 + (-linear_predictor).exp())
        } else {
            let exp_gamma = linear_predictor.exp();
            exp_gamma / (1.0 + exp_gamma)
        }
    }

    /// Applies the inverse logit link function: γ = log(ρ/(1-ρ)).
    ///
    /// # Arguments
    ///
    /// * `probability` - The probability ρ (must be in (0, 1))
    ///
    /// # Returns
    ///
    /// The linear predictor γ = logit(ρ)
    #[verified_engine::verified]
    pub fn inverse_link(probability: f64) -> f64 {
        // logit(ρ) = log(ρ/(1-ρ))
        (probability / (1.0 - probability)).ln()
    }

    /// Computes the derivative of the logit link function.
    ///
    /// d(ρ)/d(γ) = ρ(1-ρ)
    #[verified_engine::verified]
    pub fn derivative(probability: f64) -> f64 {
        probability * (1.0 - probability)
    }

    /// Applies the logit link to a vector of linear predictors.
    #[verified_engine::verified]
    pub fn link_vector(linear_predictors: &DVector<f64>) -> DVector<f64> {
        linear_predictors.map(Self::link)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_log_link() {
        // exp(0) = 1
        assert!((LogLink::link(0.0) - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

        // exp(1) ≈ 2.718
        assert!(
            (LogLink::link(1.0) - std::f64::consts::E).abs()
                < math_commons::registry::TOLERANCE_STANDARD
        );

        // exp(2) ≈ 7.389
        assert!((LogLink::link(2.0) - 7.389056099).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_log_link_inverse() {
        let lambda = 5.0;
        let eta = LogLink::inverse_link(lambda);
        let lambda_recovered = LogLink::link(eta);
        assert!((lambda - lambda_recovered).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_log_link_derivative() {
        let eta = 1.5;
        let derivative = LogLink::derivative(eta);
        let lambda = LogLink::link(eta);
        assert!((derivative - lambda).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_logit_link() {
        // expit(0) = 0.5
        assert!((LogitLink::link(0.0) - 0.5).abs() < math_commons::registry::TOLERANCE_STANDARD);

        // expit(large positive) ≈ 1
        assert!(LogitLink::link(10.0) > 0.9999);

        // expit(large negative) ≈ 0
        assert!(LogitLink::link(-10.0) < 0.0001);
    }

    #[test]
    #[verified_engine::verified]
    fn test_logit_link_inverse() {
        let rho = 0.3;
        let gamma = LogitLink::inverse_link(rho);
        let rho_recovered = LogitLink::link(gamma);
        assert!((rho - rho_recovered).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_logit_link_bounds() {
        // Should always return values in [0, 1]
        for gamma in [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
            let rho = LogitLink::link(gamma);
            assert!((0.0..=1.0).contains(&rho));
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_logit_derivative() {
        let rho = 0.3;
        let derivative = LogitLink::derivative(rho);
        // d(ρ)/d(γ) = ρ(1-ρ) = 0.3 * 0.7 = 0.21
        assert!((derivative - 0.21).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_log_link_vector() {
        let predictors = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let rates = LogLink::link_vector(&predictors);

        assert!((rates[0] - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
        assert!(
            (rates[1] - std::f64::consts::E).abs() < math_commons::registry::TOLERANCE_STANDARD
        );
        assert!((rates[2] - 7.389056099).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_logit_link_vector() {
        let predictors = DVector::from_vec(vec![-1.0, 0.0, 1.0]);
        let probs = LogitLink::link_vector(&predictors);

        assert!(probs[0] < 0.5);
        assert!((probs[1] - 0.5).abs() < math_commons::registry::TOLERANCE_STANDARD);
        assert!(probs[2] > 0.5);
    }
}
