//! Shared elemental scalar activation kernels.
//!
//! Provides fundamental scalar mathematical activation kernels and their derivatives
//! used across both matrix operations (in `activations`) and vector operations (in `deep_learning_theory::calculus`).

use std::f64::consts::{E, PI};

/// Elemental Rectified Linear Unit (ReLU) scalar kernel: f(x) = max(0, x).
#[inline]
#[verified_engine::verified]
pub(crate) fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Derivative of ReLU scalar kernel: f'(x) = 1 if x > 0, else 0.
#[inline]
#[verified_engine::verified]
pub(crate) fn relu_prime(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Elemental Sigmoid scalar kernel: \sigma(x) = 1 / (1 + e^{-x}).
#[inline]
#[verified_engine::verified]
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

/// Derivative of Sigmoid scalar kernel: \sigma'(x) = \sigma(x) * (1 - \sigma(x)).
#[inline]
#[verified_engine::verified]
pub(crate) fn sigmoid_prime(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Elemental Hyperbolic Tangent (Tanh) scalar kernel: tanh(x).
#[inline]
#[verified_engine::verified]
pub(crate) fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Derivative of Tanh scalar kernel: 1 - tanh^2(x).
#[inline]
#[verified_engine::verified]
pub(crate) fn tanh_prime(x: f64) -> f64 {
    let t = tanh(x);
    1.0 - t * t
}

/// Elemental GELU (Gaussian Error Linear Unit) scalar kernel approximation:
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
#[verified_engine::verified]
pub(crate) fn gelu(x: f64) -> f64 {
    let sqrt_2_over_pi = (2.0f64 / PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
}

/// Derivative of GELU scalar kernel approximation.
#[inline]
#[verified_engine::verified]
pub(crate) fn gelu_prime(x: f64) -> f64 {
    let sqrt_2_over_pi = (2.0f64 / PI).sqrt();
    let x_cube = x.powi(3);
    let inner = sqrt_2_over_pi * (x + 0.044715 * x_cube);
    let tanh_inner = inner.tanh();
    let sech_sq = 1.0 - tanh_inner * tanh_inner;

    let term1 = 0.5 * (1.0 + tanh_inner);
    let term2 = 0.5 * x * sech_sq * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);

    term1 + term2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_relu_scalar_behavior() {
        assert_eq!(relu(-10.0), 0.0);
        assert_eq!(relu(-0.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(2.5), 2.5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_relu_prime_scalar_behavior() {
        assert_eq!(relu_prime(-5.0), 0.0);
        assert_eq!(relu_prime(0.0), 0.0);
        assert_eq!(relu_prime(3.14), 1.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_sigmoid_scalar_behavior() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-12);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    #[verified_engine::verified]
    fn test_tanh_scalar_behavior() {
        assert_relative_eq!(tanh(0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(tanh_prime(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_gelu_scalar_behavior() {
        assert_relative_eq!(gelu(0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(gelu_prime(0.0), 0.5, epsilon = 1e-12);
    }
}
