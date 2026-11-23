//! Shared elemental scalar activation kernels.
//!
//! Provides fundamental scalar mathematical activation kernels and their derivatives
//! used across both matrix operations (in `activations`) and vector operations (in `deep_learning_theory::calculus`).

use nalgebra::storage::{RawStorage, RawStorageMut};
use nalgebra::{DVector, Dim, Matrix, RealField};
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

/// Computes numerically stable softmax from `src` into `dst` in place,
/// enforcing empty-input and non-positive/`NaN` sum zero-output guards.
#[inline]
#[verified_engine::verified]
pub(crate) fn softmax_into<T, R1, C1, S1, R2, C2, S2>(
    src: &Matrix<T, R1, C1, S1>,
    dst: &mut Matrix<T, R2, C2, S2>,
) where
    T: RealField + Copy,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorageMut<T, R2, C2>,
{
    assert_eq!(
        src.len(),
        dst.len(),
        "softmax_into source and destination lengths must match"
    );
    let mut src_iter = src.iter();
    let Some(&first) = src_iter.next() else {
        return;
    };

    let mut max_val = first;
    for &val in src_iter {
        if val > max_val {
            max_val = val;
        }
    }

    let zero = T::zero();
    let mut sum_exps = zero;
    for (&val, out) in src.iter().zip(dst.iter_mut()) {
        let exp_val = (val - max_val).exp();
        *out = exp_val;
        sum_exps += exp_val;
    }

    if sum_exps > zero {
        for out in dst.iter_mut() {
            *out /= sum_exps;
        }
    } else {
        dst.fill(zero);
    }
}

/// Computes a guarded softmax probability vector with a single output allocation.
#[inline]
#[verified_engine::verified]
pub(crate) fn softmax<T: RealField + Copy>(z: &DVector<T>) -> DVector<T> {
    let mut out = DVector::zeros(z.len());
    softmax_into(z, &mut out);
    out
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
        assert_eq!(relu_prime(3.5), 1.0);
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

    #[test]
    #[verified_engine::verified]
    fn test_softmax_kernel_guards_and_normalization() {
        let empty = DVector::<f64>::zeros(0);
        assert_eq!(softmax(&empty).len(), 0);

        let nan_vec = DVector::from_vec(vec![0.5, f64::NAN, -1.0]);
        assert_eq!(softmax(&nan_vec), DVector::zeros(3));

        let neg_inf_vec = DVector::from_vec(vec![f64::NEG_INFINITY, f64::NEG_INFINITY]);
        assert_eq!(softmax(&neg_inf_vec), DVector::zeros(2));

        let uniform = DVector::from_vec(vec![3.0, 3.0, 3.0, 3.0]);
        let probs = softmax(&uniform);
        for &p in probs.iter() {
            assert_relative_eq!(p, 0.25, epsilon = math_commons::registry::TOLERANCE_FAST);
        }
    }
}
