//! # Q-Series
//!
//! This module implements a structure for representing and manipulating q-series
//! (power series in q), which are fundamental in the theory of partitions and
//! other areas of number theory.

use crate::error::NumberTheoryError;
use crate::pure_math::algebra::traits::EuclideanDomain;
use std::ops::{Add, Div, Mul};

/// Represents a q-series, a power series in q.
/// The vector `coeffs` stores the coefficients, where the index represents the power of q.
/// The coefficient type T must be a Euclidean Domain to support division.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QSeries<T: EuclideanDomain> {
    #[allow(missing_docs)]
    pub coeffs: Vec<T>,
}

impl<T: EuclideanDomain> Default for QSeries<T> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: EuclideanDomain> QSeries<T> {
    /// Creates a new empty QSeries.
    #[verified_engine::verified]
    pub fn new() -> Self {
        QSeries { coeffs: vec![] }
    }

    /// Creates a QSeries with a given capacity.
    #[verified_engine::verified]
    pub fn with_capacity(capacity: usize) -> Self {
        QSeries {
            coeffs: Vec::with_capacity(capacity),
        }
    }

    /// Creates a QSeries from a vector of coefficients.
    #[verified_engine::verified]
    pub fn from_vec(coeffs: Vec<T>) -> Self {
        QSeries { coeffs }
    }

    /// Gets the coefficient of q^n.
    #[verified_engine::verified]
    pub fn get_coeff(&self, n: usize) -> T {
        self.coeffs.get(n).cloned().unwrap_or_else(T::zero)
    }

    /// Truncates the series to a given precision.
    #[verified_engine::verified]
    pub fn truncate(&mut self, precision: usize) {
        self.coeffs.truncate(precision);
    }

    /// Computes the power of a QSeries using exponentiation by squaring.
    #[verified_engine::verified]
    pub fn pow(&self, exp: u32) -> QSeries<T> {
        let precision = self.coeffs.len();
        if exp == 0 {
            let mut coeffs = vec![T::zero(); precision];
            if precision > 0 {
                coeffs[0] = T::one();
            }
            return QSeries::from_vec(coeffs);
        }

        let mut base = self.clone();
        let mut e = exp;

        let mut result_coeffs = vec![T::zero(); precision];
        if precision > 0 {
            result_coeffs[0] = T::one();
        }
        let mut result = QSeries::from_vec(result_coeffs);

        while e > 0 {
            if e % 2 == 1 {
                result = &result * &base;
            }
            base = &base * &base;
            e /= 2;
        }
        result
    }

    /// Divides `self` by `other` as truncated formal power series without panicking.
    ///
    /// # Errors
    ///
    /// Returns [`NumberTheoryError::DivisionByZeroQSeries`] when `other` has no coefficients,
    /// or [`NumberTheoryError::DivisionByZeroConstantTerm`] when `other`'s constant term is zero.
    #[verified_engine::verified]
    pub fn checked_div(&self, other: &Self) -> Result<QSeries<T>, NumberTheoryError> {
        if other.coeffs.is_empty() {
            return Err(NumberTheoryError::DivisionByZeroQSeries);
        }
        let b0 = other.get_coeff(0);
        if b0.is_zero() {
            return Err(NumberTheoryError::DivisionByZeroConstantTerm);
        }

        let precision = self.coeffs.len();
        let mut new_coeffs = vec![T::zero(); precision];

        for n in 0..precision {
            let mut sum_val = T::zero();
            for (i, coeff) in new_coeffs.iter().enumerate().take(n) {
                sum_val += coeff.clone() * other.get_coeff(n - i);
            }

            let numerator = self.get_coeff(n) - sum_val;
            new_coeffs[n] = numerator / b0.clone();
        }

        Ok(QSeries { coeffs: new_coeffs })
    }
}

impl<T: EuclideanDomain> Add for &QSeries<T> {
    type Output = QSeries<T>;

    #[verified_engine::verified]
    fn add(self, other: Self) -> QSeries<T> {
        let len1 = self.coeffs.len();
        let len2 = other.coeffs.len();
        let max_len = std::cmp::max(len1, len2);
        let mut new_coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            new_coeffs.push(self.get_coeff(i) + other.get_coeff(i));
        }

        QSeries { coeffs: new_coeffs }
    }
}

impl<T: EuclideanDomain> Mul for &QSeries<T> {
    type Output = QSeries<T>;

    #[verified_engine::verified]
    fn mul(self, other: Self) -> QSeries<T> {
        let len1 = self.coeffs.len();
        let len2 = other.coeffs.len();
        if len1 == 0 || len2 == 0 {
            return QSeries::new();
        }
        let precision = std::cmp::max(len1, len2);
        let mut new_coeffs = vec![T::zero(); precision];

        for i in 0..len1 {
            // Hoist coefficient cloning out of the inner loop
            let c_i = self.coeffs[i].clone();

            // Calculate the limit for the inner loop to avoid branching
            // We need i + j < precision, so j < precision - i
            let limit = if i < precision {
                std::cmp::min(len2, precision - i)
            } else {
                0
            };

            for j in 0..limit {
                let product = c_i.clone() * other.coeffs[j].clone();
                new_coeffs[i + j] += product;
            }
        }

        QSeries { coeffs: new_coeffs }
    }
}

impl<T: EuclideanDomain> Div for &QSeries<T> {
    type Output = QSeries<T>;

    /// Divides `self` by `other` as formal power series.
    ///
    /// # Panics
    ///
    /// Panics if `other` is empty or has a zero constant term. Use [`QSeries::checked_div`]
    /// for fallible division.
    #[verified_engine::verified]
    fn div(self, other: Self) -> QSeries<T> {
        match self.checked_div(other) {
            Ok(series) => series,
            Err(e) => panic!("{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_qseries_add() {
        let s1 = QSeries::from_vec(vec![1i64, 2, 3]);
        let s2 = QSeries::from_vec(vec![4i64, 5, 6, 7]);
        let s3 = &s1 + &s2;
        assert_eq!(s3.coeffs, vec![5, 7, 9, 7]);
    }

    #[test]
    #[verified_engine::verified]
    fn test_qseries_mul() {
        let s1 = QSeries::from_vec(vec![1i64, 1]); // 1+q
        let s2 = QSeries::from_vec(vec![1i64, 1]); // 1+q
        let s3 = &s1 * &s2; // (1+q)^2 = 1+2q+q^2
        assert_eq!(s3.coeffs, vec![1, 2]); // Truncated to precision 2 based on input length

        let s4 = QSeries::from_vec(vec![1i64, 1, 1]); // 1+q+q^2
        let s5 = QSeries::from_vec(vec![1i64, -1, 0]); // 1-q
        let s6 = &s4 * &s5; // (1-q^3) = 1
        assert_eq!(s6.coeffs, vec![1, 0, 0]);
    }

    #[test]
    #[verified_engine::verified]
    fn test_qseries_div() {
        // 1 / (1-q) = 1+q+q^2+...
        let one = QSeries::from_vec(vec![1i64, 0, 0, 0, 0]);
        let one_minus_q = QSeries::from_vec(vec![1i64, -1]);
        let geom_series = &one / &one_minus_q;
        assert_eq!(geom_series.coeffs, vec![1, 1, 1, 1, 1]);
    }
}
