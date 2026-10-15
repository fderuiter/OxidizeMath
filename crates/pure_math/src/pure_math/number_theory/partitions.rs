// [cite:partitions_implementation]
//! # Partition Functions
//!
//! This module implements restricted partition functions based on the work of Pushpa and Vasuki.
//! It focuses on generating functions expressed as products of the function $f_k$.
//!
//! ## Mathematical Background
//!
//! The fundamental building block is the q-product:
//! $$f_k = (q^k; q^k)_\infty = \prod_{n=1}^{\infty} (1 - q^{kn})$$
//!
//! The module implements several partition functions defined in terms of $f_k$, such as $P^*(n)$, $M(n)$, and $T^*(n)$.
//! For example, the generating function for $P^*(n)$ is given by:
//! $$\sum_{n=0}^{\infty} P^*(n)q^n = f_1^4 f_5^4$$
//!
//! These identities relate to Ramanujan's theta functions and have properties modulo primes (e.g., congruences).
//!
//! ##  Quick Start
//!
//! Compute the first few coefficients of the partition function $P^*(n)$.
//!
//! ```rust
//! use pure_math::pure_math::number_theory::partitions::{gen_p_star, QSeries};
//!
//! // Calculate coefficients up to q^10
//! let precision = 11;
//! let p_star = gen_p_star(precision);
//!
//! // The coefficient at index n is P*(n)
//! println!("P*(0) = {}", p_star.get_coeff(0));
//! println!("P*(1) = {}", p_star.get_coeff(1));
//!
//! assert_eq!(p_star.get_coeff(0), 1);
//! ```

// Define QSeries as a type alias for QSeries<i64> to preserve backward compatibility
// and allow specific usage in this module.
#[allow(missing_docs)]
pub type QSeries = crate::pure_math::number_theory::q_series::QSeries<i64>;

/// Computes the q-series for $f_k = (q^k; q^k)_\infty$ up to a given precision.
///
/// $$f_k = \prod_{i=1}^{\infty} (1 - q^{ki})$$
///
/// This implementation uses **Euler's Pentagonal Number Theorem** to compute the series
/// in $O(\sqrt{N})$ time, which is significantly faster than expanding the product term-by-term.
///
/// # Arguments
///
/// * `k` - The scaling factor for the power of q.
/// * `precision` - The number of terms to compute (i.e., maximum power of q + 1).
#[verified_engine::verified]
pub fn f_k(k: usize, precision: usize) -> QSeries {
    if k == 0 {
        return QSeries::from_vec(vec![0i64; precision]);
    }
    if precision == 0 {
        return QSeries::new();
    }

    // We want to construct the series sum_{m=-inf}^{inf} (-1)^m q^{k * m(3m-1)/2}
    // The pentagonal numbers are p_m = m(3m-1)/2 for m = 0, 1, -1, 2, -2, ...
    // m = 0 -> p = 0, coeff = 1
    // m = 1 -> p = 1, coeff = -1
    // m = -1 -> p = 2, coeff = -1
    // m = 2 -> p = 5, coeff = 1
    // m = -2 -> p = 7, coeff = 1

    let mut coeffs = vec![0i64; precision];

    // m = 0 case
    coeffs[0] = 1;

    // Iterate m from 1 upwards.
    // We handle pairs m and -m together.
    // Generalized pentagonal numbers: p(m) = m(3m-1)/2
    // p(m) = (3m^2 - m)/2
    // p(-m) = (-m(-3m-1))/2 = (3m^2 + m)/2

    let mut m = 1;
    loop {
        let p_pos = (m * (3 * m - 1)) / 2;
        let p_neg = (m * (3 * m + 1)) / 2;

        // The power in q is k * p
        let idx_pos = k.checked_mul(p_pos as usize);
        let idx_neg = k.checked_mul(p_neg as usize);

        let mut added = false;

        let sign = if m % 2 == 0 { 1 } else { -1 };

        if let Some(idx) = idx_pos
            && idx < precision
        {
            coeffs[idx] = sign;
            added = true;
        }

        if let Some(idx) = idx_neg
            && idx < precision
        {
            coeffs[idx] = sign;
            added = true;
        }

        if !added {
            // Since p(m) grows quadratically, if both p_pos and p_neg exceed precision,
            // all subsequent terms will also exceed precision.
            break;
        }

        m += 1;
    }

    QSeries { coeffs }
}

/// Generating function for $P^*(n)$.
///
/// $$\sum P^*(n)q^n = f_1^4 f_5^4$$
#[verified_engine::verified]
pub fn gen_p_star(precision: usize) -> QSeries {
    let f1 = f_k(1, precision);
    let f5 = f_k(5, precision);
    let f1_pow4 = f1.pow(4);
    let f5_pow4 = f5.pow(4);
    &f1_pow4 * &f5_pow4
}

/// Generating function for $M(n)$.
///
/// $$\sum M(n)q^n = \frac{f_2^5 f_5^5}{f_1 f_{10}}$$
#[verified_engine::verified]
pub fn gen_m(precision: usize) -> QSeries {
    let f1 = f_k(1, precision);
    let f2 = f_k(2, precision);
    let f5 = f_k(5, precision);
    let f10 = f_k(10, precision);

    let f2_pow5 = f2.pow(5);
    let f5_pow5 = f5.pow(5);

    let numerator = &f2_pow5 * &f5_pow5;
    let denominator = &f1 * &f10;

    &numerator / &denominator
}

/// Generating function for $T^*(n)$.
///
/// $$\sum T^*(n)q^n = \frac{f_1^5 f_{10}^5}{f_2 f_5}$$
#[verified_engine::verified]
pub fn gen_t_star(precision: usize) -> QSeries {
    let f1 = f_k(1, precision);
    let f2 = f_k(2, precision);
    let f5 = f_k(5, precision);
    let f10 = f_k(10, precision);

    let f1_pow5 = f1.pow(5);
    let f10_pow5 = f10.pow(5);

    let numerator = &f1_pow5 * &f10_pow5;
    let denominator = &f2 * &f5;

    &numerator / &denominator
}

/// Generating function for $A(n)$.
///
/// $$\sum A(n)q^n = \frac{f_2^6 f_7^6}{f_1^2}$$
#[verified_engine::verified]
pub fn gen_a(precision: usize) -> QSeries {
    let f1 = f_k(1, precision);
    let f2 = f_k(2, precision);
    let f7 = f_k(7, precision);

    let f2_pow6 = f2.pow(6);
    let f7_pow6 = f7.pow(6);
    let f1_pow2 = f1.pow(2);

    let numerator = &f2_pow6 * &f7_pow6;

    &numerator / &f1_pow2
}

/// Generating function for $B(n)$.
///
/// $$\sum B(n)q^n = \frac{f_1^6 f_{14}^4}{f_2^2 f_7^2}$$
#[verified_engine::verified]
pub fn gen_b(precision: usize) -> QSeries {
    let f1 = f_k(1, precision);
    let f2 = f_k(2, precision);
    let f7 = f_k(7, precision);
    let f14 = f_k(14, precision);

    let f1_pow6 = f1.pow(6);
    let f14_pow4 = f14.pow(4);
    let f2_pow2 = f2.pow(2);
    let f7_pow2 = f7.pow(2);

    let numerator = &f1_pow6 * &f14_pow4;
    let denominator = &f2_pow2 * &f7_pow2;

    &numerator / &denominator
}

/// Generating function for $K(n)$.
///
/// $$\sum K(n)q^n = f_1^2 f_2^2 f_7^2 f_{14}^2$$
#[verified_engine::verified]
pub fn gen_k(precision: usize) -> QSeries {
    let f1 = f_k(1, precision);
    let f2 = f_k(2, precision);
    let f7 = f_k(7, precision);
    let f14 = f_k(14, precision);

    let f1_pow2 = f1.pow(2);
    let f2_pow2 = f2.pow(2);
    let f7_pow2 = f7.pow(2);
    let f14_pow2 = f14.pow(2);

    &(&(&f1_pow2 * &f2_pow2) * &f7_pow2) * &f14_pow2
}

/// Generating function for $L(n)$.
///
/// $$\sum L(n)q^n = \frac{f_1^5 f_7^5}{f_2 f_{14}}$$
#[verified_engine::verified]
pub fn gen_l(precision: usize) -> QSeries {
    let f1 = f_k(1, precision);
    let f2 = f_k(2, precision);
    let f7 = f_k(7, precision);
    let f14 = f_k(14, precision);

    let f1_pow5 = f1.pow(5);
    let f7_pow5 = f7.pow(5);

    let numerator = &f1_pow5 * &f7_pow5;
    let denominator = &f2 * &f14;

    &numerator / &denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    // Re-implementation of the naive O(N^2) algorithm for verification purposes
    #[verified_engine::verified]
    fn f_k_slow(k: usize, precision: usize) -> QSeries {
        if k == 0 {
            return QSeries::from_vec(vec![0i64; precision]);
        }
        if precision == 0 {
            return QSeries::new();
        }
        if k >= precision {
            let mut coeffs = vec![0i64; precision];
            coeffs[0] = 1;
            return QSeries { coeffs };
        }
        let mut coeffs = vec![0i64; precision];
        coeffs[0] = 1;

        for i in 1.. {
            let power = i * k;
            if power >= precision {
                break;
            }
            // This is multiplication by (1 - q^power)
            for j in (power..precision).rev() {
                coeffs[j] -= coeffs[j - power];
            }
        }
        QSeries { coeffs }
    }

    #[test]
    #[verified_engine::verified]
    fn test_f_k_correctness() {
        let precision = 100;
        let k = 1;

        let fast = f_k(k, precision);
        let slow = f_k_slow(k, precision);

        assert_eq!(
            fast.coeffs, slow.coeffs,
            "Optimized f_k does not match naive implementation"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_f_k_large_k() {
        let precision = 100;
        let k = 5; // Pentagonal numbers will be scaled by 5

        let fast = f_k(k, precision);
        let slow = f_k_slow(k, precision);

        assert_eq!(
            fast.coeffs, slow.coeffs,
            "Optimized f_k does not match naive implementation for k=5"
        );
    }
}
