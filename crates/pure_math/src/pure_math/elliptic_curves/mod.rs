//! # Elliptic Curves
//!
//! This module provides functionalities related to elliptic curves and modular forms,
//! combining short-Weierstrass group arithmetic over $\mathbb{R}$ and prime fields $\mathbb{F}_p$,
//! Frobenius traces and Hasse-Weil bounds, imaginary quadratic class-number bridges, modular
//! $q$-series expansions, and Florian Breuer's bounds on modular polynomial coefficients.

use std::collections::HashMap;

use thiserror::Error;

pub mod modular_bridge;
mod modular_fp;
pub mod weierstrass;

pub use modular_bridge::*;
pub use weierstrass::*;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum EllipticCurveError {
    #[error("The theorem applies only for i + j < psi(N).")]
    #[allow(missing_docs)]
    InvalidDegree,
    #[error("Modulus p = {p} must be a prime in 5..=1000000000 for short Weierstrass curves.")]
    #[allow(missing_docs)]
    InvalidModulus { p: i64 },
    #[error("Singular Weierstrass curve over F_{p}: 4*{a}^3 + 27*{b}^2 ≡ 0 (mod {p}).")]
    #[allow(missing_docs)]
    SingularCurve { p: i64, a: i64, b: i64 },
    /// Real short-Weierstrass curve has zero discriminant $4a^3 + 27b^2 = 0$.
    #[error("Elliptic curve over R is singular because discriminant 4a^3 + 27b^2 vanishes.")]
    SingularRealCurve,
    #[error("Point ({x}, {y}) does not satisfy y^2 ≡ x^3 + ax + b (mod {p}).")]
    #[allow(missing_docs)]
    PointNotOnCurve { x: i64, y: i64, p: i64 },
    #[error("Caller buffer length {provided} is smaller than required {required} elements.")]
    #[allow(missing_docs)]
    BufferTooSmall { required: usize, provided: usize },
    /// Prime modulus exceeds the interactive lattice enumeration bound.
    #[error("Lattice point enumeration requires prime p <= {max_modulus} (got {modulus}).")]
    ModulusExceedsLatticeCap {
        /// Requested prime modulus.
        modulus: i64,
        /// Maximum supported prime modulus for full lattice enumeration.
        max_modulus: i64,
    },
}

/// Calculates the prime factorization of a given number `n`.
///
/// The result is a map where keys are prime factors and values are their exponents.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::elliptic_curves::prime_factors;
/// let factors = prime_factors(84);
/// assert_eq!(factors.get(&2), Some(&2));
/// assert_eq!(factors.get(&3), Some(&1));
/// assert_eq!(factors.get(&7), Some(&1));
/// assert_eq!(factors.get(&5), None);
/// ```
#[verified_engine::verified]
pub fn prime_factors(mut n: u64) -> HashMap<u64, u32> {
    let mut factors = HashMap::new();
    if n == 0 {
        return factors;
    }
    while n.is_multiple_of(2) {
        *factors.entry(2).or_insert(0) += 1;
        n /= 2;
    }
    let mut i = 3;
    while i * i <= n {
        while n.is_multiple_of(i) {
            *factors.entry(i).or_insert(0) += 1;
            n /= i;
        }
        i += 2;
    }
    if n > 2 {
        *factors.entry(n).or_insert(0) += 1;
    }
    factors
}

/// Computes the function `ψ(N)`, which gives the degree of the classical modular polynomial `Φ_N`.
///
/// The formula is `ψ(N) = N * Π_{p|N} (1 + 1/p)`, where the product is over the distinct prime factors of `N`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::elliptic_curves::psi;
/// assert_eq!(psi(1), 1);
/// assert_eq!(psi(2), 3); // 2 * (1 + 1/2)
/// assert_eq!(psi(6), 12); // 6 * (1 + 1/2) * (1 + 1/3) = 6 * 3/2 * 4/3
/// assert_eq!(psi(5), 6); // 5 * (1 + 1/5)
/// ```
#[verified_engine::verified]
pub fn psi(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let factors = prime_factors(n);
    let mut result = n;
    for (p, _) in factors {
        result = result / p * (p + 1);
    }
    result
}

/// Represents the lower bounds on p-adic valuations from Theorem 1.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Theorem11Bounds {
    #[allow(missing_docs)]
    pub v2_bound: Option<u64>,
    #[allow(missing_docs)]
    pub v3_bound: Option<u64>,
    #[allow(missing_docs)]
    pub v5_bound: Option<u64>,
}

/// Calculates the lower bounds on the p-adic valuations of the coefficients of `Φ_N(X,Y)`
/// as described in Theorem 1.1 of Breuer's paper.
///
/// The theorem provides bounds for primes p=2, 3, and 5.
/// Let `a_{i,j}` be a coefficient of `Φ_N(X,Y)`.
///
/// - If `2 nmid N`, then `v_2(a_{i,j}) >= 15 * (ψ(N) - i - j)`.
/// - If `3 nmid N`, then `v_3(a_{i,j}) >= 3 * (ψ(N) - i - j)`.
///   - If `N ≡ 1 mod 3`, the bound is `ceil(9/2 * (ψ(N) - i - j))`.
/// - If `5 nmid N`, then `v_5(a_{i,j}) >= 3 * (ψ(N) - i - j)`.
///
/// The function returns `None` for a given prime if the condition on `N` is not met.
///
/// # Errors
/// Returns `EllipticCurveError::InvalidDegree` if `i + j >= ψ(N)`,
/// as the theorem only applies for `i + j < ψ(N)`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::elliptic_curves::{theorem_1_1_bounds, Theorem11Bounds};
/// // For N=5, psi(5) = 6. Let's check a_{1,1} where i+j=2.
/// let bounds = theorem_1_1_bounds(5, 1, 1).unwrap();
/// assert_eq!(bounds.v2_bound, Some(15 * (6 - 2))); // 2 does not divide 5
/// assert_eq!(bounds.v3_bound, Some(3 * (6 - 2))); // 3 does not divide 5
/// assert_eq!(bounds.v5_bound, None); // 5 divides 5
///
/// // For N=7, psi(7) = 8. N=7 is 1 mod 3. Check a_{2,1} where i+j=3.
/// let bounds_7 = theorem_1_1_bounds(7, 2, 1).unwrap();
/// assert_eq!(bounds_7.v2_bound, Some(15 * (8 - 3)));
/// // N=7 is 1 mod 3, so we use the ceil(9/2 * ...) formula.
/// // ceil(4.5 * 5) = ceil(22.5) = 23
/// assert_eq!(bounds_7.v3_bound, Some(23));
/// assert_eq!(bounds_7.v5_bound, Some(3 * (8 - 3)));
/// ```
#[verified_engine::verified]
pub fn theorem_1_1_bounds(n: u64, i: u64, j: u64) -> Result<Theorem11Bounds, EllipticCurveError> {
    let psi_n = psi(n);
    if i + j >= psi_n {
        return Err(EllipticCurveError::InvalidDegree);
    }
    let diff = psi_n - i - j;

    let v2 = if !n.is_multiple_of(2) {
        Some(15 * diff)
    } else {
        None
    };
    let v3 = if !n.is_multiple_of(3) {
        if n % 3 == 1 {
            // Integer arithmetic for ceil(9 * diff / 2)
            Some((9 * diff).div_ceil(2))
        } else {
            Some(3 * diff)
        }
    } else {
        None
    };
    let v5 = if !n.is_multiple_of(5) {
        Some(3 * diff)
    } else {
        None
    };

    Ok(Theorem11Bounds {
        v2_bound: v2,
        v3_bound: v3,
        v5_bound: v5,
    })
}

/// Represents the lower bounds on p-adic valuations from Theorem 1.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Theorem12Bounds {
    #[allow(missing_docs)]
    pub v2_bound: Option<u64>,
    #[allow(missing_docs)]
    pub v3_bound: Option<u64>,
    #[allow(missing_docs)]
    pub v7_bound: Option<u64>,
}

/// Calculates the lower bounds on the p-adic valuations of the coefficients of `Φ_N(X+1728, Y+1728)`
/// as described in Theorem 1.2 of Breuer's paper.
///
/// Let `a_{i,j}` be a coefficient of `Φ_N(X+1728, Y+1728)`.
///
/// - If `2 nmid N`, then `v_2(a_{i,j}) >= 9 * (ψ(N) - i - j)`.
///   - If `N ≡ 1 mod 4`, the bound is `10 * (ψ(N) - i - j)`.
/// - If `3 nmid N`, then `v_3(a_{i,j}) >= 6 * (ψ(N) - i - j)`.
/// - If `7 nmid N`, then `v_7(a_{i,j}) >= 2 * (ψ(N) - i - j)`.
///
/// # Errors
/// Returns `EllipticCurveError::InvalidDegree` if `i + j >= ψ(N)`,
/// as the theorem only applies for `i + j < ψ(N)`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::elliptic_curves::{theorem_1_2_bounds, Theorem12Bounds};
/// // For N=5, psi(5) = 6. N=5 is 1 mod 4. Check a_{0,0}.
/// let bounds = theorem_1_2_bounds(5, 0, 0).unwrap();
/// assert_eq!(bounds.v2_bound, Some(10 * (6 - 0)));
/// assert_eq!(bounds.v3_bound, Some(6 * (6 - 0)));
/// assert_eq!(bounds.v7_bound, Some(2 * (6 - 0)));
/// ```
#[verified_engine::verified]
pub fn theorem_1_2_bounds(n: u64, i: u64, j: u64) -> Result<Theorem12Bounds, EllipticCurveError> {
    let psi_n = psi(n);
    if i + j >= psi_n {
        return Err(EllipticCurveError::InvalidDegree);
    }
    let diff = psi_n - i - j;

    let v2 = if !n.is_multiple_of(2) {
        if n % 4 == 1 {
            Some(10 * diff)
        } else {
            Some(9 * diff)
        }
    } else {
        None
    };
    let v3 = if !n.is_multiple_of(3) {
        Some(6 * diff)
    } else {
        None
    };
    let v7 = if !n.is_multiple_of(7) {
        Some(2 * diff)
    } else {
        None
    };

    Ok(Theorem12Bounds {
        v2_bound: v2,
        v3_bound: v3,
        v7_bound: v7,
    })
}

// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]
