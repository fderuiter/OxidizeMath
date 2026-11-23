//! # Finite Fields
//!
//! Implementation of Finite Fields (Integers Modulo Prime P).

use crate::error::NumberTheoryError;
use crate::pure_math::algebra::traits::{Field, Ring};
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================================
// Finite Fields (Integers Modulo Prime P)
// ============================================================================

/// A generic finite field $\mathbb{F}_p$.
/// P must be prime. This is not checked at compile time but expected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp<const P: i64> {
    #[allow(missing_docs)]
    pub value: i64,
}

impl<const P: i64> Fp<P> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(value: i64) -> Self {
        Fp {
            value: value.rem_euclid(P),
        }
    }

    /// Computes the multiplicative inverse of `self` modulo `P` without panicking.
    ///
    /// # Errors
    ///
    /// Returns [`NumberTheoryError::ZeroModulus`] if `P <= 0`,
    /// [`NumberTheoryError::DivisionByZeroFieldElement`] if `self` is zero, or
    /// [`NumberTheoryError::NoModularInverse`] if `gcd(self.value, P) != 1`.
    #[verified_engine::verified]
    pub fn checked_inverse(&self) -> Result<Self, NumberTheoryError> {
        if P <= 0 {
            return Err(NumberTheoryError::ZeroModulus);
        }
        if self.is_zero() {
            return Err(NumberTheoryError::DivisionByZeroFieldElement);
        }
        let (g, x, _) = extended_gcd(self.value, P);
        if g != 1 {
            return Err(NumberTheoryError::NoModularInverse);
        }
        Ok(Fp::new(x))
    }
}

// Implement Operations
impl<const P: i64> Add for Fp<P> {
    type Output = Self;
    #[verified_engine::verified]
    fn add(self, other: Self) -> Self {
        Fp::new(self.value + other.value)
    }
}

impl<const P: i64> AddAssign for Fp<P> {
    #[verified_engine::verified]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const P: i64> Sub for Fp<P> {
    type Output = Self;
    #[verified_engine::verified]
    fn sub(self, other: Self) -> Self {
        Fp::new(self.value - other.value)
    }
}

impl<const P: i64> SubAssign for Fp<P> {
    #[verified_engine::verified]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<const P: i64> Mul for Fp<P> {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(self, other: Self) -> Self {
        Fp::new(self.value * other.value)
    }
}

impl<const P: i64> MulAssign for Fp<P> {
    #[verified_engine::verified]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<const P: i64> Neg for Fp<P> {
    type Output = Self;
    #[verified_engine::verified]
    fn neg(self) -> Self {
        Fp::new(-self.value)
    }
}

impl<const P: i64> Ring for Fp<P> {
    #[verified_engine::verified]
    fn zero() -> Self {
        Fp::new(0)
    }
    #[verified_engine::verified]
    fn one() -> Self {
        Fp::new(1)
    }
}

impl<const P: i64> Div for Fp<P> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[verified_engine::verified]
    fn div(self, other: Self) -> Self {
        self * other.multiplicative_inverse()
    }
}

impl<const P: i64> Field for Fp<P> {
    #[verified_engine::verified]
    fn multiplicative_inverse(&self) -> Self {
        match self.checked_inverse() {
            Ok(inv) => inv,
            Err(NumberTheoryError::DivisionByZeroFieldElement) => {
                panic!("Division by zero in Fp")
            }
            Err(_) => panic!("Element has no inverse (P might not be prime)"),
        }
    }
}

// Helper for Extended Euclidean Algorithm
#[verified_engine::verified]
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    let mut s = 0;
    let mut old_s = 1;
    let mut t = 1;
    let mut old_t = 0;
    let mut r = b;
    let mut old_r = a;

    while r != 0 {
        let quotient = old_r / r;

        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;

        let temp_t = t;
        t = old_t - quotient * t;
        old_t = temp_t;
    }

    (old_r, old_s, old_t)
}

impl<const P: i64> fmt::Display for Fp<P> {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}
