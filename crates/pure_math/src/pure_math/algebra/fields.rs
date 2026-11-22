//! # Finite Fields
//!
//! Implementation of Finite Fields (Integers Modulo Prime P).

use crate::pure_math::algebra::traits::{Field, Ring};
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================================
// Finite Fields (Integers Modulo Prime P)
// ============================================================================

/// A generic finite field $\mathbb{F}_p$.
/// P must be prime. This is not checked at compile time but expected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Fp<const P: i64> {
    #[allow(missing_docs)]
    pub value: i64,
}

#[cfg(feature = "serde")]
impl<'de, const P: i64> serde::Deserialize<'de> for Fp<P> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct FpRaw {
            value: i64,
        }
        if P <= 1 {
            return Err(serde::de::Error::custom("Fp modulus P must be > 1"));
        }
        let raw = FpRaw::deserialize(deserializer)?;
        Ok(Self::new(raw.value))
    }
}

impl<const P: i64> Fp<P> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(value: i64) -> Self {
        Fp {
            value: value.rem_euclid(P),
        }
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
        if self.is_zero() {
            panic!("Division by zero in Fp");
        }
        // Extended Euclidean Algorithm to find inverse
        let (g, x, _) = extended_gcd(self.value, P);
        if g != 1 {
            panic!("Element has no inverse (P might not be prime)");
        }
        Fp::new(x)
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
