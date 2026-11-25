#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
//! Arbitrary-precision `#![no_std]` + `alloc` integer ([`BigInt`]), exact rational
//! ([`BigRational`]), and precision-preserving binary floating-point ([`BigFloat`]) types
//! backed by `std`-free `dashu`.

use crate::error::ScalarError;
use crate::fixed256::U256;
use crate::fixed256_signed::I256;
use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use dashu::base::{Abs, DivEuclid, RemEuclid};
use dashu::float::FBig;
use dashu::float::round::mode::HalfAway;
use dashu::integer::{IBig, UBig};
use dashu::rational::RBig;

type BinaryFloat = FBig<HalfAway, 2>;

/// Arbitrary-precision signed integer backed by `alloc` (`dashu::integer::IBig`).
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BigInt(IBig);

impl BigInt {
    /// Additive identity $0$.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self(IBig::ZERO)
    }

    /// Multiplicative identity $1$.
    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self(IBig::ONE)
    }

    /// Wraps a raw `dashu::integer::IBig`.
    #[inline]
    #[must_use]
    pub const fn from_ibig(inner: IBig) -> Self {
        Self(inner)
    }

    /// Borrows the underlying `dashu::integer::IBig`.
    #[inline]
    #[must_use]
    pub const fn as_ibig(&self) -> &IBig {
        &self.0
    }

    /// Consumes `self` and returns the inner `dashu::integer::IBig`.
    #[inline]
    #[must_use]
    pub fn into_ibig(self) -> IBig {
        self.0
    }

    /// Widens a stack [`U256`] into an exact `BigInt`.
    #[must_use]
    pub fn from_u256(val: U256) -> Self {
        let limbs = val.limbs();
        let lo = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
        let hi = (limbs[2] as u128) | ((limbs[3] as u128) << 64);
        Self((IBig::from(hi) << 128) | IBig::from(lo))
    }

    /// Widens a stack [`I256`] into an exact `BigInt`.
    #[must_use]
    pub fn from_i256(val: I256) -> Self {
        let mag = Self::from_u256(val.unsigned_abs());
        if val.is_negative() { -mag } else { mag }
    }

    /// Parses an integer string in base `radix` (`2..=36`).
    pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, ScalarError> {
        if !(2..=36).contains(&radix) {
            return Err(ScalarError::InvalidRadix { radix });
        }
        IBig::from_str_radix(src, radix)
            .map(Self)
            .map_err(|_| ScalarError::InvalidLiteral)
    }

    /// Returns `true` if `self == 0`.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Absolute value $|n|$.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self(self.0.clone().abs())
    }

    /// Euclidean quotient $q$ with non-negative remainder.
    pub fn checked_div_euclid(&self, rhs: &Self) -> Result<Self, ScalarError> {
        if rhs.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        Ok(Self((&self.0).div_euclid(&rhs.0)))
    }

    /// Euclidean remainder $r \in [0, |b|)$.
    pub fn checked_rem_euclid(&self, rhs: &Self) -> Result<Self, ScalarError> {
        if rhs.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        Ok(Self((&self.0).rem_euclid(&rhs.0).into()))
    }

    /// Converts `self` to `i64` if within bounds.
    pub fn to_i64(&self) -> Result<i64, ScalarError> {
        i64::try_from(&self.0).map_err(|_| ScalarError::ArithmeticOverflow)
    }

    /// Approximates `self` as `f64`.
    #[inline]
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().value()
    }
}

impl From<i64> for BigInt {
    #[inline]
    fn from(v: i64) -> Self {
        Self(IBig::from(v))
    }
}

impl From<u64> for BigInt {
    #[inline]
    fn from(v: u64) -> Self {
        Self(IBig::from(v))
    }
}

impl From<i128> for BigInt {
    #[inline]
    fn from(v: i128) -> Self {
        Self(IBig::from(v))
    }
}

impl From<u128> for BigInt {
    #[inline]
    fn from(v: u128) -> Self {
        Self(IBig::from(v))
    }
}

impl From<UBig> for BigInt {
    #[inline]
    fn from(v: UBig) -> Self {
        Self(IBig::from(v))
    }
}

impl From<U256> for BigInt {
    #[inline]
    fn from(v: U256) -> Self {
        Self::from_u256(v)
    }
}

impl From<I256> for BigInt {
    #[inline]
    fn from(v: I256) -> Self {
        Self::from_i256(v)
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Add for BigInt {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for BigInt {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for BigInt {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for BigInt {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl Rem for BigInt {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Self(self.0 % rhs.0)
    }
}

impl Neg for BigInt {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Exact reduced rational $p / q \in \mathbb{Q}$ ($q > 0$, $\gcd(|p|, q) = 1$) backed by `alloc`.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BigRational(RBig);

impl BigRational {
    /// Constructs reduced $p / q$, returning [`ScalarError::ZeroDenominator`] when $q = 0$.
    pub fn try_new(numer: BigInt, denom: BigInt) -> Result<Self, ScalarError> {
        if denom.is_zero() {
            return Err(ScalarError::ZeroDenominator);
        }
        Ok(Self(RBig::from_parts_signed(numer.0, denom.0)))
    }

    /// Embeds an exact [`BigInt`] as $n / 1$.
    #[inline]
    #[must_use]
    pub fn from_bigint(n: BigInt) -> Self {
        Self(RBig::from(n.0))
    }

    /// Wraps a `dashu::rational::RBig`.
    #[inline]
    #[must_use]
    pub const fn from_rbig(inner: RBig) -> Self {
        Self(inner)
    }

    /// Borrows the underlying `dashu::rational::RBig`.
    #[inline]
    #[must_use]
    pub const fn as_rbig(&self) -> &RBig {
        &self.0
    }

    /// Exact reduced signed numerator $p \in \mathbb{Z}$ without `f64` narrowing.
    #[inline]
    #[must_use]
    pub fn numerator(&self) -> BigInt {
        BigInt(self.0.numerator().clone())
    }

    /// Exact reduced positive denominator $q \in \mathbb{Z}_{>0}$ without `f64` narrowing.
    #[inline]
    #[must_use]
    pub fn denominator(&self) -> BigInt {
        BigInt(IBig::from(self.0.denominator().clone()))
    }

    /// Returns `true` if `self == 0`.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Exact absolute value $|p / q|$.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self(self.0.clone().abs())
    }

    /// Fallible multiplicative inverse $q / p$.
    pub fn try_inverse(&self) -> Result<Self, ScalarError> {
        if self.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        Self::try_new(self.denominator(), self.numerator())
    }

    /// Closest IEEE-754 `f64` approximation.
    #[inline]
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().value()
    }
}

impl fmt::Display for BigRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Add for BigRational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for BigRational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for BigRational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Div for BigRational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl Neg for BigRational {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Arbitrary-precision binary floating-point number preserving caller-specified mantissa bits.
#[derive(Clone, Debug)]
pub struct BigFloat(BinaryFloat);

impl BigFloat {
    /// Constructs a binary float with `prec` mantissa bits from finite `val`, rejecting
    /// `prec == 0` ([`ScalarError::ZeroPrecision`]) and `NaN` / `±Infinity` ([`ScalarError::NonFiniteFloat`]).
    pub fn try_with_prec(prec: usize, val: f64) -> Result<Self, ScalarError> {
        if prec == 0 {
            return Err(ScalarError::ZeroPrecision);
        }
        if !val.is_finite() {
            return Err(ScalarError::NonFiniteFloat);
        }
        let exact = BinaryFloat::try_from(val).map_err(|_| ScalarError::NonFiniteFloat)?;
        Ok(Self(exact.with_precision(prec).value()))
    }

    /// Converts an exact [`BigInt`] to `prec` binary mantissa bits.
    pub fn from_bigint_with_prec(prec: usize, val: &BigInt) -> Result<Self, ScalarError> {
        if prec == 0 {
            return Err(ScalarError::ZeroPrecision);
        }
        let f = BinaryFloat::from_parts(val.0.clone(), 0)
            .with_precision(prec)
            .value();
        Ok(Self(f))
    }

    /// Evaluates an exact [`BigRational`] $p / q$ rounded to `prec` binary mantissa bits.
    pub fn from_rational_with_prec(prec: usize, rat: &BigRational) -> Result<Self, ScalarError> {
        let num = Self::from_bigint_with_prec(prec, &rat.numerator())?;
        let den = Self::from_bigint_with_prec(prec, &rat.denominator())?;
        Ok(num / den)
    }

    /// Active mantissa precision in bits.
    #[inline]
    #[must_use]
    pub fn precision(&self) -> usize {
        self.0.precision()
    }

    /// Re-rounds `self` to `prec` mantissa bits.
    pub fn with_prec(&self, prec: usize) -> Result<Self, ScalarError> {
        if prec == 0 {
            return Err(ScalarError::ZeroPrecision);
        }
        Ok(Self(self.0.clone().with_precision(prec).value()))
    }

    /// Magnitude $|x|$ preserving `self.precision()`.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self(self.0.clone().abs())
    }

    /// Closest IEEE-754 `f64` value.
    #[inline]
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().value()
    }

    /// Rounds to the nearest integer as `f64` (halfway away from zero).
    #[inline]
    #[must_use]
    pub fn round_f64(&self) -> f64 {
        libm::round(self.to_f64())
    }

    /// Borrows the underlying `dashu::float::FBig<HalfAway, 2>`.
    #[inline]
    #[must_use]
    pub const fn as_fbig(&self) -> &BinaryFloat {
        &self.0
    }
}

impl PartialEq for BigFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for BigFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl fmt::Display for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Add for BigFloat {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let p = self.precision().max(rhs.precision());
        Self((self.0 + rhs.0).with_precision(p).value())
    }
}

impl Sub for BigFloat {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let p = self.precision().max(rhs.precision());
        Self((self.0 - rhs.0).with_precision(p).value())
    }
}

impl Mul for BigFloat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let p = self.precision().max(rhs.precision());
        Self((self.0 * rhs.0).with_precision(p).value())
    }
}

impl Div for BigFloat {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let p = self.precision().max(rhs.precision());
        Self((self.0 / rhs.0).with_precision(p).value())
    }
}

impl Neg for BigFloat {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
