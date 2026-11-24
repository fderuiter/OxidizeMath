#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

#[path = "big_float.rs"]
mod big_float;

pub use big_float::BigFloat;

use crate::error::ScalarError;
use crate::int256::I256;
use crate::uint256::U256;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};
use core::str::FromStr;
use dashu::base::{Abs, BitTest, Signed, UnsignedAbs};
use dashu::integer::{IBig, UBig};
use dashu::rational::RBig;

/// Arbitrary-precision signed integer backed by `dashu::integer::IBig` (`alloc` feature).
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

    /// Wraps a `dashu::integer::IBig`.
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

    /// Converts a 256-bit unsigned stack integer [`U256`] into `BigInt` without narrowing.
    #[must_use]
    pub fn from_u256(v: U256) -> Self {
        let limbs = v.limbs();
        let lo = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
        let hi = (limbs[2] as u128) | ((limbs[3] as u128) << 64);
        let u = (UBig::from(hi) << 128usize) | UBig::from(lo);
        Self(IBig::from(u))
    }

    /// Converts a 256-bit signed stack integer [`I256`] into `BigInt` without narrowing.
    #[must_use]
    pub fn from_i256(v: I256) -> Self {
        let mag = Self::from_u256(v.unsigned_abs());
        if v.is_negative() { -mag } else { mag }
    }

    /// Parses an integer in `radix` (`2..=36`).
    ///
    /// # Errors
    /// Returns [`ScalarError::InvalidDigit`] when `radix` or `src` is invalid.
    pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, ScalarError> {
        if !(2..=36).contains(&radix) {
            return Err(ScalarError::InvalidDigit);
        }
        IBig::from_str_radix(src, radix)
            .map(Self)
            .map_err(|_| ScalarError::InvalidDigit)
    }

    /// Returns `true` when `self == 0`.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Returns `true` when `self < 0`.
    #[inline]
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.0.is_negative()
    }

    /// Returns the absolute value $|x|$.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self(self.0.clone().abs())
    }

    /// Raises `self` to unsigned power `exp`.
    #[inline]
    #[must_use]
    pub fn pow(&self, exp: usize) -> Self {
        Self(self.0.pow(exp))
    }

    /// Bit length of the magnitude of `self` (`0` for `0`).
    #[inline]
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.0.clone().unsigned_abs().bit_len()
    }

    /// Approximates `self` as IEEE-754 `f64`.
    #[inline]
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().value()
    }

    /// Checked integer division returning [`ScalarError::DivisionByZero`] when `rhs == 0`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] if `rhs.is_zero()`.
    pub fn checked_div(&self, rhs: &Self) -> Result<Self, ScalarError> {
        if rhs.is_zero() {
            Err(ScalarError::DivisionByZero)
        } else {
            Ok(Self(&self.0 / &rhs.0))
        }
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

impl FromStr for BigInt {
    type Err = ScalarError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(s, 10)
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Neg for BigInt {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Add for BigInt {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<'a> Add<&'a BigInt> for &BigInt {
    type Output = BigInt;
    #[inline]
    fn add(self, rhs: &'a BigInt) -> Self::Output {
        BigInt(&self.0 + &rhs.0)
    }
}

impl AddAssign for BigInt {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for BigInt {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a BigInt> for &BigInt {
    type Output = BigInt;
    #[inline]
    fn sub(self, rhs: &'a BigInt) -> Self::Output {
        BigInt(&self.0 - &rhs.0)
    }
}

impl SubAssign for BigInt {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul for BigInt {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<'a> Mul<&'a BigInt> for &BigInt {
    type Output = BigInt;
    #[inline]
    fn mul(self, rhs: &'a BigInt) -> Self::Output {
        BigInt(&self.0 * &rhs.0)
    }
}

impl MulAssign for BigInt {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Div for BigInt {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl Rem for BigInt {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

/// Arbitrary-precision reduced rational number $p / q$ ($q > 0, \gcd(|p|, q) = 1$)
/// backed by `dashu::rational::RBig` (`alloc` feature).
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BigRational(RBig);

impl BigRational {
    /// Additive identity $0 / 1$.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self(RBig::ZERO)
    }

    /// Multiplicative identity $1 / 1$.
    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self(RBig::ONE)
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

    /// Constructs a canonical reduced rational `num / den`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `den == 0`.
    pub fn new(num: BigInt, den: BigInt) -> Result<Self, ScalarError> {
        if den.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        let (signed_num, unsigned_den) = if den.is_negative() {
            (-num.0, den.0.unsigned_abs())
        } else {
            (num.0, den.0.unsigned_abs())
        };
        Ok(Self(RBig::from_parts(signed_num, unsigned_den)))
    }

    /// Constructs a reduced rational from `i64` numerator and denominator.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `den == 0`.
    pub fn from_i64_ratio(num: i64, den: i64) -> Result<Self, ScalarError> {
        Self::new(BigInt::from(num), BigInt::from(den))
    }

    /// Exact reduced signed integer numerator $p \in \mathbb{Z}$ without `f64` narrowing.
    #[inline]
    #[must_use]
    pub fn numerator(&self) -> BigInt {
        BigInt(self.0.numerator().clone())
    }

    /// Exact reduced positive integer denominator $q \in \mathbb{Z}_{>0}$ without `f64` narrowing.
    #[inline]
    #[must_use]
    pub fn denominator(&self) -> BigInt {
        BigInt(IBig::from(self.0.denominator().clone()))
    }

    /// Returns `true` when `self == 0`.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Returns $|p / q|$.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self(self.0.clone().abs())
    }

    /// Checked multiplicative inverse $q / p$.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `self == 0`.
    pub fn checked_inv(&self) -> Result<Self, ScalarError> {
        if self.is_zero() {
            Err(ScalarError::NonInvertible)
        } else {
            Ok(Self(RBig::ONE / &self.0))
        }
    }

    /// Converts to nearest IEEE-754 `f64`.
    #[inline]
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().value()
    }
}

impl From<i64> for BigRational {
    #[inline]
    fn from(v: i64) -> Self {
        Self(RBig::from(v))
    }
}

impl From<BigInt> for BigRational {
    #[inline]
    fn from(v: BigInt) -> Self {
        Self(RBig::from(v.0))
    }
}

impl fmt::Display for BigRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Neg for BigRational {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Add for BigRational {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for BigRational {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for BigRational {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for BigRational {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul for BigRational {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign for BigRational {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Div for BigRational {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl DivAssign for BigRational {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}
