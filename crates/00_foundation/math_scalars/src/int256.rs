#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use crate::uint256::U256;
use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Rem, Sub, SubAssign};

/// Stack-allocated 256-bit two's-complement signed integer backed by [`U256`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct I256(U256);

const SIGN_LIMB_MASK: u64 = 1u64 << 63;
const MIN_MAGNITUDE: U256 = U256([0, 0, 0, SIGN_LIMB_MASK]);
const MAX_MAGNITUDE: U256 = U256([u64::MAX, u64::MAX, u64::MAX, SIGN_LIMB_MASK - 1]);

impl I256 {
    /// Additive identity $0$.
    pub const ZERO: Self = Self(U256::ZERO);
    /// Multiplicative identity $1$.
    pub const ONE: Self = Self(U256::ONE);
    /// Constant $-1$.
    pub const MINUS_ONE: Self = Self(U256::MAX);
    /// Minimum representable signed 256-bit integer $-2^{255}$.
    pub const MIN: Self = Self(MIN_MAGNITUDE);
    /// Maximum representable signed 256-bit integer $2^{255} - 1$.
    pub const MAX: Self = Self(MAX_MAGNITUDE);

    /// Constructs an `I256` from raw two's-complement little-endian limbs.
    #[inline]
    #[must_use]
    pub const fn from_raw_u256(raw: U256) -> Self {
        Self(raw)
    }

    /// Returns the underlying two's-complement [`U256`] bit pattern.
    #[inline]
    #[must_use]
    pub const fn to_raw_u256(self) -> U256 {
        self.0
    }

    /// Sign-extends an `i64` into `I256`.
    #[inline]
    #[must_use]
    pub const fn from_i64(v: i64) -> Self {
        let fill = if v < 0 { u64::MAX } else { 0 };
        Self(U256([v as u64, fill, fill, fill]))
    }

    /// Sign-extends an `i128` into `I256`.
    #[inline]
    #[must_use]
    pub const fn from_i128(v: i128) -> Self {
        let fill = if v < 0 { u64::MAX } else { 0 };
        let u = v as u128;
        Self(U256([u as u64, (u >> 64) as u64, fill, fill]))
    }

    /// Converts a non-negative `U256` ($\le 2^{255}-1$) into `I256`.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] if `v > I256::MAX`.
    pub const fn try_from_u256(v: U256) -> Result<Self, ScalarError> {
        if v.cmp_const(MAX_MAGNITUDE) > 0 {
            Err(ScalarError::ArithmeticOverflow)
        } else {
            Ok(Self(v))
        }
    }

    /// Converts `self` to `i64` if it fits in `i64::MIN..=i64::MAX`.
    #[must_use]
    pub const fn to_i64(self) -> Option<i64> {
        let limbs = self.0.limbs();
        let neg = (limbs[0] & SIGN_LIMB_MASK) != 0;
        let expected = if neg { u64::MAX } else { 0 };
        if limbs[1] == expected && limbs[2] == expected && limbs[3] == expected {
            Some(limbs[0] as i64)
        } else {
            None
        }
    }

    /// Returns `true` when `self == 0`.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0.is_zero()
    }

    /// Returns `true` when `self < 0` (most-significant sign bit set).
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        (self.0.limbs()[3] & SIGN_LIMB_MASK) != 0
    }

    /// Exact unsigned magnitude $|x|$ as [`U256`] (exact even for `I256::MIN` = $2^{255}$).
    #[inline]
    #[must_use]
    pub const fn unsigned_abs(self) -> U256 {
        if self.is_negative() {
            self.0.wrapping_neg()
        } else {
            self.0
        }
    }

    /// Wrapping two's-complement negation.
    #[inline]
    #[must_use]
    pub const fn wrapping_neg(self) -> Self {
        Self(self.0.wrapping_neg())
    }

    /// Checked negation returning [`ScalarError::ArithmeticOverflow`] for `I256::MIN`.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] if `self == I256::MIN`.
    pub const fn checked_neg(self) -> Result<Self, ScalarError> {
        if self.0.cmp_const(MIN_MAGNITUDE) == 0 {
            Err(ScalarError::ArithmeticOverflow)
        } else {
            Ok(self.wrapping_neg())
        }
    }

    /// Checked absolute value returning [`ScalarError::ArithmeticOverflow`] for `I256::MIN`.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] if `self == I256::MIN`.
    pub const fn checked_abs(self) -> Result<Self, ScalarError> {
        if self.is_negative() {
            self.checked_neg()
        } else {
            Ok(self)
        }
    }

    /// Checked signed addition.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] on signed 256-bit overflow.
    pub const fn checked_add(self, rhs: Self) -> Result<Self, ScalarError> {
        let sum = Self(self.0.wrapping_add(rhs.0));
        let same_sign = self.is_negative() == rhs.is_negative();
        if same_sign && sum.is_negative() != self.is_negative() {
            Err(ScalarError::ArithmeticOverflow)
        } else {
            Ok(sum)
        }
    }

    /// Checked signed subtraction.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] on signed 256-bit overflow.
    pub const fn checked_sub(self, rhs: Self) -> Result<Self, ScalarError> {
        let diff = Self(self.0.wrapping_sub(rhs.0));
        let diff_sign = self.is_negative() != rhs.is_negative();
        if diff_sign && diff.is_negative() != self.is_negative() {
            Err(ScalarError::ArithmeticOverflow)
        } else {
            Ok(diff)
        }
    }

    /// Checked signed multiplication.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] on signed 256-bit overflow.
    pub const fn checked_mul(self, rhs: Self) -> Result<Self, ScalarError> {
        if self.is_zero() || rhs.is_zero() {
            return Ok(Self::ZERO);
        }
        let neg = self.is_negative() ^ rhs.is_negative();
        let (mag, over) = self.unsigned_abs().overflowing_mul(rhs.unsigned_abs());
        if over {
            return Err(ScalarError::ArithmeticOverflow);
        }
        Self::from_sign_and_mag(neg, mag)
    }

    /// Checked truncated division (`self / rhs`) and remainder (`self % rhs`).
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] if `rhs == 0` or [`ScalarError::ArithmeticOverflow`] for `MIN / -1`.
    pub const fn checked_div_rem(self, rhs: Self) -> Result<(Self, Self), ScalarError> {
        if rhs.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        let (q_mag, r_mag) = match self.unsigned_abs().checked_div_rem(rhs.unsigned_abs()) {
            Ok(pair) => pair,
            Err(e) => return Err(e),
        };
        let q_neg = !q_mag.is_zero() && (self.is_negative() ^ rhs.is_negative());
        let r_neg = !r_mag.is_zero() && self.is_negative();
        let q = match Self::from_sign_and_mag(q_neg, q_mag) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let r = match Self::from_sign_and_mag(r_neg, r_mag) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        Ok((q, r))
    }

    /// Checked truncated integer quotient `self / rhs`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] or [`ScalarError::ArithmeticOverflow`].
    #[inline]
    pub const fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        match self.checked_div_rem(rhs) {
            Ok((q, _)) => Ok(q),
            Err(e) => Err(e),
        }
    }

    /// Checked truncated remainder `self % rhs`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] or [`ScalarError::ArithmeticOverflow`].
    #[inline]
    pub const fn checked_rem(self, rhs: Self) -> Result<Self, ScalarError> {
        match self.checked_div_rem(rhs) {
            Ok((_, r)) => Ok(r),
            Err(e) => Err(e),
        }
    }

    /// Checked Euclidean quotient $q$ and non-negative remainder $r \in [0, |b|)$ satisfying $a = q b + r$.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] if `rhs == 0` or [`ScalarError::ArithmeticOverflow`] for `MIN.checked_div_euclid(-1)`.
    pub const fn checked_div_rem_euclid(self, rhs: Self) -> Result<(Self, Self), ScalarError> {
        if rhs.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        let b_abs = rhs.unsigned_abs();
        let (q_mag, r_mag) = match self.unsigned_abs().checked_div_rem(b_abs) {
            Ok(pair) => pair,
            Err(e) => return Err(e),
        };
        if !self.is_negative() || r_mag.is_zero() {
            let q_neg = !q_mag.is_zero() && (self.is_negative() ^ rhs.is_negative());
            let q = match Self::from_sign_and_mag(q_neg, q_mag) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let r = match Self::from_sign_and_mag(false, r_mag) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            return Ok((q, r));
        }
        let q_adj = q_mag.wrapping_add(U256::ONE);
        let r_adj = b_abs.wrapping_sub(r_mag);
        let q_neg = !rhs.is_negative();
        let q = match Self::from_sign_and_mag(q_neg, q_adj) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let r = match Self::from_sign_and_mag(false, r_adj) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        Ok((q, r))
    }

    /// Checked Euclidean quotient `self.div_euclid(rhs)`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] or [`ScalarError::ArithmeticOverflow`].
    #[inline]
    pub const fn checked_div_euclid(self, rhs: Self) -> Result<Self, ScalarError> {
        match self.checked_div_rem_euclid(rhs) {
            Ok((q, _)) => Ok(q),
            Err(e) => Err(e),
        }
    }

    /// Checked Euclidean remainder `self.rem_euclid(rhs)` in `[0, |rhs|)`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] if `rhs == 0`.
    #[inline]
    pub const fn checked_rem_euclid(self, rhs: Self) -> Result<Self, ScalarError> {
        if rhs.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        let b_abs = rhs.unsigned_abs();
        let (_, r_mag) = match self.unsigned_abs().checked_div_rem(b_abs) {
            Ok(pair) => pair,
            Err(e) => return Err(e),
        };
        if !self.is_negative() || r_mag.is_zero() {
            Self::from_sign_and_mag(false, r_mag)
        } else {
            Self::from_sign_and_mag(false, b_abs.wrapping_sub(r_mag))
        }
    }

    /// Euclidean quotient panicking only on division by zero or `MIN / -1`.
    #[inline]
    #[must_use]
    pub fn div_euclid(self, rhs: Self) -> Self {
        self.checked_div_euclid(rhs)
            .expect("I256::div_euclid failed")
    }

    /// Non-negative Euclidean remainder panicking only on division by zero.
    #[inline]
    #[must_use]
    pub fn rem_euclid(self, rhs: Self) -> Self {
        self.checked_rem_euclid(rhs)
            .expect("I256::rem_euclid division by zero")
    }

    const fn from_sign_and_mag(neg: bool, mag: U256) -> Result<Self, ScalarError> {
        if neg {
            if mag.cmp_const(MIN_MAGNITUDE) > 0 {
                Err(ScalarError::ArithmeticOverflow)
            } else {
                Ok(Self(mag.wrapping_neg()))
            }
        } else if mag.cmp_const(MAX_MAGNITUDE) > 0 {
            Err(ScalarError::ArithmeticOverflow)
        } else {
            Ok(Self(mag))
        }
    }
}

impl PartialOrd for I256 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I256 {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_negative(), other.is_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => self.0.cmp(&other.0),
        }
    }
}

impl From<i64> for I256 {
    #[inline]
    fn from(v: i64) -> Self {
        Self::from_i64(v)
    }
}

impl From<i128> for I256 {
    #[inline]
    fn from(v: i128) -> Self {
        Self::from_i128(v)
    }
}

impl Neg for I256 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        self.checked_neg().expect("I256 negation overflow")
    }
}

impl Add for I256 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs).expect("I256 addition overflow")
    }
}

impl AddAssign for I256 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for I256 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(rhs).expect("I256 subtraction overflow")
    }
}

impl SubAssign for I256 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for I256 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(rhs).expect("I256 multiplication overflow")
    }
}

impl Div for I256 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("I256 division error")
    }
}

impl Rem for I256 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self.checked_rem(rhs).expect("I256 remainder error")
    }
}

impl fmt::Display for I256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_negative() {
            f.write_str("-")?;
        }
        write!(f, "{}", self.unsigned_abs())
    }
}
