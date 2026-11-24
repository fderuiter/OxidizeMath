#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use dashu::base::{Abs, SquareRoot};
use dashu::float::FBig;
use dashu::float::round::mode::HalfEven;

type BinaryFBig = FBig<HalfEven, 2>;

/// Arbitrary-precision binary floating-point scalar (`alloc` feature) preserving caller-specified
/// bit precision across arithmetic operations.
#[derive(Clone, Debug, PartialEq)]
pub struct BigFloat {
    inner: BinaryFBig,
    prec_bits: usize,
}

impl BigFloat {
    /// Constructs a `BigFloat` from a finite `f64` retaining `prec` bits of binary precision.
    ///
    /// # Errors
    /// - Returns [`ScalarError::ZeroPrecision`] when `prec == 0`.
    /// - Returns [`ScalarError::NonFiniteFloat`] when `val` is `NaN` or `±Infinity`.
    pub fn try_with_prec(prec: usize, val: f64) -> Result<Self, ScalarError> {
        if prec == 0 {
            return Err(ScalarError::ZeroPrecision);
        }
        if !val.is_finite() {
            return Err(ScalarError::NonFiniteFloat);
        }
        let converted = BinaryFBig::try_from(val).map_err(|_| ScalarError::NonFiniteFloat)?;
        let inner = converted.with_precision(prec).value();
        Ok(Self {
            inner,
            prec_bits: prec,
        })
    }

    /// Constructs an exact integer `val` at `prec` bits of binary precision.
    ///
    /// # Errors
    /// Returns [`ScalarError::ZeroPrecision`] when `prec == 0`.
    pub fn from_i64_with_prec(prec: usize, val: i64) -> Result<Self, ScalarError> {
        if prec == 0 {
            return Err(ScalarError::ZeroPrecision);
        }
        let inner = BinaryFBig::from(val).with_precision(prec).value();
        Ok(Self {
            inner,
            prec_bits: prec,
        })
    }

    /// Returns the active binary significand precision in bits.
    #[inline]
    #[must_use]
    pub const fn precision(&self) -> usize {
        self.prec_bits
    }

    /// Borrows the underlying `dashu::float::FBig`.
    #[inline]
    #[must_use]
    pub const fn as_fbig(&self) -> &BinaryFBig {
        &self.inner
    }

    /// Converts `self` to nearest IEEE-754 `f64`.
    #[inline]
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.inner.to_f64().value()
    }

    /// Returns $|x|$ with the same bit precision.
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .abs()
                .with_precision(self.prec_bits)
                .value(),
            prec_bits: self.prec_bits,
        }
    }

    /// Principal square root $\sqrt{x}$ evaluated at `self.precision()` bits.
    #[inline]
    #[must_use]
    pub fn sqrt(&self) -> Self {
        Self {
            inner: self.inner.sqrt().with_precision(self.prec_bits).value(),
            prec_bits: self.prec_bits,
        }
    }

    /// Checked multiplicative inverse $1 / x$ at `self.precision()` bits.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `self == 0`.
    pub fn checked_inv(&self) -> Result<Self, ScalarError> {
        if self.inner == BinaryFBig::ZERO {
            return Err(ScalarError::NonInvertible);
        }
        let one = BinaryFBig::ONE.with_precision(self.prec_bits).value();
        Ok(Self {
            inner: (one / &self.inner).with_precision(self.prec_bits).value(),
            prec_bits: self.prec_bits,
        })
    }

    fn combine_prec(&self, rhs: &Self) -> (BinaryFBig, BinaryFBig, usize) {
        let p = self.prec_bits.max(rhs.prec_bits);
        let a = self.inner.clone().with_precision(p).value();
        let b = rhs.inner.clone().with_precision(p).value();
        (a, b, p)
    }
}

impl PartialOrd for BigFloat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

impl fmt::Display for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl Neg for BigFloat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            inner: -self.inner,
            prec_bits: self.prec_bits,
        }
    }
}

impl Add for BigFloat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (a, b, p) = self.combine_prec(&rhs);
        Self {
            inner: (a + b).with_precision(p).value(),
            prec_bits: p,
        }
    }
}

impl Sub for BigFloat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let (a, b, p) = self.combine_prec(&rhs);
        Self {
            inner: (a - b).with_precision(p).value(),
            prec_bits: p,
        }
    }
}

impl Mul for BigFloat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (a, b, p) = self.combine_prec(&rhs);
        Self {
            inner: (a * b).with_precision(p).value(),
            prec_bits: p,
        }
    }
}

impl Div for BigFloat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let (a, b, p) = self.combine_prec(&rhs);
        Self {
            inner: (a / b).with_precision(p).value(),
            prec_bits: p,
        }
    }
}
