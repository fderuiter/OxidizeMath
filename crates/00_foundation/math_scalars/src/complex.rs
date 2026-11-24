#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use crate::traits::{FieldScalar, RealScalar};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Single-precision complex scalar `Complex<f32>`.
pub type Complex32 = Complex<f32>;
/// Double-precision complex scalar `Complex<f64>`.
pub type Complex64 = Complex<f64>;

/// Flat `#[repr(C)]` two-component Cayley-Dickson complex number $z = \text{re} + \text{im}\,i$.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Complex<T> {
    /// Real coordinate $\operatorname{Re}(z)$.
    pub re: T,
    /// Imaginary coordinate $\operatorname{Im}(z)$.
    pub im: T,
}

impl<T> Complex<T> {
    /// Constructs $z = \text{re} + \text{im}\,i$.
    #[inline]
    #[must_use]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: RealScalar> Complex<T> {
    /// Additive identity $0 + 0i$.
    pub const ZERO: Self = Self {
        re: T::ZERO,
        im: T::ZERO,
    };
    /// Multiplicative identity $1 + 0i$.
    pub const ONE: Self = Self {
        re: T::ONE,
        im: T::ZERO,
    };
    /// Imaginary unit $0 + 1i$.
    pub const I: Self = Self {
        re: T::ZERO,
        im: T::ONE,
    };

    /// Constructs $r e^{i\theta} = r(\cos\theta + i\sin\theta)$ from polar coordinates.
    #[inline]
    #[must_use]
    pub fn from_polar(r: T, theta: T) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            re: r * c,
            im: r * s,
        }
    }

    /// Complex conjugate $\bar{z} = \text{re} - \text{im}\,i$.
    #[inline]
    #[must_use]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared Euclidean norm $|z|^2 = \text{re}^2 + \text{im}^2$.
    #[inline]
    #[must_use]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// Overflow-safe Euclidean modulus $|z| = \operatorname{hypot}(\text{re}, \text{im})$.
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        self.re.hypot(self.im)
    }

    /// Principal argument $\arg(z) = \operatorname{atan2}(\text{im}, \text{re}) \in [-\pi, \pi]$.
    #[inline]
    #[must_use]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// Scales both components by real factor `s`.
    #[inline]
    #[must_use]
    pub fn scale(self, s: T) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Overflow-resilient Smith multiplicative inverse $1 / z$.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `self == 0` or has non-finite components.
    pub fn checked_inv(self) -> Result<Self, ScalarError> {
        if !self.re.is_finite() || !self.im.is_finite() {
            return Err(ScalarError::NonInvertible);
        }
        let ar = self.re.abs();
        let ai = self.im.abs();
        if ar == T::ZERO && ai == T::ZERO {
            return Err(ScalarError::NonInvertible);
        }
        let out = if ar >= ai {
            let ratio = self.im / self.re;
            let denom = self.re + self.im * ratio;
            Self {
                re: T::ONE / denom,
                im: -ratio / denom,
            }
        } else {
            let ratio = self.re / self.im;
            let denom = self.re * ratio + self.im;
            Self {
                re: ratio / denom,
                im: -T::ONE / denom,
            }
        };
        if out.re.is_finite() && out.im.is_finite() {
            Ok(out)
        } else {
            Err(ScalarError::NonInvertible)
        }
    }

    /// Overflow-resilient Smith division `self / rhs`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `rhs == 0` or produces non-finite output.
    pub fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        let inv = rhs.checked_inv().map_err(|_| ScalarError::DivisionByZero)?;
        Ok(self * inv)
    }
}

impl<T: RealScalar> FieldScalar for Complex<T> {
    type Real = T;
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self {
            re: T::from_f64(v),
            im: T::ZERO,
        }
    }
    #[inline]
    fn from_real(r: Self::Real) -> Self {
        Self { re: r, im: T::ZERO }
    }
    #[inline]
    fn re_part(self) -> Self::Real {
        self.re
    }
    #[inline]
    fn conj(self) -> Self {
        Self::conj(self)
    }
    #[inline]
    fn norm_sqr(self) -> Self::Real {
        Self::norm_sqr(self)
    }
    #[inline]
    fn abs(self) -> Self::Real {
        Self::norm(self)
    }
    #[inline]
    fn scale_real(self, r: Self::Real) -> Self {
        Self::scale(self, r)
    }
    #[inline]
    fn try_inv(self) -> Result<Self, ScalarError> {
        self.checked_inv()
    }
    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: Add<Output = T>> Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: AddAssign> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl<T: Sub<Output = T>> Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: SubAssign> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: RealScalar> Div for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let ar = rhs.re.abs();
        let ai = rhs.im.abs();
        if ar >= ai {
            let ratio = rhs.im / rhs.re;
            let denom = rhs.re + rhs.im * ratio;
            Self {
                re: (self.re + self.im * ratio) / denom,
                im: (self.im - self.re * ratio) / denom,
            }
        } else {
            let ratio = rhs.re / rhs.im;
            let denom = rhs.re * ratio + rhs.im;
            Self {
                re: (self.re * ratio + self.im) / denom,
                im: (self.im * ratio - self.re) / denom,
            }
        }
    }
}

impl<T: RealScalar> DivAssign for Complex<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: fmt::Display + PartialOrd + Default + Neg<Output = T> + Copy> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im < T::default() {
            write!(f, "{} - {}i", self.re, -self.im)
        } else {
            write!(f, "{} + {}i", self.re, self.im)
        }
    }
}
