#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
//! Flat-coordinate `#[repr(C)]` complex field algebra `Complex<T>` ($a + b i$, $i^2 = -1$)
//! implementing [`FieldScalar`] with overflow-safe Smith division and Euclidean hypotenuse norms.

use crate::error::ScalarError;
use crate::traits::{FieldScalar, RealScalar};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Flat-coordinate `#[repr(C)]` complex number $z = \text{re} + \text{im}\,i$.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Returns `[re, im]` in contiguous coordinate order.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [T; 2]
    where
        T: Copy,
    {
        [self.re, self.im]
    }
}

impl<T: RealScalar> Complex<T> {
    /// Imaginary unit $i = 0 + 1i$.
    pub const I: Self = Self {
        re: T::ZERO,
        im: T::ONE,
    };

    /// Embeds a real value `re` as $z = \text{re} + 0i$.
    #[inline]
    #[must_use]
    pub const fn from_real(re: T) -> Self {
        Self { re, im: T::ZERO }
    }

    /// Constructs $r (\cos \theta + i \sin \theta)$ from polar coordinates `(r, theta)`.
    #[inline]
    #[must_use]
    pub fn from_polar(r: T, theta: T) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex conjugate $\overline{z} = \text{re} - \text{im}\,i$.
    #[inline]
    #[must_use]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared modulus $|z|^2 = \text{re}^2 + \text{im}^2$.
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

    /// Principal argument $\arg(z) \in [-\pi, \pi]$.
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

    /// Overflow-safe multiplicative inverse $z^{-1}$ using Smith's scaling method.
    #[must_use]
    pub fn checked_inv(self) -> Option<Self> {
        self.checked_div_smith(Self::ONE, self)
    }

    /// Fallible multiplicative inverse returning [`ScalarError::DivisionByZero`] when $z = 0$ or non-finite.
    pub fn try_inverse(self) -> Result<Self, ScalarError> {
        self.checked_inv().ok_or(ScalarError::DivisionByZero)
    }

    /// Overflow-safe complex division `self / rhs` via Smith's algorithm, returning `None` on zero/non-finite divisor.
    #[must_use]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        self.checked_div_smith(self, rhs)
    }

    fn checked_div_smith(self, num: Self, den: Self) -> Option<Self> {
        let _ = self;
        if !num.is_finite() || !den.is_finite() {
            return None;
        }
        let abs_re = FieldScalar::abs(den.re);
        let abs_im = FieldScalar::abs(den.im);
        if abs_re == T::ZERO && abs_im == T::ZERO {
            return None;
        }
        let out = if abs_re >= abs_im {
            let r = den.im / den.re;
            let d = den.re + den.im * r;
            if d == T::ZERO {
                return None;
            }
            Self {
                re: (num.re + num.im * r) / d,
                im: (num.im - num.re * r) / d,
            }
        } else {
            let r = den.re / den.im;
            let d = den.im + den.re * r;
            if d == T::ZERO {
                return None;
            }
            Self {
                re: (num.re * r + num.im) / d,
                im: (num.im * r - num.re) / d,
            }
        };
        out.is_finite().then_some(out)
    }

    /// Complex exponential $e^{x + iy} = e^x (\cos y + i \sin y)$.
    #[must_use]
    pub fn exp(self) -> Self {
        Self::from_polar(self.re.exp(), self.im)
    }

    /// Principal natural logarithm $\ln(z) = \ln|z| + i \arg(z)$.
    #[must_use]
    pub fn ln(self) -> Self {
        Self {
            re: self.norm().ln(),
            im: self.arg(),
        }
    }

    /// Principal square root $\sqrt{z}$ with $\operatorname{Re}(\sqrt{z}) \ge 0$.
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.re == T::ZERO && self.im == T::ZERO {
            return Self::ZERO;
        }
        let m = self.norm();
        if self.re >= T::ZERO {
            let u = (T::HALF * (m + self.re)).sqrt();
            let v = if u == T::ZERO {
                T::ZERO
            } else {
                self.im / (T::TWO * u)
            };
            Self { re: u, im: v }
        } else {
            let v_mag = (T::HALF * (m - self.re)).sqrt();
            let v = v_mag.copysign(self.im);
            let u = if v_mag == T::ZERO {
                T::ZERO
            } else {
                FieldScalar::abs(self.im) / (T::TWO * v_mag)
            };
            Self { re: u, im: v }
        }
    }
}

impl<T: RealScalar> FieldScalar for Complex<T> {
    type Real = T;

    const ZERO: Self = Self {
        re: T::ZERO,
        im: T::ZERO,
    };
    const ONE: Self = Self {
        re: T::ONE,
        im: T::ZERO,
    };

    #[inline]
    fn from_real(val: T) -> Self {
        Self {
            re: val,
            im: T::ZERO,
        }
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        Self {
            re: T::from_f64(val),
            im: T::ZERO,
        }
    }

    #[inline]
    fn from_usize(val: usize) -> Self {
        Self {
            re: T::from_usize(val),
            im: T::ZERO,
        }
    }

    #[inline]
    fn conj(self) -> Self {
        Complex::conj(self)
    }

    #[inline]
    fn abs(self) -> T {
        self.norm()
    }

    #[inline]
    fn norm_sqr(self) -> T {
        Complex::norm_sqr(self)
    }

    #[inline]
    fn checked_inv(self) -> Option<Self> {
        Complex::checked_inv(self)
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

impl<T: fmt::Display + PartialOrd + Default + Copy + Neg<Output = T>> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im < T::default() {
            write!(f, "{} - {}i", self.re, -self.im)
        } else {
            write!(f, "{} + {}i", self.re, self.im)
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
        let abs_re = FieldScalar::abs(rhs.re);
        let abs_im = FieldScalar::abs(rhs.im);
        if abs_re >= abs_im {
            let r = rhs.im / rhs.re;
            let d = rhs.re + rhs.im * r;
            Self {
                re: (self.re + self.im * r) / d,
                im: (self.im - self.re * r) / d,
            }
        } else {
            let r = rhs.re / rhs.im;
            let d = rhs.im + rhs.re * r;
            Self {
                re: (self.re * r + self.im) / d,
                im: (self.im * r - self.re) / d,
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

/// Convenience alias for `Complex<f32>`.
pub type Complex32 = Complex<f32>;
/// Convenience alias for `Complex<f64>`.
pub type Complex64 = Complex<f64>;
