// [cite:math_scalars]
//! Midpoint-radius ball enclosure `Ball<T>` (`\[mid - rad, mid + rad\]`).

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::interval::Interval;
use crate::rounding::{add_bounds, mul_bounds, sub_bounds};
use crate::traits::{RealScalar, Scalar};

/// Certified midpoint-radius real ball $\{x \in \mathbb{R} : |x - \text{mid}| \le \text{rad}\}$.
///
/// Propagates floating-point midpoint rounding errors outward into `rad >= 0` using
/// portable IEEE-754 ULP bounds without FPU state mutation.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ball<T> {
    /// Center approximation.
    pub mid: T,
    /// Non-negative outward-rounded error radius.
    pub rad: T,
}

impl<T: RealScalar> Default for Ball<T> {
    #[inline]
    fn default() -> Self {
        Self::exact(T::ZERO)
    }
}

impl<T: RealScalar> Ball<T> {
    /// Constructs a ball `(mid, rad)`, returning `(NaN, NaN)` if `rad < 0` or either is `NaN`.
    #[inline]
    #[must_use]
    pub fn new(mid: T, rad: T) -> Self {
        if mid.is_nan() || rad.is_nan() || rad < T::ZERO {
            Self {
                mid: T::NAN,
                rad: T::NAN,
            }
        } else if !mid.is_finite() {
            Self {
                mid: T::ZERO,
                rad: T::INFINITY,
            }
        } else {
            Self { mid, rad }
        }
    }

    /// Zero-radius exact point ball `(val, 0)`.
    #[inline]
    #[must_use]
    pub const fn exact(val: T) -> Self {
        Self {
            mid: val,
            rad: T::ZERO,
        }
    }

    /// Invalid `NaN` ball sentinel.
    #[inline]
    #[must_use]
    pub const fn nan() -> Self {
        Self {
            mid: T::NAN,
            rad: T::NAN,
        }
    }

    /// Converts an [`Interval`] `\[lo, hi\]` into a guaranteed enclosing [`Ball`].
    #[inline]
    #[must_use]
    pub fn from_interval(iv: Interval<T>) -> Self {
        if iv.is_nan() {
            return Self::nan();
        }
        if !iv.is_finite() {
            return Self {
                mid: T::ZERO,
                rad: T::INFINITY,
            };
        }
        Self::new(iv.midpoint(), iv.radius())
    }

    /// Converts `self` into an outward-rounded [`Interval`] `\[mid - rad, mid + rad\]`.
    #[inline]
    #[must_use]
    pub fn to_interval(self) -> Interval<T> {
        if self.is_nan() {
            return Interval::nan();
        }
        if !self.rad.is_finite() {
            return Interval::entire();
        }
        Interval::new(
            sub_bounds(self.mid, self.rad).0,
            add_bounds(self.mid, self.rad).1,
        )
    }

    /// Returns `true` if `mid` or `rad` is `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.mid.is_nan() || self.rad.is_nan()
    }

    /// Returns `true` if both `mid` and `rad` are finite and non-`NaN`.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        !self.is_nan() && self.mid.is_finite() && self.rad.is_finite()
    }

    /// Rigorously checks whether `x` lies inside `self` via its enclosing interval.
    #[inline]
    #[must_use]
    pub fn contains(self, x: T) -> bool {
        self.to_interval().contains(x)
    }

    /// Number of IEEE-754 ULP steps across `self.to_interval()`.
    #[inline]
    #[must_use]
    pub fn width_ulps(self) -> u64 {
        self.to_interval().width_ulps()
    }
}

impl<T: RealScalar> Neg for Ball<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        if self.is_nan() {
            Self::nan()
        } else {
            Self {
                mid: -self.mid,
                rad: self.rad,
            }
        }
    }
}

impl<T: RealScalar> Add for Ball<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }
        let m = self.mid + rhs.mid;
        if !m.is_finite() {
            return Self::from_interval(self.to_interval() + rhs.to_interval());
        }
        let (s_lo, s_hi) = add_bounds(self.mid, rhs.mid);
        let mid_err = sub_bounds(m, s_lo).1.max_val(sub_bounds(s_hi, m).1);
        let r_sum = add_bounds(self.rad, rhs.rad).1;
        Self::new(m, add_bounds(r_sum, mid_err).1)
    }
}

impl<T: RealScalar> Sub for Ball<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<T: RealScalar> Mul for Ball<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }
        let m = self.mid * rhs.mid;
        if !m.is_finite() {
            return Self::from_interval(self.to_interval() * rhs.to_interval());
        }
        let (p_lo, p_hi) = mul_bounds(self.mid, rhs.mid);
        let mid_err = sub_bounds(m, p_lo).1.max_val(sub_bounds(p_hi, m).1);
        let t1 = mul_bounds(self.mid.abs(), rhs.rad).1;
        let t2 = mul_bounds(rhs.mid.abs(), self.rad).1;
        let t3 = mul_bounds(self.rad, rhs.rad).1;
        let r12 = add_bounds(t1, t2).1;
        let r123 = add_bounds(r12, t3).1;
        Self::new(m, add_bounds(r123, mid_err).1)
    }
}

impl<T: RealScalar> Div for Ball<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::from_interval(self.to_interval() / rhs.to_interval())
    }
}

impl<T: RealScalar> AddAssign for Ball<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: RealScalar> SubAssign for Ball<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: RealScalar> MulAssign for Ball<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: RealScalar> DivAssign for Ball<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: RealScalar> Scalar for Ball<T> {
    const ZERO: Self = Self::exact(T::ZERO);
    const ONE: Self = Self::exact(T::ONE);
    const TWO: Self = Self::exact(T::TWO);
    const HALF: Self = Self::exact(T::HALF);
    const PI: Self = Self {
        mid: T::PI,
        rad: T::EPSILON,
    };

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::from_interval(Interval::from_f64(v))
    }
    #[inline]
    fn log2_e_scale() -> Self {
        Self::from_interval(Interval::log2_e_scale())
    }
    #[inline]
    fn log10_e_scale() -> Self {
        Self::from_interval(Interval::log10_e_scale())
    }
    #[inline]
    fn real_part(self) -> f64 {
        self.mid.to_f64()
    }
    #[inline]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn abs(self) -> Self {
        Self::from_interval(self.to_interval().abs())
    }
    #[inline]
    fn sqr(self) -> Self {
        Self::from_interval(self.to_interval().sqr())
    }
    #[inline]
    fn sqrt(self) -> Self {
        Self::from_interval(self.to_interval().sqrt())
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        Self::from_interval(self.to_interval().powi(n))
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        Self::from_interval(self.to_interval().powf(exp.to_interval()))
    }
    #[inline]
    fn exp(self) -> Self {
        Self::from_interval(self.to_interval().exp())
    }
    #[inline]
    fn ln(self) -> Self {
        Self::from_interval(self.to_interval().ln())
    }
    #[inline]
    fn log2(self) -> Self {
        Self::from_interval(self.to_interval().log2())
    }
    #[inline]
    fn log10(self) -> Self {
        Self::from_interval(self.to_interval().log10())
    }
    #[inline]
    fn sin(self) -> Self {
        Self::from_interval(self.to_interval().sin())
    }
    #[inline]
    fn cos(self) -> Self {
        Self::from_interval(self.to_interval().cos())
    }
    #[inline]
    fn tan(self) -> Self {
        Self::from_interval(self.to_interval().tan())
    }
    #[inline]
    fn asin(self) -> Self {
        Self::from_interval(self.to_interval().asin())
    }
    #[inline]
    fn acos(self) -> Self {
        Self::from_interval(self.to_interval().acos())
    }
    #[inline]
    fn atan(self) -> Self {
        Self::from_interval(self.to_interval().atan())
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::from_interval(self.to_interval().atan2(other.to_interval()))
    }
    #[inline]
    fn sinh(self) -> Self {
        Self::from_interval(self.to_interval().sinh())
    }
    #[inline]
    fn cosh(self) -> Self {
        Self::from_interval(self.to_interval().cosh())
    }
    #[inline]
    fn tanh(self) -> Self {
        Self::from_interval(self.to_interval().tanh())
    }
    #[inline]
    fn asinh(self) -> Self {
        Self::from_interval(self.to_interval().asinh())
    }
    #[inline]
    fn acosh(self) -> Self {
        Self::from_interval(self.to_interval().acosh())
    }
    #[inline]
    fn atanh(self) -> Self {
        Self::from_interval(self.to_interval().atanh())
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::from_interval(self.to_interval().hypot(other.to_interval()))
    }
}
