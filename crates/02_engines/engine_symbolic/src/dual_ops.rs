//! `DualScalar` implementations for `f64` and `Complex<f64>` plus arithmetic operators on [`Dual`].
// [cite:engine_symbolic]

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};
use num_traits::Float;

use crate::dual::{Dual, DualScalar, pow_f64_dual};

#[path = "dual_complex.rs"]
mod dual_complex;

impl DualScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NAN: Self = f64::NAN;

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn from_f64(val: f64) -> Self {
        val
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn real_part(self) -> f64 {
        self
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn is_nan_val(self) -> bool {
        self.is_nan()
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn is_zero_val(self) -> bool {
        self == 0.0
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sin_with_deriv(self) -> (Self, Self) {
        Float::sin_cos(self)
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn cos_with_deriv(self) -> (Self, Self) {
        let (s, c) = Float::sin_cos(self);
        (c, -s)
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn tan_with_deriv(self) -> (Self, Self) {
        let t = Float::tan(self);
        let c = Float::cos(self);
        (t, 1.0 / (c * c))
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn exp_with_deriv(self) -> (Self, Self) {
        let e = Float::exp(self);
        (e, e)
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn ln_with_deriv(self) -> (Self, Self) {
        (Float::ln(self), 1.0 / self)
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn log10_with_deriv(self) -> (Self, Self) {
        (Float::log10(self), 1.0 / (self * core::f64::consts::LN_10))
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sqrt_with_deriv(self) -> (Self, Self) {
        let s = Float::sqrt(self);
        let ds = if s == 0.0 { 0.0 } else { 0.5 / s };
        (s, ds)
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sinh_with_deriv(self) -> (Self, Self) {
        (Float::sinh(self), Float::cosh(self))
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn cosh_with_deriv(self) -> (Self, Self) {
        (Float::cosh(self), Float::sinh(self))
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn tanh_with_deriv(self) -> (Self, Self) {
        let t = Float::tanh(self);
        (t, 1.0 - t * t)
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn asin_with_deriv(self) -> (Self, Self) {
        (Float::asin(self), 1.0 / Float::sqrt(1.0 - self * self))
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn acos_with_deriv(self) -> (Self, Self) {
        (Float::acos(self), -1.0 / Float::sqrt(1.0 - self * self))
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn atan_with_deriv(self) -> (Self, Self) {
        (Float::atan(self), 1.0 / (1.0 + self * self))
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn pow_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self> {
        pow_f64_dual(a, b)
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn abs_dual(x: Dual<Self>) -> Dual<Self> {
        if x.re.is_nan() {
            Dual::new(f64::NAN, f64::NAN)
        } else if x.re > 0.0 {
            x
        } else if x.re < 0.0 {
            -x
        } else {
            Dual::new(0.0, 0.0)
        }
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn step_dual(x: Dual<Self>) -> Dual<Self> {
        if x.re.is_nan() {
            Dual::new(f64::NAN, f64::NAN)
        } else {
            Dual::new(if x.re >= 0.0 { 1.0 } else { 0.0 }, 0.0)
        }
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn rect_dual(x: Dual<Self>) -> Dual<Self> {
        if x.re.is_nan() {
            Dual::new(f64::NAN, f64::NAN)
        } else {
            Dual::new(if Float::abs(x.re) <= 0.5 { 1.0 } else { 0.0 }, 0.0)
        }
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn min_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self> {
        if a.re.is_nan() || b.re.is_nan() {
            Dual::new(f64::NAN, f64::NAN)
        } else if a.re < b.re {
            a
        } else if b.re < a.re {
            b
        } else {
            Dual::new(a.re, 0.5 * (a.eps + b.eps))
        }
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn max_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self> {
        if a.re.is_nan() || b.re.is_nan() {
            Dual::new(f64::NAN, f64::NAN)
        } else if a.re > b.re {
            a
        } else if b.re > a.re {
            b
        } else {
            Dual::new(a.re, 0.5 * (a.eps + b.eps))
        }
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn clamp_dual(x: Dual<Self>, lo: Dual<Self>, hi: Dual<Self>) -> Dual<Self> {
        if x.re.is_nan() || lo.re.is_nan() || hi.re.is_nan() || lo.re > hi.re {
            Dual::new(f64::NAN, f64::NAN)
        } else {
            Self::min_dual(Self::max_dual(x, lo), hi)
        }
    }

    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn atan2_dual(y: Dual<Self>, x: Dual<Self>) -> Dual<Self> {
        let r2 = x.re * x.re + y.re * y.re;
        let val = Float::atan2(y.re, x.re);
        let eps = if r2 == 0.0 {
            0.0
        } else {
            (x.re * y.eps - y.re * x.eps) / r2
        };
        Dual::new(val, eps)
    }
}

impl<T: DualScalar> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.eps)
    }
}

impl<T: DualScalar> Add for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.eps + rhs.eps)
    }
}

impl<T: DualScalar> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.eps - rhs.eps)
    }
}

impl<T: DualScalar> Mul for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.re * rhs.re, self.re * rhs.eps + self.eps * rhs.re)
    }
}

impl<T: DualScalar> Div for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn div(self, rhs: Self) -> Self {
        let inv = T::ONE / rhs.re;
        let re = self.re * inv;
        let eps = (self.eps - re * rhs.eps) * inv;
        Self::new(re, eps)
    }
}

impl Rem for Dual<f64> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn rem(self, rhs: Self) -> Self {
        let q = Float::trunc(self.re / rhs.re);
        Self::new(self.re % rhs.re, self.eps - q * rhs.eps)
    }
}

impl<T: DualScalar> Add<T> for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn add(self, rhs: T) -> Self {
        Self::new(self.re + rhs, self.eps)
    }
}

impl<T: DualScalar> Sub<T> for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sub(self, rhs: T) -> Self {
        Self::new(self.re - rhs, self.eps)
    }
}

impl<T: DualScalar> Mul<T> for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn mul(self, rhs: T) -> Self {
        Self::new(self.re * rhs, self.eps * rhs)
    }
}

impl<T: DualScalar> Div<T> for Dual<T> {
    type Output = Self;
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn div(self, rhs: T) -> Self {
        let inv = T::ONE / rhs;
        Self::new(self.re * inv, self.eps * inv)
    }
}

impl<T: DualScalar> AddAssign for Dual<T> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: DualScalar> SubAssign for Dual<T> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: DualScalar> MulAssign for Dual<T> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: DualScalar> DivAssign for Dual<T> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: DualScalar> MulAssign<f64> for Dual<T> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn mul_assign(&mut self, rhs: f64) {
        let s = T::from_f64(rhs);
        *self = Self::new(self.re * s, self.eps * s);
    }
}
