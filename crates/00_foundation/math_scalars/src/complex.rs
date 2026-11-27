// [cite:math_scalars]
//! `Complex<T>` implementations of [`RefOps`], [`Scalar`], [`Ring`], [`Field`], and [`ComplexField`].

use crate::ieee::powi_field;
use crate::traits::{ComplexField, Field, RealField, Ring};
pub use num_complex::{Complex, Complex32, Complex64};

/// Overflow-safe complex division $(a + bi) / (c + di)$ via Smith's rescaled ratio algorithm.
#[inline]
pub(crate) fn smith_div<T: RealField>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    if rhs.re.abs() >= rhs.im.abs() {
        let ratio = rhs.im / rhs.re;
        let denom = rhs.re + rhs.im * ratio;
        Complex::new(
            (lhs.re + lhs.im * ratio) / denom,
            (lhs.im - lhs.re * ratio) / denom,
        )
    } else {
        let ratio = rhs.re / rhs.im;
        let denom = rhs.im + rhs.re * ratio;
        Complex::new(
            (lhs.re * ratio + lhs.im) / denom,
            (lhs.im * ratio - lhs.re) / denom,
        )
    }
}

#[inline]
fn complex_sqrt<T: RealField>(z: Complex<T>) -> Complex<T> {
    if z.im == T::ZERO {
        return if z.re >= T::ZERO {
            Complex::new(z.re.sqrt(), T::ZERO)
        } else {
            Complex::new(T::ZERO, (-z.re).sqrt().copysign(z.im))
        };
    }
    let r = z.re.hypot(z.im);
    if z.re >= T::ZERO {
        let t = (T::HALF * (r + z.re)).sqrt();
        Complex::new(t, T::HALF * z.im / t)
    } else {
        let t = (T::HALF * (r - z.re)).sqrt();
        Complex::new(T::HALF * z.im.abs() / t, t.copysign(z.im))
    }
}

#[inline]
fn complex_ln<T: RealField>(z: Complex<T>) -> Complex<T> {
    Complex::new(z.re.hypot(z.im).ln(), z.im.atan2(z.re))
}

#[inline]
fn complex_exp<T: RealField>(z: Complex<T>) -> Complex<T> {
    if z.im == T::ZERO {
        return Complex::new(z.re.exp(), T::ZERO);
    }
    let r = z.re.exp();
    let (s, c) = z.im.sin_cos();
    Complex::new(r * c, r * s)
}

impl<T: RealField + num_traits::NumAssign> Ring for Complex<T> {
    #[inline]
    fn zero() -> Self {
        Self::new(T::ZERO, T::ZERO)
    }
    #[inline]
    fn one() -> Self {
        Self::new(T::ONE, T::ZERO)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.re == T::ZERO && self.im == T::ZERO
    }
}

impl<T: RealField + num_traits::NumAssign> Field for Complex<T> {
    #[inline]
    fn ref_div(&self, rhs: &Self) -> Self {
        smith_div(*self, *rhs)
    }
    #[inline]
    fn try_inv(&self) -> Option<Self> {
        (!self.is_zero()).then(|| smith_div(Self::one(), *self))
    }
}

impl<T: RealField + num_traits::NumAssign> ComplexField for Complex<T> {
    type Real = T;

    #[inline]
    fn from_real(re: T) -> Self {
        Self::new(re, T::ZERO)
    }
    #[inline]
    fn imag_unit() -> Self {
        Self::new(T::ZERO, T::ONE)
    }
    #[inline]
    fn nan() -> Self {
        Self::new(T::NAN, T::NAN)
    }
    #[inline]
    fn real(&self) -> T {
        self.re
    }
    #[inline]
    fn imag(&self) -> T {
        self.im
    }
    #[inline]
    fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }
    #[inline]
    fn norm_sqr(&self) -> T {
        self.re * self.re + self.im * self.im
    }
    #[inline]
    fn modulus(&self) -> T {
        self.re.hypot(self.im)
    }
    #[inline]
    fn scale_real(&self, factor: T) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
    #[inline]
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline]
    fn is_nan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    #[inline]
    fn abs(self) -> Self {
        Self::new(self.re.hypot(self.im), T::ZERO)
    }
    #[inline]
    fn sqrt(self) -> Self {
        complex_sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        complex_exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        complex_ln(self)
    }
    #[inline]
    fn log10(self) -> Self {
        let inv_ln10 = T::ONE / T::from_f64(core::f64::consts::LN_10);
        let l = complex_ln(self);
        Self::new(l.re * inv_ln10, l.im * inv_ln10)
    }
    #[inline]
    fn sin(self) -> Self {
        let (sr, cr) = self.re.sin_cos();
        Self::new(sr * self.im.cosh(), cr * self.im.sinh())
    }
    #[inline]
    fn cos(self) -> Self {
        let (sr, cr) = self.re.sin_cos();
        Self::new(cr * self.im.cosh(), -sr * self.im.sinh())
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (sr, cr) = self.re.sin_cos();
        let sh = self.im.sinh();
        let ch = self.im.cosh();
        (Self::new(sr * ch, cr * sh), Self::new(cr * ch, -sr * sh))
    }
    #[inline]
    fn tan(self) -> Self {
        smith_div(ComplexField::sin(self), ComplexField::cos(self))
    }
    #[inline]
    fn sinh(self) -> Self {
        let (si, ci) = self.im.sin_cos();
        Self::new(self.re.sinh() * ci, self.re.cosh() * si)
    }
    #[inline]
    fn cosh(self) -> Self {
        let (si, ci) = self.im.sin_cos();
        Self::new(self.re.cosh() * ci, self.re.sinh() * si)
    }
    #[inline]
    fn tanh(self) -> Self {
        smith_div(ComplexField::sinh(self), ComplexField::cosh(self))
    }
    #[inline]
    fn asin(self) -> Self {
        let i = Self::imag_unit();
        let one = Self::one();
        let inner = i * self + complex_sqrt(one - self * self);
        -i * complex_ln(inner)
    }
    #[inline]
    fn acos(self) -> Self {
        let i = Self::imag_unit();
        let one = Self::one();
        let inner = self + i * complex_sqrt(one - self * self);
        -i * complex_ln(inner)
    }
    #[inline]
    fn atan(self) -> Self {
        let i = Self::imag_unit();
        let one = Self::one();
        let ratio = smith_div(one + i * self, one - i * self);
        (Self::new(T::ZERO, -T::HALF)) * complex_ln(ratio)
    }
    #[inline]
    fn powf(self, exp: T) -> Self {
        if self.is_zero() {
            return if exp > T::ZERO {
                Self::zero()
            } else if exp == T::ZERO {
                Self::one()
            } else {
                Self::nan()
            };
        }
        complex_exp(complex_ln(self).scale_real(exp))
    }
    #[inline]
    fn powc(self, exp: Self) -> Self {
        if exp.im == T::ZERO {
            return ComplexField::powf(self, exp.re);
        }
        if self.is_zero() {
            return if exp.re > T::ZERO {
                Self::zero()
            } else {
                Self::nan()
            };
        }
        complex_exp(exp * complex_ln(self))
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_field(self, n)
    }
}
