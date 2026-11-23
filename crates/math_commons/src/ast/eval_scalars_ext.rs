//! `EvalScalar` implementations for `Dual<Complex64>` and `Jet2<N>`.
// [cite:math_commons]

use engine_symbolic::{Dual, Jet2};
use num_complex::Complex64;

use super::EvalScalar;

impl EvalScalar for Dual<Complex64> {
    const NAN: Self = Self::new(
        Complex64::new(f64::NAN, f64::NAN),
        Complex64::new(f64::NAN, f64::NAN),
    );

    #[inline]
    fn from_real(val: f64) -> Self {
        Self::constant(Complex64::new(val, 0.0))
    }
    #[inline]
    fn imag_unit() -> Self {
        Self::constant(Complex64::i())
    }
    #[inline]
    fn is_invalid_scalar(self) -> bool {
        self.re.re.is_nan() || self.re.im.is_nan()
    }
    #[inline]
    fn div_checked(self, rhs: Self) -> Self {
        if rhs.re.norm_sqr() < 1e-12 {
            Self::NAN
        } else {
            self / rhs
        }
    }
    #[inline]
    fn pow_scalar(self, rhs: Self) -> Self {
        self.powf(rhs)
    }
    #[inline]
    fn extra_const(name: &str) -> Option<Self> {
        name.eq_ignore_ascii_case("i").then(Self::imag_unit)
    }
    #[inline]
    fn sin_s(self) -> Self {
        self.sin()
    }
    #[inline]
    fn cos_s(self) -> Self {
        self.cos()
    }
    #[inline]
    fn tan_s(self) -> Self {
        self.tan()
    }
    #[inline]
    fn exp_s(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln_s(self) -> Self {
        self.ln()
    }
    #[inline]
    fn log10_s(self) -> Self {
        self.log10()
    }
    #[inline]
    fn sqrt_s(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn abs_s(self) -> Self {
        self.abs()
    }
    #[inline]
    fn sinh_s(self) -> Self {
        self.sinh()
    }
    #[inline]
    fn cosh_s(self) -> Self {
        self.cosh()
    }
    #[inline]
    fn tanh_s(self) -> Self {
        self.tanh()
    }
    #[inline]
    fn asin_s(self) -> Self {
        self.asin()
    }
    #[inline]
    fn acos_s(self) -> Self {
        self.acos()
    }
    #[inline]
    fn atan_s(self) -> Self {
        self.atan()
    }
    #[inline]
    fn step_s(self) -> Self {
        if self.is_invalid_scalar() {
            Self::NAN
        } else {
            self.step()
        }
    }
    #[inline]
    fn rect_s(self) -> Self {
        if self.is_invalid_scalar() {
            Self::NAN
        } else {
            self.rect()
        }
    }
    #[inline]
    fn min_s(self, other: Self) -> Self {
        if self.is_invalid_scalar() || other.is_invalid_scalar() {
            Self::NAN
        } else {
            self.min(other)
        }
    }
    #[inline]
    fn max_s(self, other: Self) -> Self {
        if self.is_invalid_scalar() || other.is_invalid_scalar() {
            Self::NAN
        } else {
            self.max(other)
        }
    }
    #[inline]
    fn clamp_s(self, lo: Self, hi: Self) -> Self {
        if self.is_invalid_scalar() || lo.is_invalid_scalar() || hi.is_invalid_scalar() {
            Self::NAN
        } else {
            self.clamp(lo, hi)
        }
    }
    #[inline]
    fn atan2_s(self, other: Self) -> Self {
        if self.is_invalid_scalar() || other.is_invalid_scalar() {
            Self::NAN
        } else {
            self.atan2(other)
        }
    }
}

impl<const N: usize> EvalScalar for Jet2<N> {
    const NAN: Self = Self::constant(f64::NAN);

    #[inline]
    fn from_real(val: f64) -> Self {
        Self::constant(val)
    }
    #[inline]
    fn imag_unit() -> Self {
        Self::NAN
    }
    #[inline]
    fn is_invalid_scalar(self) -> bool {
        self.val.is_nan()
    }
    #[inline]
    fn div_checked(self, rhs: Self) -> Self {
        if rhs.val == 0.0 {
            Self::NAN
        } else {
            self / rhs
        }
    }
    #[inline]
    fn pow_scalar(self, rhs: Self) -> Self {
        self.powf(rhs)
    }
    #[inline]
    fn sin_s(self) -> Self {
        self.sin()
    }
    #[inline]
    fn cos_s(self) -> Self {
        self.cos()
    }
    #[inline]
    fn tan_s(self) -> Self {
        self.tan()
    }
    #[inline]
    fn exp_s(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln_s(self) -> Self {
        self.ln()
    }
    #[inline]
    fn log10_s(self) -> Self {
        self.log10()
    }
    #[inline]
    fn sqrt_s(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn abs_s(self) -> Self {
        self.abs()
    }
    #[inline]
    fn sinh_s(self) -> Self {
        self.sinh()
    }
    #[inline]
    fn cosh_s(self) -> Self {
        self.cosh()
    }
    #[inline]
    fn tanh_s(self) -> Self {
        self.tanh()
    }
    #[inline]
    fn asin_s(self) -> Self {
        self.asin()
    }
    #[inline]
    fn acos_s(self) -> Self {
        self.acos()
    }
    #[inline]
    fn atan_s(self) -> Self {
        self.atan()
    }
    #[inline]
    fn step_s(self) -> Self {
        self.step()
    }
    #[inline]
    fn rect_s(self) -> Self {
        self.rect()
    }
    #[inline]
    fn min_s(self, other: Self) -> Self {
        self.min(other)
    }
    #[inline]
    fn max_s(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline]
    fn clamp_s(self, lo: Self, hi: Self) -> Self {
        self.clamp(lo, hi)
    }
    #[inline]
    fn atan2_s(self, other: Self) -> Self {
        self.atan2(other)
    }
}
