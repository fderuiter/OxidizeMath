// [cite:math_scalars]
//! `0`-ULP `libm`-backed single-precision (`f32`) implementations of the scalar hierarchy.

use crate::ieee::{hypot_f32, powi_field, sqrt_f32};
use crate::traits::{ComplexField, Field, RealField, Ring};
use core::cmp::Ordering;

const F32_INV_2POW24: f32 = 1.0 / ((1u32 << 24) as f32);
const F32_MAX_BELOW_ONE: f32 = f32::from_bits(0x3f7f_ffff);
const F32_MIN_ABOVE_ZERO: f32 = 1.0e-7_f32;

impl Ring for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl Field for f32 {
    #[inline]
    fn ref_div(&self, rhs: &Self) -> Self {
        *self / *rhs
    }
    #[inline]
    fn try_inv(&self) -> Option<Self> {
        (*self != 0.0).then(|| 1.0 / *self)
    }
}

impl ComplexField for f32 {
    type Real = Self;

    #[inline]
    fn from_real(re: Self) -> Self {
        re
    }
    #[inline]
    fn from_f64(val: f64) -> Self {
        val as f32
    }
    #[inline]
    fn imag_unit() -> Self {
        Self::NAN
    }
    #[inline]
    fn nan() -> Self {
        Self::NAN
    }
    #[inline]
    fn real(&self) -> Self {
        *self
    }
    #[inline]
    fn imag(&self) -> Self {
        0.0
    }
    #[inline]
    fn conj(&self) -> Self {
        *self
    }
    #[inline]
    fn norm_sqr(&self) -> Self {
        *self * *self
    }
    #[inline]
    fn modulus(&self) -> Self {
        libm::fabsf(*self)
    }
    #[inline]
    fn scale_real(&self, factor: Self) -> Self {
        *self * factor
    }
    #[inline]
    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
    #[inline]
    fn is_nan(&self) -> bool {
        f32::is_nan(*self)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        sqrt_f32(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10f(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        libm::sincosf(self)
    }
    #[inline]
    fn tan(self) -> Self {
        libm::tanf(self)
    }
    #[inline]
    fn sinh(self) -> Self {
        libm::sinhf(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        libm::coshf(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanhf(self)
    }
    #[inline]
    fn asin(self) -> Self {
        libm::asinf(self)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acosf(self)
    }
    #[inline]
    fn atan(self) -> Self {
        libm::atanf(self)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        libm::powf(self, exp)
    }
    #[inline]
    fn powc(self, exp: Self) -> Self {
        libm::powf(self, exp)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_field(self, n)
    }
}

impl RealField for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;
    const E: Self = core::f32::consts::E;
    const EPSILON: Self = f32::EPSILON;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX_VALUE: Self = f32::MAX;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const DEFAULT_TOL_SCALE: Self = 16.0;

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn from_usize(value: usize) -> Self {
        value as f32
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        f32::clamp(self, min, max)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysignf(self, sign)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        hypot_f32(self, other)
    }
    #[inline]
    fn diff_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        ((f64::from(a) * f64::from(b)) - (f64::from(c) * f64::from(d))) as f32
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::roundf(self)
    }
    #[inline]
    fn fract(self) -> Self {
        self - libm::truncf(self)
    }
    #[inline]
    fn expm1(self) -> Self {
        libm::expm1f(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2f(self)
    }
    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
        let r = self % rhs;
        if r < 0.0 { r + libm::fabsf(rhs) } else { r }
    }
    #[inline]
    fn next_up(self) -> Self {
        f32::next_up(self)
    }
    #[inline]
    fn next_down(self) -> Self {
        f32::next_down(self)
    }
    #[inline]
    fn total_cmp(&self, other: &Self) -> Ordering {
        f32::total_cmp(self, other)
    }
    #[inline]
    fn open_unit_from_f64(value: f64) -> Self {
        (value as f32).clamp(F32_MIN_ABOVE_ZERO, F32_MAX_BELOW_ONE)
    }
    #[inline]
    fn uniform_closed_open_from_u64(u64_word: u64) -> Self {
        (((u64_word >> 40) as u32) as f32) * F32_INV_2POW24
    }
    #[inline]
    fn uniform_open_from_u64(u64_word: u64) -> Self {
        ((((u64_word >> 40) as u32 as f32) + 0.5) * F32_INV_2POW24)
            .clamp(F32_MIN_ABOVE_ZERO, F32_MAX_BELOW_ONE)
    }
}
