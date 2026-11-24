// [cite:math_scalars]
//! Deterministic `libm`-backed [`Scalar`] and [`RealScalar`] implementations for `f64` and `f32`.

use crate::rounding::{
    next_down_f32, next_down_f64, next_up_f32, next_up_f64, step_down_f32, step_down_f64,
    step_up_f32, step_up_f64, ulp_distance_f32, ulp_distance_f64,
};
use crate::traits::{RealScalar, Scalar, powi_generic};

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn real_part(self) -> f64 {
        self
    }
    #[inline]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_generic(self, n)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        libm::pow(self, exp)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        libm::sincos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        libm::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        libm::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        libm::atan(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    #[inline]
    fn sinh(self) -> Self {
        libm::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        libm::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanh(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        libm::asinh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        libm::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        libm::atanh(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypot(self, other)
    }
}

impl RealScalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX_VAL: Self = f64::MAX;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const NAN: Self = f64::NAN;
    const TAU: Self = core::f64::consts::TAU;
    const FRAC_PI_2: Self = core::f64::consts::FRAC_PI_2;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn is_infinite(self) -> bool {
        f64::is_infinite(self)
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        f64::is_sign_negative(self)
    }
    #[inline]
    fn min_val(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max_val(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysign(self, sign)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        libm::trunc(self)
    }
    #[inline]
    fn fma(self, mul: Self, add: Self) -> Self {
        libm::fma(self, mul, add)
    }
    #[inline]
    fn next_up(self) -> Self {
        next_up_f64(self)
    }
    #[inline]
    fn next_down(self) -> Self {
        next_down_f64(self)
    }
    #[inline]
    fn step_up(self, ulps: u32) -> Self {
        step_up_f64(self, ulps)
    }
    #[inline]
    fn step_down(self, ulps: u32) -> Self {
        step_down_f64(self, ulps)
    }
    #[inline]
    fn ulp_distance(self, other: Self) -> u64 {
        ulp_distance_f64(self, other)
    }
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn real_part(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_generic(self, n)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        libm::powf(self, exp)
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
    fn log2(self) -> Self {
        libm::log2f(self)
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
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
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
    fn asinh(self) -> Self {
        libm::asinhf(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        libm::acoshf(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        libm::atanhf(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypotf(self, other)
    }
}

impl RealScalar for f32 {
    const EPSILON: Self = f32::EPSILON;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX_VAL: Self = f32::MAX;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const TAU: Self = core::f32::consts::TAU;
    const FRAC_PI_2: Self = core::f32::consts::FRAC_PI_2;

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_infinite(self) -> bool {
        f32::is_infinite(self)
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        f32::is_sign_negative(self)
    }
    #[inline]
    fn min_val(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max_val(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysignf(self, sign)
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
    fn trunc(self) -> Self {
        libm::truncf(self)
    }
    #[inline]
    fn fma(self, mul: Self, add: Self) -> Self {
        libm::fmaf(self, mul, add)
    }
    #[inline]
    fn next_up(self) -> Self {
        next_up_f32(self)
    }
    #[inline]
    fn next_down(self) -> Self {
        next_down_f32(self)
    }
    #[inline]
    fn step_up(self, ulps: u32) -> Self {
        step_up_f32(self, ulps)
    }
    #[inline]
    fn step_down(self, ulps: u32) -> Self {
        step_down_f32(self, ulps)
    }
    #[inline]
    fn ulp_distance(self, other: Self) -> u64 {
        ulp_distance_f32(self, other)
    }
}
