//! `Scalar` and `RealScalar` implementations for IEEE-754 binary32 (`f32`).
// [cite:math_commons]

use crate::traits::{RealScalar, Scalar, powi_binary};
use crate::ulp::{next_down_f32, next_up_f32, pad_ulp_down_f32, pad_ulp_up_f32, ulp_distance_f32};

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn signum(self) -> Self {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else if self == 0.0 {
            0.0
        } else {
            f32::NAN
        }
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_binary(self, n)
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
    fn hypot(self, other: Self) -> Self {
        libm::hypotf(self, other)
    }
}

impl RealScalar for f32 {
    const EPSILON: Self = f32::EPSILON;
    const SQRT_EPSILON: Self = 3.452_669_8e-4;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX_VALUE: Self = f32::MAX;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
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
    fn next_up(self) -> Self {
        next_up_f32(self)
    }
    #[inline]
    fn next_down(self) -> Self {
        next_down_f32(self)
    }
    #[inline]
    fn pad_ulp_up(self, steps: u32) -> Self {
        pad_ulp_up_f32(self, steps)
    }
    #[inline]
    fn pad_ulp_down(self, steps: u32) -> Self {
        pad_ulp_down_f32(self, steps)
    }
    #[inline]
    fn ulp_distance(self, other: Self) -> u64 {
        ulp_distance_f32(self, other)
    }
}
