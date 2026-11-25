// [cite:math_commons]
use crate::traits::{ComplexField, Field, RealField, Ring, Scalar};

impl Scalar for f32 {}
impl Scalar for f64 {}

macro_rules! impl_ieee_float {
    (
        $t:ty,
        $consts:ident,
        fabs = $fabs:path,
        sqrt = $sqrt:path,
        cbrt = $cbrt:path,
        hypot = $hypot:path,
        exp = $exp:path,
        expm1 = $expm1:path,
        ln = $ln:path,
        ln1p = $ln1p:path,
        log2 = $log2:path,
        log10 = $log10:path,
        sin = $sin:path,
        cos = $cos:path,
        sincos = $sincos:path,
        tan = $tan:path,
        asin = $asin:path,
        acos = $acos:path,
        atan = $atan:path,
        atan2 = $atan2:path,
        sinh = $sinh:path,
        cosh = $cosh:path,
        tanh = $tanh:path,
        asinh = $asinh:path,
        acosh = $acosh:path,
        atanh = $atanh:path,
        pow = $pow:path,
        floor = $floor:path,
        ceil = $ceil:path,
        round = $round:path,
        copysign = $copysign:path
    ) => {
        impl Ring for $t {
            #[inline(always)]
            fn zero() -> Self {
                0.0
            }
            #[inline(always)]
            fn one() -> Self {
                1.0
            }
            #[inline(always)]
            fn is_zero(&self) -> bool {
                *self == 0.0
            }
        }

        impl Field for $t {
            #[inline(always)]
            fn checked_inverse(&self) -> Option<Self> {
                if *self == 0.0 || !<$t>::is_finite(*self) {
                    None
                } else {
                    let inv = 1.0 / *self;
                    <$t>::is_finite(inv).then_some(inv)
                }
            }
            #[inline(always)]
            fn multiplicative_inverse(&self) -> Self {
                1.0 / *self
            }
        }

        impl ComplexField for $t {
            type Real = $t;

            #[inline(always)]
            fn from_real(re: Self) -> Self {
                re
            }
            #[inline(always)]
            fn nan() -> Self {
                <$t>::NAN
            }
            #[inline(always)]
            fn imag_unit() -> Self {
                <$t>::NAN
            }
            #[inline(always)]
            fn real(self) -> Self {
                self
            }
            #[inline(always)]
            fn imag(self) -> Self {
                0.0
            }
            #[inline(always)]
            fn conj(self) -> Self {
                self
            }
            #[inline(always)]
            fn norm_sqr(self) -> Self {
                self * self
            }
            #[inline(always)]
            fn abs(self) -> Self {
                $fabs(self)
            }
            #[inline(always)]
            fn arg(self) -> Self {
                $atan2(0.0, self)
            }
            #[inline(always)]
            fn scale(self, factor: Self) -> Self {
                self * factor
            }
            #[inline(always)]
            fn is_all_finite(self) -> bool {
                <$t>::is_finite(self)
            }
            #[inline(always)]
            fn is_any_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            #[inline(always)]
            fn csqrt(self) -> Self {
                $sqrt(self)
            }
            #[inline(always)]
            fn cbrt(self) -> Self {
                $cbrt(self)
            }
            #[inline(always)]
            fn exp(self) -> Self {
                $exp(self)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                $ln(self)
            }
            #[inline(always)]
            fn log2(self) -> Self {
                $log2(self)
            }
            #[inline(always)]
            fn log10(self) -> Self {
                $log10(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                $sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                $cos(self)
            }
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                $sincos(self)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                $tan(self)
            }
            #[inline(always)]
            fn asin(self) -> Self {
                $asin(self)
            }
            #[inline(always)]
            fn acos(self) -> Self {
                $acos(self)
            }
            #[inline(always)]
            fn atan(self) -> Self {
                $atan(self)
            }
            #[inline(always)]
            fn sinh(self) -> Self {
                $sinh(self)
            }
            #[inline(always)]
            fn cosh(self) -> Self {
                $cosh(self)
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                $tanh(self)
            }
            #[inline(always)]
            fn asinh(self) -> Self {
                $asinh(self)
            }
            #[inline(always)]
            fn acosh(self) -> Self {
                $acosh(self)
            }
            #[inline(always)]
            fn atanh(self) -> Self {
                $atanh(self)
            }
            #[inline(always)]
            fn powf(self, exp: Self) -> Self {
                $pow(self, exp)
            }
            #[inline(always)]
            fn powc(self, exp: Self) -> Self {
                $pow(self, exp)
            }
            #[inline(always)]
            fn powi(self, exp: i32) -> Self {
                num_traits::Float::powi(self, exp)
            }
            #[inline(always)]
            fn floor(self) -> Self {
                $floor(self)
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                $ceil(self)
            }
            #[inline(always)]
            fn round(self) -> Self {
                $round(self)
            }
            #[inline(always)]
            fn signum(self) -> Self {
                if <$t>::is_nan(self) {
                    <$t>::NAN
                } else {
                    $copysign(1.0, self)
                }
            }
        }

        impl RealField for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const HALF: Self = 0.5;
            const PI: Self = core::$consts::consts::PI;
            const FRAC_PI_2: Self = core::$consts::consts::FRAC_PI_2;
            const TAU: Self = core::$consts::consts::TAU;
            const E: Self = core::$consts::consts::E;
            const EPSILON: Self = <$t>::EPSILON;
            const MIN_POSITIVE: Self = <$t>::MIN_POSITIVE;
            const MIN_VALUE: Self = <$t>::MIN;
            const MAX_VALUE: Self = <$t>::MAX;
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            const NAN: Self = <$t>::NAN;
            const DEFAULT_TOL_SCALE: Self = 16.0;

            #[inline(always)]
            fn from_f64(val: f64) -> Self {
                val as $t
            }
            #[inline(always)]
            fn from_usize(n: usize) -> Self {
                n as $t
            }
            #[inline(always)]
            fn from_i32(n: i32) -> Self {
                n as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                f64::from(self)
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
            #[inline(always)]
            fn hypot(self, other: Self) -> Self {
                $hypot(self, other)
            }
            #[inline(always)]
            fn atan2(self, other: Self) -> Self {
                $atan2(self, other)
            }
            #[inline(always)]
            fn expm1(self) -> Self {
                $expm1(self)
            }
            #[inline(always)]
            fn ln_1p(self) -> Self {
                $ln1p(self)
            }
            #[inline(always)]
            fn copysign(self, sign: Self) -> Self {
                $copysign(self, sign)
            }
            #[inline(always)]
            fn min(self, other: Self) -> Self {
                <$t>::min(self, other)
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                <$t>::max(self, other)
            }
            #[inline(always)]
            fn clamp(self, lo: Self, hi: Self) -> Self {
                <$t>::clamp(self, lo, hi)
            }
            #[inline(always)]
            fn rem_euclid(self, rhs: Self) -> Self {
                num_traits::Euclid::rem_euclid(&self, &rhs)
            }
            #[inline(always)]
            fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
                <$t>::total_cmp(self, other)
            }
        }
    };
}

impl_ieee_float!(
    f32,
    f32,
    fabs = libm::fabsf,
    sqrt = libm::sqrtf,
    cbrt = libm::cbrtf,
    hypot = libm::hypotf,
    exp = libm::expf,
    expm1 = libm::expm1f,
    ln = libm::logf,
    ln1p = libm::log1pf,
    log2 = libm::log2f,
    log10 = libm::log10f,
    sin = libm::sinf,
    cos = libm::cosf,
    sincos = libm::sincosf,
    tan = libm::tanf,
    asin = libm::asinf,
    acos = libm::acosf,
    atan = libm::atanf,
    atan2 = libm::atan2f,
    sinh = libm::sinhf,
    cosh = libm::coshf,
    tanh = libm::tanhf,
    asinh = libm::asinhf,
    acosh = libm::acoshf,
    atanh = libm::atanhf,
    pow = libm::powf,
    floor = libm::floorf,
    ceil = libm::ceilf,
    round = libm::roundf,
    copysign = libm::copysignf
);

impl_ieee_float!(
    f64,
    f64,
    fabs = libm::fabs,
    sqrt = libm::sqrt,
    cbrt = libm::cbrt,
    hypot = libm::hypot,
    exp = libm::exp,
    expm1 = libm::expm1,
    ln = libm::log,
    ln1p = libm::log1p,
    log2 = libm::log2,
    log10 = libm::log10,
    sin = libm::sin,
    cos = libm::cos,
    sincos = libm::sincos,
    tan = libm::tan,
    asin = libm::asin,
    acos = libm::acos,
    atan = libm::atan,
    atan2 = libm::atan2,
    sinh = libm::sinh,
    cosh = libm::cosh,
    tanh = libm::tanh,
    asinh = libm::asinh,
    acosh = libm::acosh,
    atanh = libm::atanh,
    pow = libm::pow,
    floor = libm::floor,
    ceil = libm::ceil,
    round = libm::round,
    copysign = libm::copysign
);
