// [cite:math_scalars]
//! `0`-ULP `libm`-backed IEEE-754 scalar kernels (`f32`, `f64`) and primitive integer rings.

use crate::traits::{ComplexField, Field, RealField, RefOps, Ring};
use core::cmp::Ordering;

/// Computes the correctly rounded IEEE-754 double-precision square root $\sqrt{x}$ via `libm::sqrt`.
///
/// Reference: IEEE Std 754-2019 §9.2; FDLIBM `e_sqrt.c` (`libm::sqrt`).
///
/// # Stability
/// Achieves `0`-ULP error across all normal and subnormal `f64` inputs on both native and
/// `wasm32-unknown-unknown` targets (including exact boundary rounding such as
/// `sqrt_f64(3.9999999999999996) == 1.9999999999999998`).
///
/// # Domain of Validity
/// Returns `0.0` (preserving sign of `-0.0`) for zero, `+INFINITY` for `+INFINITY`, and `NaN` for
/// negative nonzero inputs or `NaN`.
#[inline]
#[must_use]
pub fn sqrt_f64(x: f64) -> f64 {
    libm::sqrt(x)
}

/// Computes the correctly rounded IEEE-754 single-precision square root $\sqrt{x}$ via `libm::sqrtf`.
///
/// Reference: IEEE Std 754-2019 §9.2; FDLIBM `e_sqrtf.c` (`libm::sqrtf`).
///
/// # Stability
/// Achieves `0`-ULP IEEE-754 rounding across all normal and subnormal `f32` inputs on host and WASM.
///
/// # Domain of Validity
/// Defined for `x >= 0.0` (preserving `-0.0`); returns `NaN` for `x < 0.0` or `NaN`.
#[inline]
#[must_use]
pub fn sqrt_f32(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[inline]
fn veltkamp_split(a: f64) -> (f64, f64) {
    const FACTOR: f64 = 134_217_729.0; // 2^27 + 1
    let c = FACTOR * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

#[inline]
fn dekker_two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (a_hi, a_lo) = veltkamp_split(a);
    let (b_hi, b_lo) = veltkamp_split(b);
    let err = (((a_hi * b_hi - p) + a_hi * b_lo) + a_lo * b_hi) + a_lo * b_lo;
    (p, err)
}

#[inline]
fn dekker_two_sq(a: f64) -> (f64, f64) {
    let p = a * a;
    let (hi, lo) = veltkamp_split(a);
    let err = ((hi * hi - p) + 2.0 * hi * lo) + lo * lo;
    (p, err)
}

/// Computes the `0`-ULP overflow- and underflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$.
///
/// Reference: IEEE Std 754-2019 §9.2; Borges (2020), ACM TOMS Algorithm 1014, DOI: 10.1145/3428446.
///
/// # Stability
/// Rescales exponents by an exact power of two, accumulates $x^2 + y^2$ in 106-bit Dekker
/// doubled precision, and refines `libm::sqrt` by the exact squared residual for `0`-ULP accuracy
/// across native and `wasm32-unknown-unknown` targets.
///
/// # Domain of Validity
/// Valid for all `f64` pairs; returns `+INFINITY` if either argument is infinite (even if the other is `NaN`).
#[inline]
#[must_use]
pub fn hypot_f64(x: f64, y: f64) -> f64 {
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let ax = libm::fabs(x);
    let ay = libm::fabs(y);
    let (hi, lo) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    if hi == 0.0 {
        return 0.0;
    }
    if lo <= hi * f64::from_bits(0x3e30_0000_0000_0000) {
        return hi;
    }
    let exp = libm::ilogb(hi);
    let a = libm::scalbn(hi, -exp);
    let b = libm::scalbn(lo, -exp);
    let (h2, h2_err) = dekker_two_sq(a);
    let (l2, l2_err) = dekker_two_sq(b);
    let sum_hi = h2 + l2;
    let sum_lo = ((h2 - sum_hi) + l2) + (h2_err + l2_err);
    let r = libm::sqrt(sum_hi);
    let (r2, r2_err) = dekker_two_sq(r);
    let rem = ((sum_hi - r2) - r2_err) + sum_lo;
    libm::scalbn(r + rem / (2.0 * r), exp)
}

/// Computes the `0`-ULP overflow- and underflow-safe single-precision hypotenuse $\sqrt{x^2 + y^2}$.
///
/// Reference: IEEE Std 754-2019 §9.2 (`libm::sqrt` over exact widened `f64` sum of squares).
///
/// # Stability
/// Widens 24-bit `f32` significands into exact 53-bit `f64` products before evaluating `libm::sqrt`,
/// achieving `0`-ULP rounding with zero intermediate overflow or underflow.
///
/// # Domain of Validity
/// Valid for all `f32` pairs; returns `+INFINITY` if either operand is infinite.
#[inline]
#[must_use]
pub fn hypot_f32(x: f32, y: f32) -> f32 {
    if x.is_infinite() || y.is_infinite() {
        return f32::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    let ax = f64::from(x);
    let ay = f64::from(y);
    libm::sqrt(ax * ax + ay * ay) as f32
}

#[inline]
pub(crate) fn powi_field<T: Field>(mut base: T, exp: i32) -> T {
    if exp == 0 {
        return T::one();
    }
    let mut n = exp.unsigned_abs();
    if exp < 0 {
        base = T::one() / base;
    }
    let mut acc = T::one();
    while n > 0 {
        if (n & 1) == 1 {
            acc *= base.clone();
        }
        n >>= 1;
        if n > 0 {
            let next = base.ref_mul(&base);
            base = next;
        }
    }
    acc
}

macro_rules! impl_signed_int_ring {
    ($($t:ty),* $(,)?) => {
        $(
            impl Ring for $t {
                #[inline]
                fn zero() -> Self { 0 }
                #[inline]
                fn one() -> Self { 1 }
                #[inline]
                fn is_zero(&self) -> bool { *self == 0 }
            }
        )*
    };
}

impl_signed_int_ring!(i32, i64, i128, isize);

#[cfg(feature = "dashu")]
impl Ring for dashu::integer::IBig {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
    #[inline]
    fn is_zero(&self) -> bool {
        Self::is_zero(self)
    }
    #[inline]
    fn ring_ref_add(&self, rhs: &Self) -> Self {
        self + rhs
    }
    #[inline]
    fn ring_ref_sub(&self, rhs: &Self) -> Self {
        self - rhs
    }
    #[inline]
    fn ring_ref_mul(&self, rhs: &Self) -> Self {
        self * rhs
    }
    #[inline]
    fn ring_ref_neg(&self) -> Self {
        -self
    }
}

impl Ring for f64 {
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

impl Field for f64 {
    #[inline]
    fn ref_div(&self, rhs: &Self) -> Self {
        *self / *rhs
    }
    #[inline]
    fn try_inv(&self) -> Option<Self> {
        (*self != 0.0).then(|| 1.0 / *self)
    }
}

impl ComplexField for f64 {
    type Real = Self;

    #[inline]
    fn from_real(re: Self) -> Self {
        re
    }
    #[inline]
    fn from_f64(val: f64) -> Self {
        val
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
        libm::fabs(*self)
    }
    #[inline]
    fn scale_real(&self, factor: Self) -> Self {
        *self * factor
    }
    #[inline]
    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
    #[inline]
    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        sqrt_f64(self)
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
    fn powf(self, exp: Self) -> Self {
        libm::pow(self, exp)
    }
    #[inline]
    fn powc(self, exp: Self) -> Self {
        libm::pow(self, exp)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_field(self, n)
    }
}

impl RealField for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const TAU: Self = core::f64::consts::TAU;
    const E: Self = core::f64::consts::E;
    const EPSILON: Self = f64::EPSILON;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX_VALUE: Self = f64::MAX;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const NAN: Self = f64::NAN;
    const DEFAULT_TOL_SCALE: Self = 16.0;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_usize(value: usize) -> Self {
        value as f64
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        f64::clamp(self, min, max)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysign(self, sign)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        hypot_f64(self, other)
    }
    #[inline]
    fn diff_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        const SPLIT_MAX: f64 = 1.0e145;
        if libm::fabs(a) < SPLIT_MAX
            && libm::fabs(b) < SPLIT_MAX
            && libm::fabs(c) < SPLIT_MAX
            && libm::fabs(d) < SPLIT_MAX
        {
            let (ab, ab_err) = dekker_two_prod(a, b);
            let (cd, cd_err) = dekker_two_prod(c, d);
            let diff = ab - cd;
            let s_err = ((ab - diff) - cd) + (ab_err - cd_err);
            diff + s_err
        } else {
            a * b - c * d
        }
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
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
    fn fract(self) -> Self {
        self - libm::trunc(self)
    }
    #[inline]
    fn expm1(self) -> Self {
        libm::expm1(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2(self)
    }
    #[inline]
    fn rem_euclid(self, rhs: Self) -> Self {
        let r = self % rhs;
        if r < 0.0 { r + libm::fabs(rhs) } else { r }
    }
    #[inline]
    fn next_up(self) -> Self {
        f64::next_up(self)
    }
    #[inline]
    fn next_down(self) -> Self {
        f64::next_down(self)
    }
    #[inline]
    fn total_cmp(&self, other: &Self) -> Ordering {
        f64::total_cmp(self, other)
    }
}

#[path = "ieee_f32.rs"]
mod ieee_f32;
