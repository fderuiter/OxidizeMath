//! Core scalar traits separating general differentiable/enclosure operations (`Scalar`)
//! from ordered IEEE-754 floating-point operations (`RealScalar`).
// [cite:math_commons]

use crate::ulp::{next_down_f64, next_up_f64, pad_ulp_down_f64, pad_ulp_up_f64, ulp_distance_f64};
use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// General algebraic and elementary transcendental scalar trait.
///
/// Implemented uniformly for real floats (`f32`, `f64`), forward automatic differentiation
/// jets (`Dual`, `DualVec`, `HyperDual`), and verified set enclosures (`Interval`, `Ball`).
/// Does **not** require `PartialOrd` so dual numbers and interval enclosures participate
/// in generic numerical kernels without false total-ordering assumptions.
pub trait Scalar:
    Copy
    + Clone
    + Debug
    + Default
    + PartialEq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Additive identity `0`.
    const ZERO: Self;
    /// Multiplicative identity `1`.
    const ONE: Self;
    /// Constant `2`.
    const TWO: Self;
    /// Constant `0.5`.
    const HALF: Self;
    /// Archimedes' constant $\pi$.
    const PI: Self;
    /// Full circle constant $\tau = 2\pi$.
    const TAU: Self;

    /// Converts a constant `f64` literal into `Self`.
    fn from_f64(v: f64) -> Self;

    /// Converts an unsigned index/count into `Self`.
    #[inline]
    fn from_usize(v: usize) -> Self {
        Self::from_f64(v as f64)
    }

    /// Returns `true` if all components of `self` are finite.
    fn is_finite(self) -> bool;

    /// Absolute value $|x|$.
    fn abs(self) -> Self;

    /// Signum function returning `-1`, `0`, or `+1` (or subgradient enclosure).
    fn signum(self) -> Self;

    /// Reciprocal $1 / x$.
    #[inline]
    fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Principal square root $\sqrt{x}$.
    fn sqrt(self) -> Self;

    /// Integer power $x^n$ computed via binary exponentiation.
    fn powi(self, n: i32) -> Self;

    /// Real power $x^y$.
    fn powf(self, exp: Self) -> Self;

    /// Natural exponential $e^x$.
    fn exp(self) -> Self;

    /// Natural logarithm $\ln(x)$.
    fn ln(self) -> Self;

    /// Sine $\sin(x)$.
    fn sin(self) -> Self;

    /// Cosine $\cos(x)$.
    fn cos(self) -> Self;

    /// Simultaneous $(\sin(x), \cos(x))$.
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    /// Tangent $\tan(x)$.
    fn tan(self) -> Self;

    /// Principal inverse sine $\arcsin(x)$.
    fn asin(self) -> Self;

    /// Principal inverse cosine $\arccos(x)$.
    fn acos(self) -> Self;

    /// Principal inverse tangent $\arctan(x)$.
    fn atan(self) -> Self;

    /// Four-quadrant inverse tangent $\operatorname{atan2}(y, x)$.
    fn atan2(self, other: Self) -> Self;

    /// Hyperbolic sine $\sinh(x)$.
    fn sinh(self) -> Self;

    /// Hyperbolic cosine $\cosh(x)$.
    fn cosh(self) -> Self;

    /// Hyperbolic tangent $\tanh(x)$.
    fn tanh(self) -> Self;

    /// Overflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$.
    fn hypot(self, other: Self) -> Self;
}

/// Ordered IEEE-754 floating-point scalar (`f32` or `f64`) supporting comparisons
/// and portable bit-level outward ULP rounding.
pub trait RealScalar: Scalar + PartialOrd {
    /// Machine epsilon ($2^{-52}$ for `f64`, $2^{-23}$ for `f32`).
    const EPSILON: Self;
    /// Square root of machine epsilon ($\sqrt{\epsilon}$).
    const SQRT_EPSILON: Self;
    /// Smallest positive normal value.
    const MIN_POSITIVE: Self;
    /// Largest finite representable value.
    const MAX_VALUE: Self;
    /// Positive infinity $+\infty$.
    const INFINITY: Self;
    /// Negative infinity $-\infty$.
    const NEG_INFINITY: Self;
    /// Quiet Not-a-Number (`NaN`).
    const NAN: Self;

    /// Widens `self` to `f64`.
    fn to_f64(self) -> f64;
    /// Returns `true` if `self` is `NaN`.
    fn is_nan(self) -> bool;
    /// IEEE-754 minimum of `self` and `other` (propagating non-NaN where standard).
    fn min(self, other: Self) -> Self;
    /// IEEE-754 maximum of `self` and `other`.
    fn max(self, other: Self) -> Self;
    /// Copies the sign of `sign` onto the magnitude of `self`.
    fn copysign(self, sign: Self) -> Self;
    /// Largest integer less than or equal to `self`.
    fn floor(self) -> Self;
    /// Smallest integer greater than or equal to `self`.
    fn ceil(self) -> Self;
    /// Least representable float strictly greater than `self`.
    fn next_up(self) -> Self;
    /// Greatest representable float strictly less than `self`.
    fn next_down(self) -> Self;
    /// Pads `self` upward by `steps` IEEE-754 ULPs.
    fn pad_ulp_up(self, steps: u32) -> Self;
    /// Pads `self` downward by `steps` IEEE-754 ULPs.
    fn pad_ulp_down(self, steps: u32) -> Self;
    /// Exact IEEE-754 ULP distance between `self` and `other`.
    fn ulp_distance(self, other: Self) -> u64;
}

#[inline]
pub(crate) fn powi_binary<T: Scalar>(mut base: T, n: i32) -> T {
    if n == 0 {
        return T::ONE;
    }
    let mut exp: u32 = n.unsigned_abs();
    if n < 0 {
        base = T::ONE / base;
    }
    let mut acc = T::ONE;
    while exp > 0 {
        if (exp & 1) != 0 {
            acc *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    acc
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const TAU: Self = core::f64::consts::TAU;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
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
            f64::NAN
        }
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_binary(self, n)
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
    fn hypot(self, other: Self) -> Self {
        libm::hypot(self, other)
    }
}

impl RealScalar for f64 {
    const EPSILON: Self = f64::EPSILON;
    const SQRT_EPSILON: Self = 1.490_116_119_384_765_6e-8;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX_VALUE: Self = f64::MAX;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const NAN: Self = f64::NAN;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
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
    fn next_up(self) -> Self {
        next_up_f64(self)
    }
    #[inline]
    fn next_down(self) -> Self {
        next_down_f64(self)
    }
    #[inline]
    fn pad_ulp_up(self, steps: u32) -> Self {
        pad_ulp_up_f64(self, steps)
    }
    #[inline]
    fn pad_ulp_down(self, steps: u32) -> Self {
        pad_ulp_down_f64(self, steps)
    }
    #[inline]
    fn ulp_distance(self, other: Self) -> u64 {
        ulp_distance_f64(self, other)
    }
}
