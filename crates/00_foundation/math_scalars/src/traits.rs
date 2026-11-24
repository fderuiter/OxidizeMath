#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Unordered field scalar contract supporting both real (`f32`, `f64`) and complex
/// ([`crate::Complex<T>`]) numerical linear algebra without requiring `PartialOrd`.
pub trait FieldScalar:
    Copy
    + Clone
    + fmt::Debug
    + fmt::Display
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
    /// Associated ordered real magnitude type (`Self` for real fields, `T` for `Complex<T>`).
    type Real: RealScalar;

    /// Additive identity $0$.
    const ZERO: Self;
    /// Multiplicative identity $1$.
    const ONE: Self;

    /// Converts an `f64` constant into `Self` along the real axis.
    fn from_f64(v: f64) -> Self;

    /// Embeds an associated real value into `Self` along the real axis.
    fn from_real(r: Self::Real) -> Self;

    /// Extracts the real part $\operatorname{Re}(z)$ of `self`.
    fn re_part(self) -> Self::Real;

    /// Involutive conjugate $\bar{z}$ (identity for real scalars, $(a, -b)$ for `Complex`).
    fn conj(self) -> Self;

    /// Non-negative squared modulus $|z|^2 = z \bar{z}$ in [`Self::Real`].
    fn norm_sqr(self) -> Self::Real;

    /// Non-negative Euclidean modulus $|z|$ in [`Self::Real`].
    fn abs(self) -> Self::Real;

    /// Multiplies `self` by a real scalar `r`.
    fn scale_real(self, r: Self::Real) -> Self;

    /// Computes $1 / z$, returning [`ScalarError::NonInvertible`] when `self` is zero or non-finite.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] if `self == Self::ZERO` or `!self.is_finite()`.
    fn try_inv(self) -> Result<Self, ScalarError>;

    /// Returns `true` if all coordinates are finite (neither `NaN` nor infinite).
    fn is_finite(self) -> bool;

    /// Returns `true` if any coordinate is `NaN`.
    fn is_nan(self) -> bool;
}

/// Ordered IEEE-754 real scalar contract extending [`FieldScalar`] with `PartialOrd`,
/// machine epsilon bounds, and `#![no_std]` transcendental operations via `libm`.
pub trait RealScalar: FieldScalar<Real = Self> + PartialOrd {
    /// Constant $2$.
    const TWO: Self;
    /// Constant $1/2$.
    const HALF: Self;
    /// Archimedes' constant $\pi$.
    const PI: Self;
    /// Full-turn constant $\tau = 2\pi$.
    const TAU: Self;
    /// Machine epsilon $\varepsilon_{\text{mach}}$.
    const EPSILON: Self;
    /// Smallest positive normal floating-point value.
    const MIN_POSITIVE: Self;
    /// Largest finite representable floating-point value.
    const MAX_VALUE: Self;
    /// Quiet `NaN`.
    const NAN: Self;
    /// Default relative pivot / convergence tolerance multiplier (`16.0`).
    const DEFAULT_TOL_SCALE: Self;

    /// Converts `self` to `f64`.
    fn to_f64(self) -> f64;

    /// Converts an unsigned index `n` into `Self`.
    fn from_usize(n: usize) -> Self;

    /// Principal square root $\sqrt{x}$.
    fn sqrt(self) -> Self;

    /// Returns a value with the magnitude of `self` and the sign bit of `sign`.
    fn copysign(self, sign: Self) -> Self;

    /// Overflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$.
    fn hypot(self, other: Self) -> Self;

    /// Minimum of `self` and `other`.
    fn min(self, other: Self) -> Self;

    /// Maximum of `self` and `other`.
    fn max(self, other: Self) -> Self;

    /// Clamps `self` to `[lo, hi]`.
    fn clamp_val(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Sine $\sin(x)$ in radians.
    fn sin(self) -> Self;

    /// Cosine $\cos(x)$ in radians.
    fn cos(self) -> Self;

    /// Simultaneous $(\sin(x), \cos(x))$ in radians.
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    /// Four-quadrant inverse tangent $\operatorname{atan2}(y, x)$ in radians.
    fn atan2(self, other: Self) -> Self;

    /// Inverse cosine $\arccos(x)$ in radians.
    fn acos(self) -> Self;

    /// Natural logarithm $\ln(x)$.
    fn ln(self) -> Self;

    /// Natural exponential $e^x$.
    fn exp(self) -> Self;

    /// Floating-point power $x^y$.
    fn powf(self, exp: Self) -> Self;

    /// Largest integer less than or equal to `self`.
    fn floor(self) -> Self;

    /// Smallest integer greater than or equal to `self`.
    fn ceil(self) -> Self;
}

impl FieldScalar for f64 {
    type Real = f64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn from_real(r: Self::Real) -> Self {
        r
    }
    #[inline]
    fn re_part(self) -> Self::Real {
        self
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
    #[inline]
    fn norm_sqr(self) -> Self::Real {
        self * self
    }
    #[inline]
    fn abs(self) -> Self::Real {
        f64::abs(self)
    }
    #[inline]
    fn scale_real(self, r: Self::Real) -> Self {
        self * r
    }
    #[inline]
    fn try_inv(self) -> Result<Self, ScalarError> {
        if !self.is_finite() || self == 0.0 {
            return Err(ScalarError::NonInvertible);
        }
        let inv = 1.0 / self;
        if inv.is_finite() {
            Ok(inv)
        } else {
            Err(ScalarError::NonInvertible)
        }
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

impl RealScalar for f64 {
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const TAU: Self = core::f64::consts::TAU;
    const EPSILON: Self = f64::EPSILON;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX_VALUE: Self = f64::MAX;
    const NAN: Self = f64::NAN;
    const DEFAULT_TOL_SCALE: Self = 16.0;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_usize(n: usize) -> Self {
        n as f64
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        if self.is_infinite() || other.is_infinite() {
            return f64::INFINITY;
        }
        if self.is_nan() || other.is_nan() {
            return f64::NAN;
        }
        let ax = self.abs();
        let ay = other.abs();
        let (hi, lo) = if ax >= ay { (ax, ay) } else { (ay, ax) };
        if hi == 0.0 {
            return 0.0;
        }
        let r = lo / hi;
        hi * libm::sqrt(1.0 + r * r)
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
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acos(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        libm::pow(self, exp)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
}

impl FieldScalar for f32 {
    type Real = f32;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn from_real(r: Self::Real) -> Self {
        r
    }
    #[inline]
    fn re_part(self) -> Self::Real {
        self
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
    #[inline]
    fn norm_sqr(self) -> Self::Real {
        self * self
    }
    #[inline]
    fn abs(self) -> Self::Real {
        f32::abs(self)
    }
    #[inline]
    fn scale_real(self, r: Self::Real) -> Self {
        self * r
    }
    #[inline]
    fn try_inv(self) -> Result<Self, ScalarError> {
        if !self.is_finite() || self == 0.0 {
            return Err(ScalarError::NonInvertible);
        }
        let inv = 1.0 / self;
        if inv.is_finite() {
            Ok(inv)
        } else {
            Err(ScalarError::NonInvertible)
        }
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl RealScalar for f32 {
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;
    const EPSILON: Self = f32::EPSILON;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX_VALUE: Self = f32::MAX;
    const NAN: Self = f32::NAN;
    const DEFAULT_TOL_SCALE: Self = 16.0;

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn from_usize(n: usize) -> Self {
        n as f32
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysignf(self, sign)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypotf(self, other)
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
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acosf(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        libm::powf(self, exp)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
}
