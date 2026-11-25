#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
//! Unordered [`FieldScalar`] and ordered [`RealScalar`] traits decoupling linear-algebraic
//! field operations (`f32`, `f64`, [`crate::Complex`]) from totally/partially ordered real math.

use crate::error::ScalarError;
use core::fmt::{Debug, Display};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Unordered `#![no_std]` field scalar contract supporting real and complex linear algebra.
///
/// Types implementing `FieldScalar` form an approximate or exact field with conjugation,
/// an associated ordered real magnitude type [`Self::Real`], and fallible inversion.
pub trait FieldScalar:
    Copy
    + Clone
    + PartialEq
    + Debug
    + Display
    + Default
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
    /// Ordered real magnitude type associated with `Self` (`Self` for reals, `T` for `Complex<T>`).
    type Real: RealScalar;

    /// Additive identity $0$.
    const ZERO: Self;
    /// Multiplicative identity $1$.
    const ONE: Self;

    /// Embeds an associated real scalar `val` along the real axis.
    #[must_use]
    fn from_real(val: Self::Real) -> Self;

    /// Converts an `f64` constant into `Self` along the real axis.
    #[must_use]
    fn from_f64(val: f64) -> Self;

    /// Converts a non-negative `usize` index or count into `Self`.
    #[must_use]
    fn from_usize(val: usize) -> Self;

    /// Returns the involutive conjugate $\overline{z}$ (`self` for real scalars).
    #[must_use]
    fn conj(self) -> Self;

    /// Returns the non-negative Euclidean modulus $|z| \in \mathbb{R}_{\ge 0}$.
    #[must_use]
    fn abs(self) -> Self::Real;

    /// Returns the squared modulus $|z|^2 = z \overline{z} \in \mathbb{R}_{\ge 0}$.
    #[must_use]
    fn norm_sqr(self) -> Self::Real;

    /// Returns `Some(self^-1)` when `self` is non-zero and finite, or `None` otherwise.
    #[must_use]
    fn checked_inv(self) -> Option<Self>;

    /// Returns `Ok(self^-1)` when invertible, or `Err(ScalarError::DivisionByZero)` otherwise.
    fn try_inverse(self) -> Result<Self, ScalarError> {
        self.checked_inv().ok_or(ScalarError::DivisionByZero)
    }

    /// Returns `true` when every component of `self` is finite (neither `NaN` nor infinite).
    #[must_use]
    fn is_finite(self) -> bool;

    /// Returns `true` when any component of `self` is `NaN`.
    #[must_use]
    fn is_nan(self) -> bool;

    /// Returns `true` when `self == Self::ZERO`.
    #[must_use]
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

/// Ordered `#![no_std]` real scalar contract extending [`FieldScalar`] with `PartialOrd`,
/// machine-epsilon constants, and `libm`-backed elementary functions.
pub trait RealScalar: FieldScalar<Real = Self> + PartialOrd {
    /// Constant $2$.
    const TWO: Self;
    /// Constant $1/2$.
    const HALF: Self;
    /// Archimedes' constant $\pi$.
    const PI: Self;
    /// Full-turn constant $\tau = 2\pi$.
    const TAU: Self;
    /// Machine epsilon $\varepsilon_{\mathrm{mach}}$.
    const EPSILON: Self;
    /// Smallest positive normal value (`MIN_POSITIVE`).
    const MIN_POSITIVE: Self;
    /// Largest finite representable value (`MAX`).
    const MAX_VALUE: Self;
    /// Quiet `NaN` value.
    const NAN: Self;
    /// Default numerical rank/singularity tolerance multiplier (`16.0`).
    const DEFAULT_TOL_SCALE: Self;

    /// Widens or converts `self` to `f64`.
    #[must_use]
    fn to_f64(self) -> f64;

    /// Principal square root $\sqrt{x}$.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Returns a value with the magnitude of `self` and the sign bit of `sign`.
    #[must_use]
    fn copysign(self, sign: Self) -> Self;

    /// Overflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$.
    #[must_use]
    fn hypot(self, other: Self) -> Self;

    /// IEEE-754 minimum of `self` and `other`.
    #[must_use]
    fn min(self, other: Self) -> Self;

    /// IEEE-754 maximum of `self` and `other`.
    #[must_use]
    fn max(self, other: Self) -> Self;

    /// Natural logarithm $\ln(x)$.
    #[must_use]
    fn ln(self) -> Self;

    /// Natural exponential $e^x$.
    #[must_use]
    fn exp(self) -> Self;

    /// Sine $\sin(x)$ in radians.
    #[must_use]
    fn sin(self) -> Self;

    /// Cosine $\cos(x)$ in radians.
    #[must_use]
    fn cos(self) -> Self;

    /// Four-quadrant arctangent $\operatorname{atan2}(y, x)$ in $[-\pi, \pi]$.
    #[must_use]
    fn atan2(self, x: Self) -> Self;

    /// Largest integer less than or equal to `self`.
    #[must_use]
    fn floor(self) -> Self;

    /// Smallest integer greater than or equal to `self`.
    #[must_use]
    fn ceil(self) -> Self;

    /// Nearest integer, rounding half-way cases away from zero.
    #[must_use]
    fn round(self) -> Self;

    /// Real power $x^y$.
    #[must_use]
    fn powf(self, exp: Self) -> Self;

    /// Integer power $x^n$ via binary exponentiation.
    #[must_use]
    fn powi(self, exp: i32) -> Self;
}

fn powi_real<T: RealScalar>(mut base: T, exp: i32) -> T {
    if exp == 0 {
        return T::ONE;
    }
    let mut n: u32 = exp.unsigned_abs();
    if exp < 0 {
        base = T::ONE / base;
    }
    let mut acc = T::ONE;
    while n > 0 {
        if (n & 1) == 1 {
            acc *= base;
        }
        n >>= 1;
        if n > 0 {
            base *= base;
        }
    }
    acc
}

impl FieldScalar for f32 {
    type Real = f32;

    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_real(val: f32) -> Self {
        val
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        val as f32
    }

    #[inline]
    fn from_usize(val: usize) -> Self {
        val as f32
    }

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn abs(self) -> f32 {
        libm::fabsf(self)
    }

    #[inline]
    fn norm_sqr(self) -> f32 {
        self * self
    }

    #[inline]
    fn checked_inv(self) -> Option<Self> {
        if self == 0.0 || !self.is_finite() {
            return None;
        }
        let inv = 1.0_f32 / self;
        inv.is_finite().then_some(inv)
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
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::expf(self)
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
    fn atan2(self, x: Self) -> Self {
        libm::atan2f(self, x)
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
    fn powf(self, exp: Self) -> Self {
        libm::powf(self, exp)
    }
    #[inline]
    fn powi(self, exp: i32) -> Self {
        powi_real(self, exp)
    }
}

impl FieldScalar for f64 {
    type Real = f64;

    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_real(val: f64) -> Self {
        val
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        val
    }

    #[inline]
    fn from_usize(val: usize) -> Self {
        val as f64
    }

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn abs(self) -> f64 {
        libm::fabs(self)
    }

    #[inline]
    fn norm_sqr(self) -> f64 {
        self * self
    }

    #[inline]
    fn checked_inv(self) -> Option<Self> {
        if self == 0.0 || !self.is_finite() {
            return None;
        }
        let inv = 1.0_f64 / self;
        inv.is_finite().then_some(inv)
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
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysign(self, sign)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        if self.is_infinite() || other.is_infinite() {
            return f64::INFINITY;
        }
        if self.is_nan() || other.is_nan() {
            return f64::NAN;
        }
        let ax = libm::fabs(self);
        let ay = libm::fabs(other);
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
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
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
    fn atan2(self, x: Self) -> Self {
        libm::atan2(self, x)
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
    fn powf(self, exp: Self) -> Self {
        libm::pow(self, exp)
    }
    #[inline]
    fn powi(self, exp: i32) -> Self {
        powi_real(self, exp)
    }
}
