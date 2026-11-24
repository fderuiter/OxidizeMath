#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Pure `#![no_std]` IEEE-754 floating-point scalar abstraction (`f32`, `f64`).

use core::fmt::{Debug, Display};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Pure `core` IEEE-754 square root for `f64` using exponent halving and five Heron steps.
/// `[cite:engine_linalg]`
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if !x.is_finite() {
        return x;
    }
    if x < f64::MIN_POSITIVE {
        // Scale subnormal significand by 2^52 into normal range, then scale root by 2^-26.
        const SCALE_UP: f64 = 4_503_599_627_370_496.0;
        const SCALE_DOWN: f64 = 1.0 / 67_108_864.0;
        return sqrt_f64_normal(x * SCALE_UP) * SCALE_DOWN;
    }
    sqrt_f64_normal(x)
}

#[inline]
fn sqrt_f64_normal(x: f64) -> f64 {
    let mut g = f64::from_bits((x.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    0.5 * (g + x / g)
}

/// Pure `core` overflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$ for `f64`.
/// `[cite:engine_linalg]`
#[inline]
pub fn hypot_f64(x: f64, y: f64) -> f64 {
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    let ay = y.abs();
    let (hi, lo) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    if hi == 0.0 {
        return 0.0;
    }
    let r = lo / hi;
    hi * sqrt_f64(1.0 + r * r)
}

/// Real IEEE-754 floating-point scalar (`f32` or `f64`) usable in `#![no_std]` linear solvers.
pub trait Scalar:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Debug
    + Display
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
    /// Additive identity `0`.
    const ZERO: Self;
    /// Multiplicative identity `1`.
    const ONE: Self;
    /// Constant `2`.
    const TWO: Self;
    /// Constant `0.5`.
    const HALF: Self;
    /// Machine epsilon (`f32::EPSILON` or `f64::EPSILON`).
    const EPSILON: Self;
    /// Smallest positive normal value (`MIN_POSITIVE`).
    const MIN_POSITIVE: Self;
    /// Largest finite value (`MAX`).
    const MAX_VALUE: Self;
    /// Quiet `NaN`.
    const NAN: Self;
    /// Default relative pivot / rank tolerance multiplier.
    const DEFAULT_TOL_SCALE: Self;

    /// Converts an `f64` literal or threshold into `Self`.
    fn from_f64(v: f64) -> Self;
    /// Converts a `usize` count or dimension into `Self`.
    fn from_usize(v: usize) -> Self;
    /// Widens `Self` into `f64`.
    fn to_f64(self) -> f64;
    /// Absolute value $|x|$.
    fn abs(self) -> Self;
    /// Non-negative square root $\sqrt{x}$ in pure `core`.
    fn sqrt(self) -> Self;
    /// Magnitude of `self` with the sign of `sign`.
    fn copysign(self, sign: Self) -> Self;
    /// Overflow-safe $\sqrt{\text{self}^2 + \text{other}^2}$.
    fn hypot(self, other: Self) -> Self;
    /// Maximum of two finite scalars.
    fn max(self, other: Self) -> Self;
    /// Minimum of two finite scalars.
    fn min(self, other: Self) -> Self;
    /// Returns `true` if neither infinite nor `NaN`.
    fn is_finite(self) -> bool;
    /// Returns `true` if `self` is `NaN`.
    fn is_nan(self) -> bool;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const EPSILON: Self = f64::EPSILON;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX_VALUE: Self = f64::MAX;
    const NAN: Self = f64::NAN;
    const DEFAULT_TOL_SCALE: Self = 16.0;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as f64
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        sqrt_f64(self)
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        hypot_f64(self, other)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
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

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const EPSILON: Self = f32::EPSILON;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX_VALUE: Self = f32::MAX;
    const NAN: Self = f32::NAN;
    const DEFAULT_TOL_SCALE: Self = 16.0;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        sqrt_f64(f64::from(self)) as f32
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f32::copysign(self, sign)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        hypot_f64(f64::from(self), f64::from(other)) as f32
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
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
