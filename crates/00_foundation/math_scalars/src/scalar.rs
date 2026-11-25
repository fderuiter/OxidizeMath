//! `#![no_std]` floating-point [`Scalar`] contract backed by `libm`.
// [cite:math_scalars]

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// `#![no_std]` Tier 0 real floating-point scalar contract for deterministic ODE, SDE, PDE,
/// SPDE, and dimensional quantity kernels.
///
/// Implemented for `f64` and `f32` via `libm` so transcendental evaluations are available in bare
/// `#![no_std]` builds and deterministic across native and `wasm32-unknown-unknown` targets.
pub trait Scalar:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Debug
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
    /// Full-turn constant $\tau = 2\pi$.
    const TAU: Self;
    /// Machine epsilon for this floating-point type.
    const EPSILON: Self;

    /// Converts an `f64` literal or parameter into `Self`.
    fn from_f64(value: f64) -> Self;
    /// Converts `Self` into an `f64` value.
    fn to_f64(self) -> f64;
    /// Converts a `usize` count or grid dimension into `Self`.
    fn from_usize(value: usize) -> Self;
    /// Returns `true` if `self` is neither infinite nor `NaN`.
    fn is_finite(self) -> bool;
    /// Returns the absolute value $|x|$.
    fn abs(self) -> Self;
    /// Returns the principal square root $\sqrt{x}$.
    fn sqrt(self) -> Self;
    /// Returns the natural logarithm $\ln(x)$.
    fn ln(self) -> Self;
    /// Returns the natural exponential $e^x$.
    fn exp(self) -> Self;
    /// Returns the sine $\sin(x)$.
    fn sin(self) -> Self;
    /// Returns the cosine $\cos(x)$.
    fn cos(self) -> Self;
    /// Raises `self` to a real floating-point power `exp` ($x^{\text{exp}}$).
    fn powf(self, exp: Self) -> Self;
    /// Returns the minimum of `self` and `other` following IEEE-754 `fmin` semantics.
    fn min(self, other: Self) -> Self;
    /// Returns the maximum of `self` and `other` following IEEE-754 `fmax` semantics.
    fn max(self, other: Self) -> Self;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const TAU: Self = core::f64::consts::TAU;
    const EPSILON: Self = f64::EPSILON;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as f64
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline(always)]
    fn powf(self, exp: Self) -> Self {
        libm::pow(self, exp)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        libm::fmin(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        libm::fmax(self, other)
    }
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;
    const EPSILON: Self = f32::EPSILON;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as f32
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline(always)]
    fn powf(self, exp: Self) -> Self {
        libm::powf(self, exp)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        libm::fminf(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        libm::fmaxf(self, other)
    }
}
