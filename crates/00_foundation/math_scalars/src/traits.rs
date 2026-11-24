// [cite:math_scalars]
//! Core trait hierarchy separating general differentiable/enclosure [`Scalar`] capabilities
//! from ordered IEEE-754 floating-point [`RealScalar`] capabilities.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// General algebraic and deterministic `libm` elementary scalar trait.
///
/// Implemented uniformly for primitive IEEE-754 reals (`f64`, `f32`), forward-mode
/// automatic differentiation types (`Dual<T>`, `DualVec<T, N>`, `HyperDual<T>`),
/// outward-rounded intervals (`Interval<T>`), and midpoint-radius balls (`Ball<T>`).
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
    /// Constant $\pi$.
    const PI: Self;

    /// Embeds a constant `f64` literal into `Self` (with zero tangent / rigorous point enclosure).
    fn from_f64(v: f64) -> Self;

    /// Extracts the primal / midpoint `f64` component of `Self`.
    fn real_part(self) -> f64;

    /// Returns `true` when all scalar/tangent/bound components are finite.
    fn is_finite_val(self) -> bool;

    /// Magnitude $|x|$.
    fn abs(self) -> Self;

    /// Algebraic square $x^2$ (exploits non-negativity on `Interval<T>`).
    #[inline]
    fn sqr(self) -> Self {
        self * self
    }

    /// Multiplicative reciprocal $1 / x$.
    #[inline]
    fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Principal square root $\sqrt{x}$.
    fn sqrt(self) -> Self;
    /// Integer power $x^n$ via binary exponentiation.
    fn powi(self, n: i32) -> Self;
    /// Real power $x^y$.
    fn powf(self, exp: Self) -> Self;
    /// Natural exponential $e^x$.
    fn exp(self) -> Self;
    /// Natural logarithm $\ln(x)$.
    fn ln(self) -> Self;
    /// Base-2 logarithm $\log_2(x)$.
    fn log2(self) -> Self;
    /// Base-10 logarithm $\log_{10}(x)$.
    fn log10(self) -> Self;
    /// Sine $\sin(x)$.
    fn sin(self) -> Self;
    /// Cosine $\cos(x)$.
    fn cos(self) -> Self;
    /// Simultaneous $(\sin x, \cos x)$.
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
    /// Tangent $\tan(x)$.
    fn tan(self) -> Self;
    /// Principal arcsine $\arcsin(x)$.
    fn asin(self) -> Self;
    /// Principal arccosine $\arccos(x)$.
    fn acos(self) -> Self;
    /// Principal arctangent $\arctan(x)$.
    fn atan(self) -> Self;
    /// Two-argument four-quadrant arctangent $\operatorname{atan2}(y, x)$.
    fn atan2(self, other: Self) -> Self;
    /// Hyperbolic sine $\sinh(x)$.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine $\cosh(x)$.
    fn cosh(self) -> Self;
    /// Hyperbolic tangent $\tanh(x)$.
    fn tanh(self) -> Self;
    /// Inverse hyperbolic sine $\operatorname{asinh}(x)$.
    fn asinh(self) -> Self;
    /// Inverse hyperbolic cosine $\operatorname{acosh}(x)$.
    fn acosh(self) -> Self;
    /// Inverse hyperbolic tangent $\operatorname{atanh}(x)$.
    fn atanh(self) -> Self;
    /// Euclidean hypotenuse $\sqrt{x^2 + y^2}$.
    fn hypot(self, other: Self) -> Self;
}

/// Ordered IEEE-754 floating-point scalar supporting bit-level directed ULP stepping.
///
/// Intentionally separated from [`Scalar`] because intervals, balls, and dual numbers do not
/// form a total order compatible with branch-driven numerical comparisons.
pub trait RealScalar: Scalar + PartialOrd {
    /// Machine epsilon ($2^{-52}$ for `f64`, $2^{-23}$ for `f32`).
    const EPSILON: Self;
    /// Smallest positive normal number.
    const MIN_POSITIVE: Self;
    /// Largest finite representable value.
    const MAX_VAL: Self;
    /// Positive infinity $+\infty$.
    const INFINITY: Self;
    /// Negative infinity $-\infty$.
    const NEG_INFINITY: Self;
    /// Quiet `NaN`.
    const NAN: Self;
    /// Full circle constant $2\pi$.
    const TAU: Self;
    /// Quarter circle constant $\pi/2$.
    const FRAC_PI_2: Self;

    /// Widens `self` losslessly into `f64`.
    fn to_f64(self) -> f64;
    /// Returns `true` if `self` is `NaN`.
    fn is_nan(self) -> bool;
    /// Returns `true` if `self` is neither infinite nor `NaN`.
    fn is_finite(self) -> bool;
    /// Returns `true` if `self` is $+\infty$ or $-\infty$.
    fn is_infinite(self) -> bool;
    /// Returns `true` if the IEEE-754 sign bit of `self` is set.
    fn is_sign_negative(self) -> bool;
    /// IEEE-754 minimum (propagates non-`NaN` per Rust `min`).
    fn min_val(self, other: Self) -> Self;
    /// IEEE-754 maximum (propagates non-`NaN` per Rust `max`).
    fn max_val(self, other: Self) -> Self;
    /// Magnitude of `self` with sign bit of `sign`.
    fn copysign(self, sign: Self) -> Self;
    /// Largest integer $\le \text{self}$.
    fn floor(self) -> Self;
    /// Smallest integer $\ge \text{self}$.
    fn ceil(self) -> Self;
    /// Nearest integer, rounding half-way cases away from zero.
    fn round(self) -> Self;
    /// Integer part truncated toward zero.
    fn trunc(self) -> Self;
    /// Deterministic software fused multiply-add `self * mul + add`.
    fn fma(self, mul: Self, add: Self) -> Self;
    /// Smallest IEEE-754 representable value strictly greater than `self`.
    fn next_up(self) -> Self;
    /// Largest IEEE-754 representable value strictly less than `self`.
    fn next_down(self) -> Self;
    /// Steps `self` upward by `ulps` representable increments.
    fn step_up(self, ulps: u32) -> Self;
    /// Steps `self` downward by `ulps` representable increments.
    fn step_down(self, ulps: u32) -> Self;
    /// Ordered IEEE-754 ULP distance between `self` and `other`.
    fn ulp_distance(self, other: Self) -> u64;
}

/// Binary exponentiation helper for integer powers across any [`Scalar`].
#[inline]
pub(crate) fn powi_generic<S: Scalar>(base: S, n: i32) -> S {
    if n == 0 {
        return S::ONE;
    }
    let mut exp = n.unsigned_abs();
    let mut acc = S::ONE;
    let mut cur = if n < 0 { base.recip() } else { base };
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= cur;
        }
        exp >>= 1;
        if exp > 0 {
            cur = cur.sqr();
        }
    }
    acc
}
