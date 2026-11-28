// [cite:math_scalars]
//! Unified `#![no_std]` algebraic and numerical scalar trait hierarchy.

use core::cmp::Ordering;
use core::fmt::{Debug, Display};
use core::iter::Sum;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// Borrowed-reference ring arithmetic enabling zero-copy evaluation over non-`Copy` scalars
/// (such as arbitrary-precision integers, rationals, floats, and polynomials).
pub trait RefOps<Output = Self>: Sized {
    /// Computes `self + rhs` by shared reference without consuming either operand.
    fn ref_add(&self, rhs: &Self) -> Output;

    /// Computes `self - rhs` by shared reference without consuming either operand.
    fn ref_sub(&self, rhs: &Self) -> Output;

    /// Computes `self * rhs` by shared reference without consuming either operand.
    fn ref_mul(&self, rhs: &Self) -> Output;

    /// Computes `-self` by shared reference without consuming the operand.
    fn ref_neg(&self) -> Output;
}

/// Base value contract for mathematical scalars across exact, complex, dual, and real domains.
///
/// Deliberately omits `Copy` and `PartialOrd` so heap-backed arbitrary-precision types
/// (`dashu`), polynomials, complex numbers, and dual numbers participate without wrappers.
pub trait Scalar: Sized + Clone + PartialEq + Debug + Send + Sync + 'static {}

impl<T: Sized + Clone + PartialEq + Debug + Send + Sync + 'static> Scalar for T {}

/// Unital associative ring $(R, +, \cdot, 0, 1)$ supporting owned and borrowed arithmetic.
pub trait Ring:
    Scalar
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// Returns the additive identity $0_R$.
    fn zero() -> Self;

    /// Returns the multiplicative identity $1_R$.
    fn one() -> Self;

    /// Returns `true` if `self` equals the additive identity $0_R$.
    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    /// Borrowed addition hook backing [`RefOps::ref_add`] (override on heap-backed rings to avoid cloning).
    #[inline]
    fn ring_ref_add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.clone()
    }

    /// Borrowed subtraction hook backing [`RefOps::ref_sub`] (override on heap-backed rings to avoid cloning).
    #[inline]
    fn ring_ref_sub(&self, rhs: &Self) -> Self {
        self.clone() - rhs.clone()
    }

    /// Borrowed multiplication hook backing [`RefOps::ref_mul`] (override on heap-backed rings to avoid cloning).
    #[inline]
    fn ring_ref_mul(&self, rhs: &Self) -> Self {
        self.clone() * rhs.clone()
    }

    /// Borrowed negation hook backing [`RefOps::ref_neg`] (override on heap-backed rings to avoid cloning).
    #[inline]
    fn ring_ref_neg(&self) -> Self {
        -self.clone()
    }
}

impl<T: Ring> RefOps<T> for T {
    #[inline]
    fn ref_add(&self, rhs: &Self) -> T {
        Ring::ring_ref_add(self, rhs)
    }

    #[inline]
    fn ref_sub(&self, rhs: &Self) -> T {
        Ring::ring_ref_sub(self, rhs)
    }

    #[inline]
    fn ref_mul(&self, rhs: &Self) -> T {
        Ring::ring_ref_mul(self, rhs)
    }

    #[inline]
    fn ref_neg(&self) -> T {
        Ring::ring_ref_neg(self)
    }
}

/// Commutative division ring $(F, +, \cdot, 0, 1)$ where every nonzero element is invertible.
pub trait Field: Ring + Div<Output = Self> {
    /// Computes `self / rhs` by shared reference.
    fn ref_div(&self, rhs: &Self) -> Self;

    /// Returns the multiplicative inverse $x^{-1}$, or `None` when `self.is_zero()`.
    fn try_inv(&self) -> Option<Self>;

    /// Returns the multiplicative inverse $x^{-1}$.
    ///
    /// # Panics
    /// Panics if `self.is_zero()`; prefer [`Self::try_inv`] in fallible contexts.
    #[inline]
    fn inv(&self) -> Self {
        self.try_inv()
            .expect("Field::inv called on non-invertible zero element")
    }

    /// Alias for [`Self::inv`] preserving compatibility with abstract-algebra callers.
    #[inline]
    fn multiplicative_inverse(&self) -> Self {
        self.inv()
    }
}

/// Analytic scalar domain closed under elementary transcendental functions and norm extraction.
///
/// Implemented for `f32`, `f64`, [`crate::Complex<T>`], and [`crate::Dual<T>`] without requiring
/// `Copy` or `PartialOrd`.
pub trait ComplexField: Field + DivAssign {
    /// Underlying real magnitude and coordinate field.
    type Real: RealField;

    /// Embeds a real value $x \in \mathbb{R}$ into `Self` ($x + 0i$ or $x + 0\varepsilon$).
    fn from_real(re: Self::Real) -> Self;

    /// Embeds an `f64` literal into `Self` via [`Self::from_real`] and [`ComplexField::from_f64`].
    #[inline]
    fn from_f64(val: f64) -> Self {
        Self::from_real(<Self::Real as ComplexField>::from_f64(val))
    }

    /// Returns the imaginary unit $i$ for complex fields, or `Self::nan()` for purely real/dual domains.
    fn imag_unit() -> Self;

    /// Returns a quiet `NaN` sentinel representing an invalid or out-of-domain evaluation.
    fn nan() -> Self;

    /// Extracts the real part $\operatorname{Re}(z)$.
    fn real(&self) -> Self::Real;

    /// Extracts the imaginary part $\operatorname{Im}(z)$ (`0` for real and dual scalars).
    fn imag(&self) -> Self::Real;

    /// Returns the complex conjugate $\bar{z}$ (identity on real and dual scalars).
    fn conj(&self) -> Self;

    /// Returns the squared modulus $|z|^2 = \operatorname{Re}(z)^2 + \operatorname{Im}(z)^2$.
    fn norm_sqr(&self) -> Self::Real;

    /// Returns the overflow-safe Euclidean modulus $|z|$ in [`Self::Real`].
    fn modulus(&self) -> Self::Real;

    /// Scales `self` by a real factor $s \in \mathbb{R}$.
    fn scale_real(&self, factor: Self::Real) -> Self;

    /// Returns `true` if all components of `self` are finite (`!is_nan` and `!is_infinite`).
    fn is_finite(&self) -> bool;

    /// Returns `true` if any component of `self` is `NaN`.
    fn is_nan(&self) -> bool;

    /// Returns $|z|$ embedded back into `Self`.
    fn abs(self) -> Self;

    /// Principal square root $\sqrt{z}$.
    fn sqrt(self) -> Self;

    /// Natural exponential $e^z$.
    fn exp(self) -> Self;

    /// Principal natural logarithm $\ln(z)$.
    fn ln(self) -> Self;

    /// Principal base-10 logarithm $\log_{10}(z)$.
    fn log10(self) -> Self;

    /// Sine $\sin(z)$.
    fn sin(self) -> Self;

    /// Cosine $\cos(z)$.
    fn cos(self) -> Self;

    /// Simultaneous $(\sin z, \cos z)$.
    fn sin_cos(self) -> (Self, Self);

    /// Tangent $\tan(z)$.
    fn tan(self) -> Self;

    /// Hyperbolic sine $\sinh(z)$.
    fn sinh(self) -> Self;

    /// Hyperbolic cosine $\cosh(z)$.
    fn cosh(self) -> Self;

    /// Hyperbolic tangent $\tanh(z)$.
    fn tanh(self) -> Self;

    /// Principal inverse sine $\arcsin(z)$.
    fn asin(self) -> Self;

    /// Principal inverse cosine $\arccos(z)$.
    fn acos(self) -> Self;

    /// Principal inverse tangent $\arctan(z)$.
    fn atan(self) -> Self;

    /// Raises `self` to a real exponent $p \in \mathbb{R}$.
    fn powf(self, exp: Self::Real) -> Self;

    /// Raises `self` to a scalar exponent $w \in \text{Self}$.
    fn powc(self, exp: Self) -> Self;

    /// Raises `self` to a signed integer power $n \in \mathbb{Z}$.
    fn powi(self, n: i32) -> Self;
}

/// `#![no_std]` IEEE-754 real floating-point field contract unifying linear algebra, differential
/// equations, continuum PDEs, curve fitting, and interval arithmetic.
///
/// Backed by `libm` on both host and `wasm32-unknown-unknown` targets for bit-identical determinism
/// and `0`-ULP square-root accuracy.
pub trait RealField:
    ComplexField<Real = Self>
    + Copy
    + bytemuck::Pod
    + bytemuck::Zeroable
    + PartialOrd
    + Default
    + Display
    + Sum
    + Rem<Output = Self>
    + RemAssign
{
    /// Additive identity `0.0`.
    const ZERO: Self;
    /// Multiplicative identity `1.0`.
    const ONE: Self;
    /// Constant `2.0`.
    const TWO: Self;
    /// Constant `0.5`.
    const HALF: Self;
    /// Archimedes' constant $\pi$.
    const PI: Self;
    /// Full-turn constant $\tau = 2\pi$.
    const TAU: Self;
    /// Euler's number $e$.
    const E: Self;
    /// Machine epsilon (`1.0 + EPSILON != 1.0`).
    const EPSILON: Self;
    /// Smallest positive normal value.
    const MIN_POSITIVE: Self;
    /// Largest finite representable value.
    const MAX_VALUE: Self;
    /// Positive IEEE-754 infinity.
    const INFINITY: Self;
    /// Negative IEEE-754 infinity.
    const NEG_INFINITY: Self;
    /// Quiet IEEE-754 `NaN`.
    const NAN: Self;
    /// Default relative rank/pivot tolerance multiplier above `EPSILON`.
    const DEFAULT_TOL_SCALE: Self;

    /// Converts `Self` into an `f64` value.
    fn to_f64(self) -> f64;

    /// Converts a `usize` count or grid index into `Self`.
    fn from_usize(value: usize) -> Self;

    /// IEEE-754 minimum (`fmin` semantics propagating non-`NaN` operands).
    fn min(self, other: Self) -> Self;

    /// IEEE-754 maximum (`fmax` semantics propagating non-`NaN` operands).
    fn max(self, other: Self) -> Self;

    /// Clamps `self` to `[min, max]`.
    fn clamp(self, min: Self, max: Self) -> Self;

    /// Returns a value with the magnitude of `self` and the sign bit of `sign`.
    fn copysign(self, sign: Self) -> Self;

    /// Overflow- and underflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$.
    fn hypot(self, other: Self) -> Self;

    /// Cancellation-resistant difference of products $a b - c d$.
    #[inline]
    fn diff_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        a * b - c * d
    }

    /// Four-quadrant inverse tangent $\operatorname{atan2}(y, x) \in [-\pi, \pi]$.
    fn atan2(self, other: Self) -> Self;

    /// Largest integer less than or equal to `self`.
    fn floor(self) -> Self;

    /// Smallest integer greater than or equal to `self`.
    fn ceil(self) -> Self;

    /// Nearest integer, rounding half-way cases away from zero.
    fn round(self) -> Self;

    /// Fractional part `self - trunc(self)`.
    fn fract(self) -> Self;

    /// Accurate $e^x - 1$ for small $|x|$.
    fn expm1(self) -> Self;

    /// Base-2 logarithm $\log_2(x)$.
    fn log2(self) -> Self;

    /// Non-negative Euclidean remainder in `[0, |rhs|)`.
    fn rem_euclid(self, rhs: Self) -> Self;

    /// Smallest representable floating-point value strictly greater than `self` (directed outward rounding).
    fn next_up(self) -> Self;

    /// Largest representable floating-point value strictly less than `self` (directed outward rounding).
    fn next_down(self) -> Self;

    /// IEEE-754 total ordering comparison (`-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN`).
    fn total_cmp(&self, other: &Self) -> Ordering;

    /// Converts an `f64` value in `[0, 1]` into a strictly open `(0, 1)` scalar safe for `ln`/`probit`.
    #[inline]
    fn open_unit_from_f64(value: f64) -> Self {
        const INV_2POW53: f64 = 1.0 / ((1_u64 << 53) as f64);
        const F64_MIN_ABOVE_ZERO: f64 = 1.0 / ((1_u64 << 54) as f64);
        const F64_MAX_BELOW_ONE: f64 = 1.0 - INV_2POW53;
        <Self as ComplexField>::from_f64(value.clamp(F64_MIN_ABOVE_ZERO, F64_MAX_BELOW_ONE))
    }

    /// Converts a raw 64-bit PRNG word into a uniform sample in $[0, 1)$.
    #[inline]
    fn uniform_closed_open_from_u64(u64_word: u64) -> Self {
        const INV_2POW53: f64 = 1.0 / ((1u64 << 53) as f64);
        <Self as ComplexField>::from_f64(((u64_word >> 11) as f64) * INV_2POW53)
    }

    /// Converts a raw 64-bit PRNG word into a strictly open sample in $(0, 1)$.
    #[inline]
    fn uniform_open_from_u64(u64_word: u64) -> Self {
        const INV_2POW53: f64 = 1.0 / ((1u64 << 53) as f64);
        Self::open_unit_from_f64((((u64_word >> 11) as f64) + 0.5) * INV_2POW53)
    }
}
