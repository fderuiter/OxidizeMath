// [cite:math_commons]
use crate::error::ScalarDomainError;
use core::fmt;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// Base value contract shared across real, complex, dual, integer, modular, and arbitrary-precision scalars.
///
/// Deliberately omits `Copy`, `PartialOrd`, and `to_f64` so heap-backed (`dashu::IBig`, `RBig`, `FBig`)
/// and unordered (`Complex<T>`) algebraic types participate on equal footing with `f32` and `f64`.
pub trait Scalar: Clone + PartialEq + fmt::Debug + Send + Sync + 'static {}

/// Associative unital ring $(R, +, \cdot, 0, 1)$ with additive inverses.
pub trait Ring:
    Scalar
    + Sized
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

    /// Returns `true` when `self` equals [`Self::zero`].
    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    /// Raises `self` to unsigned integer power `exp` via binary exponentiation in $O(\log \text{exp})$ multiplications.
    #[inline]
    fn pow_u64(&self, mut exp: u64) -> Self {
        let mut result = Self::one();
        let mut base = self.clone();
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base.clone();
            }
            exp >>= 1;
            if exp > 0 {
                let next = base.clone() * base;
                base = next;
            }
        }
        result
    }
}

/// Commutative division ring (field) $(F, +, \cdot, 0, 1)$ where every nonzero element has an inverse.
pub trait Field: Ring + Div<Output = Self> {
    /// Computes the multiplicative inverse $x^{-1}$, returning `None` when `self` is zero or non-invertible.
    fn checked_inverse(&self) -> Option<Self>;

    /// Fallibly computes $x^{-1}$, returning [`ScalarDomainError::DivisionByZero`] when not invertible.
    #[inline]
    fn try_inverse(&self) -> Result<Self, ScalarDomainError> {
        self.checked_inverse()
            .ok_or(ScalarDomainError::DivisionByZero)
    }

    /// Fallibly divides `self` by `rhs` via [`Self::checked_inverse`].
    #[inline]
    fn checked_div(&self, rhs: &Self) -> Option<Self> {
        rhs.checked_inverse().map(|inv| self.clone() * inv)
    }

    /// Returns the multiplicative inverse $x^{-1}$ (panicking on exact zero in integer/rational fields or returning `1/0` in IEEE-754).
    fn multiplicative_inverse(&self) -> Self;
}

/// Analytical scalar field closed under complex-analytic elementary functions (`f32`, `f64`, `Complex<T>`, `Dual<T>`).
pub trait ComplexField: Field + Copy + DivAssign {
    /// Underlying ordered real coordinate field (`Self` for real/dual scalars, `T` for `Complex<T>`).
    type Real: RealField;

    /// Embeds a real scalar `re` on the real axis (`re + 0i`).
    fn from_real(re: Self::Real) -> Self;

    /// Converts an `f64` constant into `Self` along the real axis.
    #[inline(always)]
    fn from_real_f64(val: f64) -> Self {
        Self::from_real(Self::Real::from_f64(val))
    }

    /// Returns a quiet IEEE-754 `NaN` sentinel.
    fn nan() -> Self;

    /// Returns the imaginary unit $i$ (`Complex::new(0, 1)` for complex fields, `Self::nan()` for purely real fields).
    fn imag_unit() -> Self;

    /// Extracts the real part $\operatorname{Re}(z)$.
    fn real(self) -> Self::Real;

    /// Extracts the imaginary part $\operatorname{Im}(z)$ (`0` for purely real fields).
    fn imag(self) -> Self::Real;

    /// Returns the complex conjugate $\bar{z}$.
    fn conj(self) -> Self;

    /// Returns the squared modulus $|z|^2 = \operatorname{Re}(z)^2 + \operatorname{Im}(z)^2$.
    fn norm_sqr(self) -> Self::Real;

    /// Returns the Euclidean modulus $|z|$ using overflow-safe hypotenuse evaluation.
    fn abs(self) -> Self::Real;

    /// Returns the principal argument $\arg(z) \in (-\pi, \pi]$.
    fn arg(self) -> Self::Real;

    /// Multiplies `self` by real scalar `factor`.
    fn scale(self, factor: Self::Real) -> Self;

    /// Returns `true` if all components of `self` are finite.
    fn is_all_finite(self) -> bool;

    /// Returns `true` if any component of `self` is `NaN`.
    fn is_any_nan(self) -> bool;

    /// Principal-branch square root $\sqrt{z}$ over real or complex scalars.
    fn csqrt(self) -> Self;
    /// Real or principal cube root $z^{1/3}$.
    fn cbrt(self) -> Self;
    /// Base-$e$ exponential $e^z$.
    fn exp(self) -> Self;
    /// Principal natural logarithm $\ln(z)$.
    fn ln(self) -> Self;
    /// Base-2 logarithm $\log_2(z)$.
    fn log2(self) -> Self;
    /// Base-10 logarithm $\log_{10}(z)$.
    fn log10(self) -> Self;
    /// Sine $\sin(z)$.
    fn sin(self) -> Self;
    /// Cosine $\cos(z)$.
    fn cos(self) -> Self;
    /// Simultaneous $(\sin z, \cos z)$.
    fn sin_cos(self) -> (Self, Self);
    /// Tangent $\tan(z)$.
    fn tan(self) -> Self;
    /// Principal inverse sine $\arcsin(z)$.
    fn asin(self) -> Self;
    /// Principal inverse cosine $\arccos(z)$.
    fn acos(self) -> Self;
    /// Principal inverse tangent $\arctan(z)$.
    fn atan(self) -> Self;
    /// Hyperbolic sine $\sinh(z)$.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine $\cosh(z)$.
    fn cosh(self) -> Self;
    /// Hyperbolic tangent $\tanh(z)$.
    fn tanh(self) -> Self;
    /// Inverse hyperbolic sine $\operatorname{arsinh}(z)$.
    fn asinh(self) -> Self;
    /// Inverse hyperbolic cosine $\operatorname{arcosh}(z)$.
    fn acosh(self) -> Self;
    /// Inverse hyperbolic tangent $\operatorname{artanh}(z)$.
    fn atanh(self) -> Self;
    /// Raises `self` to real exponent `exp`.
    fn powf(self, exp: Self::Real) -> Self;
    /// Raises `self` to scalar/complex exponent `exp`.
    fn powc(self, exp: Self) -> Self;
    /// Raises `self` to signed 32-bit integer exponent `exp`.
    fn powi(self, exp: i32) -> Self;
    /// Componentwise floor $\lfloor z \rfloor$ (or real floor).
    fn floor(self) -> Self;
    /// Componentwise ceiling $\lceil z \rceil$ (or real ceiling).
    fn ceil(self) -> Self;
    /// Componentwise nearest-integer rounding.
    fn round(self) -> Self;
    /// Unit-modulus signum ($z / |z|$ or IEEE-754 sign).
    fn signum(self) -> Self;
}

/// Ordered real floating-point or dual-real scalar contract (`f32`, `f64`, `Dual<T>`) with `0`-ULP `libm` accuracy.
pub trait RealField:
    ComplexField<Real = Self>
    + PartialOrd
    + fmt::Display
    + Default
    + Rem<Output = Self>
    + RemAssign
    + num_traits::Num
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
    /// $\pi / 2$.
    const FRAC_PI_2: Self;
    /// Full circle constant $\tau = 2\pi$.
    const TAU: Self;
    /// Euler's constant $e$.
    const E: Self;
    /// Machine epsilon $\varepsilon$.
    const EPSILON: Self;
    /// Smallest positive normal value.
    const MIN_POSITIVE: Self;
    /// Most negative finite value.
    const MIN_VALUE: Self;
    /// Largest finite value.
    const MAX_VALUE: Self;
    /// Positive infinity $+\infty$.
    const INFINITY: Self;
    /// Negative infinity $-\infty$.
    const NEG_INFINITY: Self;
    /// Quiet IEEE-754 `NaN` sentinel.
    const NAN: Self;
    /// Default relative pivot / rank tolerance multiplier (`16.0`).
    const DEFAULT_TOL_SCALE: Self;

    /// Converts an `f64` constant or threshold into `Self`.
    fn from_f64(val: f64) -> Self;

    /// Converts an unsigned index `n` into `Self`.
    fn from_usize(n: usize) -> Self;

    /// Converts a signed 32-bit integer `n` into `Self`.
    fn from_i32(n: i32) -> Self;

    /// Converts `self` (or its primal real value) to `f64`.
    fn to_f64(self) -> f64;

    /// Returns `true` if `self` is neither infinite nor `NaN`.
    fn is_finite(self) -> bool;

    /// Returns `true` if `self` is `NaN`.
    fn is_nan(self) -> bool;

    /// Exact `0`-ULP IEEE-754 square root $\sqrt{x}$ backed by `libm::sqrt` / `libm::sqrtf`.
    fn sqrt(self) -> Self;

    /// Overflow- and underflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$.
    fn hypot(self, other: Self) -> Self;

    /// Four-quadrant inverse tangent $\operatorname{atan2}(y, x) \in [-\pi, \pi]$.
    fn atan2(self, other: Self) -> Self;

    /// Accurate $e^x - 1$ for small $|x|$.
    fn expm1(self) -> Self;

    /// Accurate $\ln(1 + x)$ for small $|x|$.
    fn ln_1p(self) -> Self;

    /// Returns a value with the magnitude of `self` and the sign of `sign`.
    fn copysign(self, sign: Self) -> Self;

    /// IEEE-754 minimum $\min(x, y)$ matching `f64::min`.
    fn min(self, other: Self) -> Self;

    /// IEEE-754 maximum $\max(x, y)$ matching `f64::max`.
    fn max(self, other: Self) -> Self;

    /// Clamps `self` into `[lo, hi]`.
    fn clamp(self, lo: Self, hi: Self) -> Self;

    /// Euclidean remainder in `[0, |rhs|)`.
    fn rem_euclid(self, rhs: Self) -> Self;

    /// Total ordering over real values (delegating to `f64::total_cmp` / `f32::total_cmp`).
    fn total_cmp(&self, other: &Self) -> core::cmp::Ordering;
}
