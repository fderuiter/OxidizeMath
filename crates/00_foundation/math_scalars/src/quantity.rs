//! Zero-cost `#[repr(transparent)]` physical [`Quantity<T, D, F>`] and dimension-safe operators.
// [cite:math_scalars]

use core::cmp::Ordering;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::dim::{AddDim, DimSpec, Dimensionless, HalfDim, NegDim, SubDim};
use crate::frame::UnitFrame;
use crate::scalar::Scalar;

/// Physical quantity with scalar representation `T`, seven-exponent SI dimension `D`, and
/// unit-system frame `F`.
///
/// Guaranteed by `#[repr(transparent)]` to have the exact size, alignment, and ABI of `T`
/// (`8` bytes for `f64`, `4` bytes for `f32`) with zero `Deref` coercions.
///
/// # Compile-Time Guards
///
/// Calling `.ln()` on a dimensional length fails to compile:
///
/// ```compile_fail
/// use math_scalars::Meters;
/// let x = Meters::new(10.0);
/// let _ = x.ln();
/// ```
///
/// Calling `.exp()` on a dimensional energy fails to compile:
///
/// ```compile_fail
/// use math_scalars::Joules;
/// let e = Joules::new(1.0);
/// let _ = e.exp();
/// ```
///
/// Taking `.sqrt()` of an odd-exponent quantity fails to compile:
///
/// ```compile_fail
/// use math_scalars::Meters;
/// let x = Meters::new(9.0);
/// let _ = x.sqrt();
/// ```
///
/// Adding quantities with mismatched SI dimensions fails to compile:
///
/// ```compile_fail
/// use math_scalars::{Meters, Seconds};
/// let _ = Meters::new(1.0) + Seconds::new(1.0);
/// ```
///
/// Assigning or adding a [`crate::ViewportFrame`] length to a [`crate::GalacticKpcFrame`] length
/// without explicit scale conversion fails to compile:
///
/// ```compile_fail
/// use math_scalars::{Kiloparsecs, ViewportLength};
/// let vp = ViewportLength::new(0.55);
/// let _kpc: Kiloparsecs = vp;
/// ```
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Quantity<T, D: DimSpec, F: UnitFrame>(
    T,
    #[cfg_attr(feature = "serde", serde(skip))] PhantomData<fn() -> (D, F)>,
);

impl<T: Copy, D: DimSpec, F: UnitFrame> Copy for Quantity<T, D, F> {}

impl<T: Clone, D: DimSpec, F: UnitFrame> Clone for Quantity<T, D, F> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<T: Default, D: DimSpec, F: UnitFrame> Default for Quantity<T, D, F> {
    #[inline(always)]
    fn default() -> Self {
        Self(T::default(), PhantomData)
    }
}

impl<T: PartialEq, D: DimSpec, F: UnitFrame> PartialEq for Quantity<T, D, F> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq, D: DimSpec, F: UnitFrame> Eq for Quantity<T, D, F> {}

impl<T: PartialOrd, D: DimSpec, F: UnitFrame> PartialOrd for Quantity<T, D, F> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: fmt::Debug, D: DimSpec, F: UnitFrame> fmt::Debug for Quantity<T, D, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quantity<{}:{:?}>({:?})",
            F::NAME,
            D::exponents(),
            self.0
        )
    }
}

impl<T: fmt::Display, D: DimSpec, F: UnitFrame> fmt::Display for Quantity<T, D, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl<T, D: DimSpec, F: UnitFrame> Quantity<T, D, F> {
    /// Wraps a raw scalar `value` into a typed [`Quantity`] in frame `F`.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self(value, PhantomData)
    }

    /// Extracts the underlying scalar value by value.
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: Copy, D: DimSpec, F: UnitFrame> Quantity<T, D, F> {
    /// Returns the underlying scalar value (`Copy` accessor).
    #[inline(always)]
    pub const fn get(self) -> T {
        self.0
    }

    /// Returns the underlying scalar value (`&self` accessor).
    #[inline(always)]
    pub const fn value(&self) -> T {
        self.0
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> Quantity<T, D, F> {
    /// Additive identity `0` for this dimension and frame.
    pub const ZERO: Self = Self::new(T::ZERO);

    /// Returns `true` if the underlying scalar is finite.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Returns the magnitude $|q|$ preserving dimension `D` and frame `F`.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(self.0.abs())
    }

    /// Returns the minimum of `self` and `other` in matching units.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self::new(self.0.min(other.0))
    }

    /// Returns the maximum of `self` and `other` in matching units.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self::new(self.0.max(other.0))
    }

    /// Clamps `self` to `[lo, hi]` in matching units.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Computes the multiplicative inverse $1 / q$ with negated SI exponents.
    #[inline(always)]
    pub fn recip(self) -> Quantity<T, D::Output, F>
    where
        D: NegDim,
    {
        Quantity::new(T::ONE / self.0)
    }

    /// Computes the principal square root $\sqrt{q}$ when all seven SI base exponents are even.
    #[inline(always)]
    pub fn sqrt(self) -> Quantity<T, D::Output, F>
    where
        D: HalfDim,
    {
        Quantity::new(self.0.sqrt())
    }
}

impl<T: Scalar, F: UnitFrame> Quantity<T, Dimensionless, F> {
    /// Multiplicative identity `1` for dimensionless quantities.
    pub const ONE: Self = Self::new(T::ONE);

    /// Natural logarithm $\ln(q)$, restricted to [`Dimensionless`] ratios.
    #[inline(always)]
    pub fn ln(self) -> Self {
        Self::new(self.0.ln())
    }

    /// Natural exponential $e^q$, restricted to [`Dimensionless`] ratios.
    #[inline(always)]
    pub fn exp(self) -> Self {
        Self::new(self.0.exp())
    }

    /// Sine $\sin(q)$, restricted to [`Dimensionless`] angles/phases.
    #[inline(always)]
    pub fn sin(self) -> Self {
        Self::new(self.0.sin())
    }

    /// Cosine $\cos(q)$, restricted to [`Dimensionless`] angles/phases.
    #[inline(always)]
    pub fn cos(self) -> Self {
        Self::new(self.0.cos())
    }

    /// Raises a [`Dimensionless`] ratio to an arbitrary real exponent `exp`.
    #[inline(always)]
    pub fn powf(self, exp: T) -> Self {
        Self::new(self.0.powf(exp))
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> Add for Quantity<T, D, F> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> AddAssign for Quantity<T, D, F> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> Sub for Quantity<T, D, F> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.0 - rhs.0)
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> SubAssign for Quantity<T, D, F> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> Neg for Quantity<T, D, F> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.0)
    }
}

impl<T: Scalar, D1: AddDim<D2>, D2: DimSpec, F: UnitFrame> Mul<Quantity<T, D2, F>>
    for Quantity<T, D1, F>
{
    type Output = Quantity<T, D1::Output, F>;
    #[inline(always)]
    fn mul(self, rhs: Quantity<T, D2, F>) -> Self::Output {
        Quantity::new(self.0 * rhs.0)
    }
}

impl<T: Scalar, D1: SubDim<D2>, D2: DimSpec, F: UnitFrame> Div<Quantity<T, D2, F>>
    for Quantity<T, D1, F>
{
    type Output = Quantity<T, D1::Output, F>;
    #[inline(always)]
    fn div(self, rhs: Quantity<T, D2, F>) -> Self::Output {
        Quantity::new(self.0 / rhs.0)
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> Mul<T> for Quantity<T, D, F> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, scalar: T) -> Self::Output {
        Self::new(self.0 * scalar)
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> MulAssign<T> for Quantity<T, D, F> {
    #[inline(always)]
    fn mul_assign(&mut self, scalar: T) {
        self.0 *= scalar;
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> Div<T> for Quantity<T, D, F> {
    type Output = Self;
    #[inline(always)]
    fn div(self, scalar: T) -> Self::Output {
        Self::new(self.0 / scalar)
    }
}

impl<T: Scalar, D: DimSpec, F: UnitFrame> DivAssign<T> for Quantity<T, D, F> {
    #[inline(always)]
    fn div_assign(&mut self, scalar: T) {
        self.0 /= scalar;
    }
}

impl<D: DimSpec, F: UnitFrame> Mul<Quantity<f64, D, F>> for f64 {
    type Output = Quantity<f64, D, F>;
    #[inline(always)]
    fn mul(self, rhs: Quantity<f64, D, F>) -> Self::Output {
        Quantity::new(self * rhs.0)
    }
}

impl<D: DimSpec, F: UnitFrame> Mul<Quantity<f32, D, F>> for f32 {
    type Output = Quantity<f32, D, F>;
    #[inline(always)]
    fn mul(self, rhs: Quantity<f32, D, F>) -> Self::Output {
        Quantity::new(self * rhs.0)
    }
}

impl<D: NegDim, F: UnitFrame> Div<Quantity<f64, D, F>> for f64 {
    type Output = Quantity<f64, D::Output, F>;
    #[inline(always)]
    fn div(self, rhs: Quantity<f64, D, F>) -> Self::Output {
        Quantity::new(self / rhs.0)
    }
}

impl<D: NegDim, F: UnitFrame> Div<Quantity<f32, D, F>> for f32 {
    type Output = Quantity<f32, D::Output, F>;
    #[inline(always)]
    fn div(self, rhs: Quantity<f32, D, F>) -> Self::Output {
        Quantity::new(self / rhs.0)
    }
}
