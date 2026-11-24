//! Distinct affine [`Celsius`] and absolute SI [`Kelvin`] temperature representations.
// [cite:math_scalars]

use core::fmt;
use core::ops::{Add, Sub};

use crate::dim::DimTemperature;
use crate::frame::SiFrame;
use crate::quantity::Quantity;
use crate::scalar::Scalar;

/// Exact SI ice-point offset converting degrees Celsius to absolute Kelvin (`273.15 K`).
pub const CELSIUS_ZERO_KELVIN_OFFSET: f64 = 273.15;

/// Absolute thermodynamic temperature in SI Kelvin ($\Theta^1$).
pub type Kelvin<T = f64> = Quantity<T, DimTemperature, SiFrame>;

/// Affine temperature in degrees Celsius ($^\circ\text{C}$).
///
/// Because `0 °C = 273.15 K` is an affine origin shift rather than a linear vector space origin,
/// [`Celsius`] forbids direct multiplication, ratios, reciprocals, and Arrhenius evaluation until
/// explicitly converted into [`Kelvin`] via [`Celsius::to_kelvin`].
///
/// # Compile-Time Guard
///
/// Passing [`Celsius`] where [`Kelvin`] is required fails to compile:
///
/// ```compile_fail
/// use math_scalars::{Celsius, Kelvin};
/// fn arrhenius_inv_t(t: Kelvin) -> f64 {
///     t.recip().get()
/// }
/// let room = Celsius::new(25.0);
/// let _ = arrhenius_inv_t(room);
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Celsius<T = f64>(T);

impl<T: Copy> Celsius<T> {
    /// Wraps a raw degrees-Celsius reading into an affine [`Celsius`] value.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Returns the raw degrees-Celsius reading (`Copy` accessor).
    #[inline(always)]
    pub const fn get(self) -> T {
        self.0
    }

    /// Returns the raw degrees-Celsius reading (`&self` accessor).
    #[inline(always)]
    pub const fn value(&self) -> T {
        self.0
    }
}

impl<T: Scalar> Celsius<T> {
    /// Returns `true` if the underlying scalar is finite.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Clamps this affine temperature to `[lo, hi]`.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.max(lo.0).min(hi.0))
    }

    /// Explicitly converts affine degrees Celsius to absolute SI [`Kelvin`] ($T_{\text{K}} = T_{^\circ\text{C}} + 273.15$).
    #[inline(always)]
    pub fn to_kelvin(self) -> Kelvin<T> {
        Kelvin::new(self.0 + T::from_f64(CELSIUS_ZERO_KELVIN_OFFSET))
    }

    /// Explicitly converts absolute SI [`Kelvin`] to affine degrees Celsius ($T_{^\circ\text{C}} = T_{\text{K}} - 273.15$).
    #[inline(always)]
    pub fn from_kelvin(kelvin: Kelvin<T>) -> Self {
        Self(kelvin.get() - T::from_f64(CELSIUS_ZERO_KELVIN_OFFSET))
    }
}

impl<T: Scalar> Sub for Celsius<T> {
    type Output = Kelvin<T>;

    /// Subtracting two affine [`Celsius`] readings yields a linear temperature interval in [`Kelvin`].
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Kelvin::new(self.0 - rhs.0)
    }
}

impl<T: Scalar> Add<Kelvin<T>> for Celsius<T> {
    type Output = Self;

    /// Offsetting an affine [`Celsius`] point by a linear temperature interval (`Kelvin`) yields a [`Celsius`] point.
    #[inline(always)]
    fn add(self, delta: Kelvin<T>) -> Self::Output {
        Self(self.0 + delta.get())
    }
}

impl<T: Scalar> Sub<Kelvin<T>> for Celsius<T> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, delta: Kelvin<T>) -> Self::Output {
        Self(self.0 - delta.get())
    }
}

impl<T: fmt::Display> fmt::Display for Celsius<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} °C", self.0)
    }
}
