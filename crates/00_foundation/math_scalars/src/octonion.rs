#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
//! Flat-coordinate `#[repr(C)]` Cayley-Dickson octonion alternative division algebra [`Octonion`].
//!
//! Multiplication doubles [`Quaternion`] via $(a, b)(c, d) = (a c - \overline{d} b, d a + b \overline{c})$,
//! preserving Hurwitz norm multiplicativity $\|x y\|^2 = \|x\|^2 \|y\|^2$ and Moufang loop laws.

use crate::error::ScalarError;
use crate::quaternion::Quaternion;
use crate::traits::RealScalar;
use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Flat-coordinate `#[repr(C)]` octonion $x = \sum_{k=0}^{7} e_k \mathbf{e}_k \in \mathbb{O}$.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Octonion<T> {
    /// Real scalar basis coordinate $e_0$.
    pub e0: T,
    /// Imaginary basis coordinate $e_1$.
    pub e1: T,
    /// Imaginary basis coordinate $e_2$.
    pub e2: T,
    /// Imaginary basis coordinate $e_3$.
    pub e3: T,
    /// Imaginary basis coordinate $e_4$.
    pub e4: T,
    /// Imaginary basis coordinate $e_5$.
    pub e5: T,
    /// Imaginary basis coordinate $e_6$.
    pub e6: T,
    /// Imaginary basis coordinate $e_7$.
    pub e7: T,
}

impl<T> Octonion<T> {
    /// Constructs an octonion from its eight flat coordinates.
    #[inline]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(e0: T, e1: T, e2: T, e3: T, e4: T, e5: T, e6: T, e7: T) -> Self {
        Self {
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            e7,
        }
    }

    /// Constructs an octonion from a contiguous 8-element coordinate array.
    #[inline]
    #[must_use]
    pub const fn from_array(c: [T; 8]) -> Self
    where
        T: Copy,
    {
        Self::new(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
    }

    /// Returns `[e0, e1, e2, e3, e4, e5, e6, e7]` in contiguous coordinate order.
    #[inline]
    #[must_use]
    pub const fn to_array(self) -> [T; 8]
    where
        T: Copy,
    {
        [
            self.e0, self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7,
        ]
    }

    /// Constructs $(a, b) \in \mathbb{H} \oplus \mathbb{H}$ from Cayley-Dickson quaternion halves.
    #[inline]
    #[must_use]
    pub const fn from_quaternions(lo: Quaternion<T>, hi: Quaternion<T>) -> Self
    where
        T: Copy,
    {
        Self::new(lo.w, lo.x, lo.y, lo.z, hi.w, hi.x, hi.y, hi.z)
    }

    /// Splits `self` into its Cayley-Dickson quaternion pair `(lo, hi)`.
    #[inline]
    #[must_use]
    pub const fn to_quaternions(self) -> (Quaternion<T>, Quaternion<T>)
    where
        T: Copy,
    {
        (
            Quaternion::new(self.e0, self.e1, self.e2, self.e3),
            Quaternion::new(self.e4, self.e5, self.e6, self.e7),
        )
    }
}

impl<T: RealScalar> Octonion<T> {
    /// Additive identity $0 \in \mathbb{O}$.
    pub const ZERO: Self = Self::from_array([T::ZERO; 8]);
    /// Multiplicative identity $1 \in \mathbb{O}$.
    pub const ONE: Self = Self::new(
        T::ONE,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
    );

    /// Cayley-Dickson conjugate $\overline{(a, b)} = (\overline{a}, -b)$.
    #[inline]
    #[must_use]
    pub fn conj(self) -> Self {
        let (a, b) = self.to_quaternions();
        Self::from_quaternions(a.conj(), -b)
    }

    /// Euclidean inner product $\langle x, y \rangle = \sum_{k=0}^{7} x_k y_k$.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        let (a, b) = self.to_quaternions();
        let (c, d) = rhs.to_quaternions();
        a.dot(c) + b.dot(d)
    }

    /// Squared Euclidean norm $\|x\|^2 = \sum_{k=0}^{7} e_k^2$.
    #[inline]
    #[must_use]
    pub fn norm_sqr(self) -> T {
        self.dot(self)
    }

    /// Overflow-safe Euclidean norm $\|x\| = \sqrt{\|a\|^2 + \|b\|^2}$.
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        let (a, b) = self.to_quaternions();
        a.norm().hypot(b.norm())
    }

    /// Scales all eight coordinates by `s`.
    #[inline]
    #[must_use]
    pub fn scale(self, s: T) -> Self {
        let (a, b) = self.to_quaternions();
        Self::from_quaternions(a.scale(s), b.scale(s))
    }

    /// Returns `true` when all eight coordinates are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        let (a, b) = self.to_quaternions();
        a.is_finite() && b.is_finite()
    }

    /// Alternative division-algebra inverse $x^{-1} = \overline{x} / \|x\|^2$.
    #[must_use]
    pub fn checked_inv(self) -> Option<Self> {
        if !self.is_finite() {
            return None;
        }
        let m = self.norm();
        if m == T::ZERO || !m.is_finite() {
            return None;
        }
        let unit = self.scale(T::ONE / m);
        let inv = unit.conj().scale(T::ONE / m);
        inv.is_finite().then_some(inv)
    }

    /// Fallible inverse returning [`ScalarError::DivisionByZero`] when $x = 0$ or non-finite.
    pub fn try_inverse(self) -> Result<Self, ScalarError> {
        self.checked_inv().ok_or(ScalarError::DivisionByZero)
    }
}

impl<T: fmt::Display> fmt::Display for Octonion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({} + {}e1 + {}e2 + {}e3 + {}e4 + {}e5 + {}e6 + {}e7)",
            self.e0, self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7
        )
    }
}

impl<T: Copy + Add<Output = T>> Add for Octonion<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (a, b) = self.to_quaternions();
        let (c, d) = rhs.to_quaternions();
        Self::from_quaternions(a + c, b + d)
    }
}

impl<T: Copy + Add<Output = T>> AddAssign for Octonion<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Copy + Sub<Output = T>> Sub for Octonion<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let (a, b) = self.to_quaternions();
        let (c, d) = rhs.to_quaternions();
        Self::from_quaternions(a - c, b - d)
    }
}

impl<T: Copy + Sub<Output = T>> SubAssign for Octonion<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>> Mul
    for Octonion<T>
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (a, b) = self.to_quaternions();
        let (c, d) = rhs.to_quaternions();
        let d_conj = Quaternion::new(d.w, -d.x, -d.y, -d.z);
        let c_conj = Quaternion::new(c.w, -c.x, -c.y, -c.z);
        Self::from_quaternions(a * c - d_conj * b, d * a + b * c_conj)
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>> MulAssign
    for Octonion<T>
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Copy + Neg<Output = T>> Neg for Octonion<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let (a, b) = self.to_quaternions();
        Self::from_quaternions(-a, -b)
    }
}
