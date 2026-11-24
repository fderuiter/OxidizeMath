#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use crate::quaternion::Quaternion;
use crate::traits::RealScalar;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Flat `#[repr(C)]` eight-dimensional non-associative alternative Cayley-Dickson division
/// algebra $\mathbb{O}$ constructed by doubling [`Quaternion<T>`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Octonion<T> {
    /// Real scalar unit $e_0 = 1$.
    pub e0: T,
    /// Imaginary basis unit $e_1$.
    pub e1: T,
    /// Imaginary basis unit $e_2$.
    pub e2: T,
    /// Imaginary basis unit $e_3$.
    pub e3: T,
    /// Imaginary basis unit $e_4$.
    pub e4: T,
    /// Imaginary basis unit $e_5$.
    pub e5: T,
    /// Imaginary basis unit $e_6$.
    pub e6: T,
    /// Imaginary basis unit $e_7$.
    pub e7: T,
}

impl<T> Octonion<T> {
    /// Constructs an octonion from its eight flat coordinates `e0..=e7`.
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

    /// Constructs an octonion from `[e0, e1, e2, e3, e4, e5, e6, e7]`.
    #[inline]
    #[must_use]
    pub const fn from_array(c: [T; 8]) -> Self
    where
        T: Copy,
    {
        Self {
            e0: c[0],
            e1: c[1],
            e2: c[2],
            e3: c[3],
            e4: c[4],
            e5: c[5],
            e6: c[6],
            e7: c[7],
        }
    }

    /// Returns coordinates `[e0, e1, e2, e3, e4, e5, e6, e7]`.
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

    /// Constructs an octonion $(a, b) \in \mathbb{H} \oplus \mathbb{H}\ell$ from two quaternions.
    #[inline]
    #[must_use]
    pub const fn from_quaternions(a: Quaternion<T>, b: Quaternion<T>) -> Self
    where
        T: Copy,
    {
        Self {
            e0: a.w,
            e1: a.x,
            e2: a.y,
            e3: a.z,
            e4: b.w,
            e5: b.x,
            e6: b.y,
            e7: b.z,
        }
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
    /// Additive identity $0$.
    pub const ZERO: Self = Self::new(
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
        T::ZERO,
    );
    /// Multiplicative identity $1$.
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

    /// Returns basis unit $e_k$ for $k \in 0..8$ (`ZERO` if $k \ge 8$).
    #[must_use]
    pub fn basis(k: usize) -> Self {
        let mut c = [T::ZERO; 8];
        if k < 8 {
            c[k] = T::ONE;
        }
        Self::from_array(c)
    }

    /// Octonion conjugate $x^* = (a^*, -b)$.
    #[inline]
    #[must_use]
    pub fn conj(self) -> Self {
        let (a, b) = self.to_quaternions();
        Self::from_quaternions(a.conj(), -b)
    }

    /// Euclidean inner product $\langle x, y \rangle = \sum_{i=0}^7 x_i y_i$.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        let (a, b) = self.to_quaternions();
        let (c, d) = rhs.to_quaternions();
        a.dot(c) + b.dot(d)
    }

    /// Squared norm $\|x\|^2 = \sum_{i=0}^7 e_i^2$.
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

    /// Checked two-sided multiplicative inverse $x^{-1} = x^* / \|x\|^2$.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when $\|x\| = 0$ or coordinates are non-finite.
    pub fn checked_inv(self) -> Result<Self, ScalarError> {
        if !self.is_finite() {
            return Err(ScalarError::NonInvertible);
        }
        let m = self.norm();
        if m == T::ZERO || !m.is_finite() {
            return Err(ScalarError::NonInvertible);
        }
        let unit_conj = self.conj().scale(T::ONE / m);
        let out = unit_conj.scale(T::ONE / m);
        if out.is_finite() {
            assert!(out.e0.is_finite() && out.e1.is_finite());
            assert!(out.e2.is_finite() && out.e3.is_finite());
            assert!(out.e4.is_finite() && out.e5.is_finite());
            assert!(out.e6.is_finite() && out.e7.is_finite());
            Ok(out)
        } else {
            Err(ScalarError::NonInvertible)
        }
    }

    /// Right division `self * rhs.checked_inv()?`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `rhs` is not invertible.
    pub fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        let inv = rhs.checked_inv().map_err(|_| ScalarError::DivisionByZero)?;
        Ok(self * inv)
    }
}

impl<T: Neg<Output = T>> Neg for Octonion<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(
            -self.e0, -self.e1, -self.e2, -self.e3, -self.e4, -self.e5, -self.e6, -self.e7,
        )
    }
}

impl<T: Add<Output = T>> Add for Octonion<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.e0 + rhs.e0,
            self.e1 + rhs.e1,
            self.e2 + rhs.e2,
            self.e3 + rhs.e3,
            self.e4 + rhs.e4,
            self.e5 + rhs.e5,
            self.e6 + rhs.e6,
            self.e7 + rhs.e7,
        )
    }
}

impl<T: AddAssign> AddAssign for Octonion<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.e0 += rhs.e0;
        self.e1 += rhs.e1;
        self.e2 += rhs.e2;
        self.e3 += rhs.e3;
        self.e4 += rhs.e4;
        self.e5 += rhs.e5;
        self.e6 += rhs.e6;
        self.e7 += rhs.e7;
    }
}

impl<T: Sub<Output = T>> Sub for Octonion<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.e0 - rhs.e0,
            self.e1 - rhs.e1,
            self.e2 - rhs.e2,
            self.e3 - rhs.e3,
            self.e4 - rhs.e4,
            self.e5 - rhs.e5,
            self.e6 - rhs.e6,
            self.e7 - rhs.e7,
        )
    }
}

impl<T: SubAssign> SubAssign for Octonion<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.e0 -= rhs.e0;
        self.e1 -= rhs.e1;
        self.e2 -= rhs.e2;
        self.e3 -= rhs.e3;
        self.e4 -= rhs.e4;
        self.e5 -= rhs.e5;
        self.e6 -= rhs.e6;
        self.e7 -= rhs.e7;
    }
}

/// Cayley-Dickson multiplication $(a, b)(c, d) = (ac - d^* b,\; da + b c^*)$.
impl<T: RealScalar> Mul for Octonion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (a, b) = self.to_quaternions();
        let (c, d) = rhs.to_quaternions();
        Self::from_quaternions(a * c - d.conj() * b, d * a + b * c.conj())
    }
}

impl<T: RealScalar> MulAssign for Octonion<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: RealScalar> Div for Octonion<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("Octonion division by zero")
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
