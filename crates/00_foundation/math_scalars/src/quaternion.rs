#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use crate::traits::RealScalar;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Flat `#[repr(C)]` scalar-first Hamilton quaternion $q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Quaternion<T> {
    /// Real scalar coordinate $w$.
    pub w: T,
    /// Imaginary $\mathbf{i}$ coordinate $x$.
    pub x: T,
    /// Imaginary $\mathbf{j}$ coordinate $y$.
    pub y: T,
    /// Imaginary $\mathbf{k}$ coordinate $z$.
    pub z: T,
}

impl<T> Quaternion<T> {
    /// Constructs $q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$.
    #[inline]
    #[must_use]
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    /// Constructs a quaternion from a scalar-first `[w, x, y, z]` array.
    #[inline]
    #[must_use]
    pub const fn from_wxyz(coords: [T; 4]) -> Self
    where
        T: Copy,
    {
        Self {
            w: coords[0],
            x: coords[1],
            y: coords[2],
            z: coords[3],
        }
    }

    /// Returns scalar-first coordinates `[w, x, y, z]`.
    #[inline]
    #[must_use]
    pub const fn to_wxyz(self) -> [T; 4]
    where
        T: Copy,
    {
        [self.w, self.x, self.y, self.z]
    }
}

impl<T: RealScalar> Quaternion<T> {
    /// Additive identity $0$.
    pub const ZERO: Self = Self {
        w: T::ZERO,
        x: T::ZERO,
        y: T::ZERO,
        z: T::ZERO,
    };
    /// Multiplicative identity $1$.
    pub const ONE: Self = Self {
        w: T::ONE,
        x: T::ZERO,
        y: T::ZERO,
        z: T::ZERO,
    };
    /// Basis unit $\mathbf{i}$.
    pub const I: Self = Self {
        w: T::ZERO,
        x: T::ONE,
        y: T::ZERO,
        z: T::ZERO,
    };
    /// Basis unit $\mathbf{j}$.
    pub const J: Self = Self {
        w: T::ZERO,
        x: T::ZERO,
        y: T::ONE,
        z: T::ZERO,
    };
    /// Basis unit $\mathbf{k}$.
    pub const K: Self = Self {
        w: T::ZERO,
        x: T::ZERO,
        y: T::ZERO,
        z: T::ONE,
    };

    /// Quaternion conjugate $q^* = w - x\mathbf{i} - y\mathbf{j} - z\mathbf{k}$.
    #[inline]
    #[must_use]
    pub fn conj(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Euclidean inner product $q_1 \cdot q_2$.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Squared norm $\|q\|^2 = w^2 + x^2 + y^2 + z^2$.
    #[inline]
    #[must_use]
    pub fn norm_sqr(self) -> T {
        self.dot(self)
    }

    /// Euclidean norm $\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2}$.
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        self.w.hypot(self.x).hypot(self.y.hypot(self.z))
    }

    /// Scales all four components by `s`.
    #[inline]
    #[must_use]
    pub fn scale(self, s: T) -> Self {
        Self {
            w: self.w * s,
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Returns `true` when all four coordinates are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Checked two-sided multiplicative inverse $q^{-1} = q^* / \|q\|^2$.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when $\|q\| = 0$ or coordinates are non-finite.
    pub fn checked_inv(self) -> Result<Self, ScalarError> {
        if !self.is_finite() {
            return Err(ScalarError::NonInvertible);
        }
        let m = self.norm();
        if m == T::ZERO || !m.is_finite() {
            return Err(ScalarError::NonInvertible);
        }
        assert!(m > T::ZERO);
        assert!(m.is_finite());
        let unit_conj = self.conj().scale(T::ONE / m);
        debug_assert!(unit_conj.is_finite());
        let out = unit_conj.scale(T::ONE / m);
        if out.is_finite() {
            assert!(out.w.is_finite() && out.x.is_finite());
            assert!(out.y.is_finite() && out.z.is_finite());
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

impl<T: Neg<Output = T>> Neg for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<T: Add<Output = T>> Add for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: AddAssign> AddAssign for Quaternion<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.w += rhs.w;
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T: Sub<Output = T>> Sub for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: SubAssign> SubAssign for Quaternion<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.w -= rhs.w;
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Mul for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> MulAssign for Quaternion<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: RealScalar> Div for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("Quaternion division by zero")
    }
}

impl<T: fmt::Display> fmt::Display for Quaternion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}i + {}j + {}k)", self.w, self.x, self.y, self.z)
    }
}
