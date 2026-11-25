#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
//! Flat-coordinate `#[repr(C)]` Hamilton quaternion division ring [`Quaternion`] and
//! unit-norm 3D rotation group [`UnitQuaternion`] ($SO(3)$ double cover).

use crate::error::ScalarError;
use crate::traits::{FieldScalar, RealScalar};
use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Flat-coordinate `#[repr(C)]` Hamilton quaternion $q = w + x i + y j + z k$.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Quaternion<T> {
    /// Scalar (real) component $w$.
    pub w: T,
    /// $i$-imaginary component $x$.
    pub x: T,
    /// $j$-imaginary component $y$.
    pub y: T,
    /// $k$-imaginary component $z$.
    pub z: T,
}

impl<T> Quaternion<T> {
    /// Constructs $q = w + x i + y j + z k$.
    #[inline]
    #[must_use]
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    /// Constructs from `[w, x, y, z]` array.
    #[inline]
    #[must_use]
    pub const fn from_wxyz(wxyz: [T; 4]) -> Self
    where
        T: Copy,
    {
        Self {
            w: wxyz[0],
            x: wxyz[1],
            y: wxyz[2],
            z: wxyz[3],
        }
    }

    /// Returns `[w, x, y, z]` in contiguous coordinate order.
    #[inline]
    #[must_use]
    pub const fn as_wxyz(self) -> [T; 4]
    where
        T: Copy,
    {
        [self.w, self.x, self.y, self.z]
    }
}

impl<T: RealScalar> Quaternion<T> {
    /// Additive identity $0$.
    pub const ZERO: Self = Self::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    /// Multiplicative identity $1$.
    pub const ONE: Self = Self::new(T::ONE, T::ZERO, T::ZERO, T::ZERO);

    /// Quaternion conjugate $\overline{q} = w - x i - y j - z k$.
    #[inline]
    #[must_use]
    pub fn conj(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Euclidean inner product $\langle p, q \rangle = w_1 w_2 + x_1 x_2 + y_1 y_2 + z_1 z_2$.
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

    /// Overflow-safe Euclidean norm $\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2}$.
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        self.w.hypot(self.x).hypot(self.y.hypot(self.z))
    }

    /// Scales all four coordinates by `s`.
    #[inline]
    #[must_use]
    pub fn scale(self, s: T) -> Self {
        Self::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }

    /// Returns `true` when all four coordinates are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Multiplicative inverse $q^{-1} = \overline{q} / \|q\|^2$, returning `None` if zero or non-finite.
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

    /// Fallible division-ring inverse returning [`ScalarError::DivisionByZero`] when $q = 0$ or non-finite.
    pub fn try_inverse(self) -> Result<Self, ScalarError> {
        self.checked_inv().ok_or(ScalarError::DivisionByZero)
    }

    /// Normalizes `self` to unit length, returning [`ScalarError::NonInvertible`] if degenerate.
    pub fn try_normalize(self) -> Result<Self, ScalarError> {
        if !self.is_finite() {
            return Err(ScalarError::NonFiniteFloat);
        }
        let n = self.norm();
        if n <= T::from_f64(1e-12) || !n.is_finite() {
            return Err(ScalarError::NonInvertible);
        }
        Ok(self.scale(T::ONE / n))
    }
}

impl<T: fmt::Display> fmt::Display for Quaternion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}i + {}j + {}k)", self.w, self.x, self.y, self.z)
    }
}

impl<T: Add<Output = T>> Add for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.w + rhs.w,
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
        )
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
        Self::new(
            self.w - rhs.w,
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
        )
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

impl<T: Neg<Output = T>> Neg for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.w, -self.x, -self.y, -self.z)
    }
}

/// Unit-norm quaternion `#[repr(C)]` representing a 3D rotation in $SO(3)$.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct UnitQuaternion<T>(pub(crate) Quaternion<T>);

impl<T: RealScalar> Default for UnitQuaternion<T> {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: RealScalar> UnitQuaternion<T> {
    /// Identity 3D rotation $q = (1, 0, 0, 0)$.
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self(Quaternion::ONE)
    }

    /// Returns the underlying unit [`Quaternion`].
    #[inline]
    #[must_use]
    pub const fn quaternion(self) -> Quaternion<T> {
        self.0
    }

    /// Returns `[w, x, y, z]` unit quaternion coordinates.
    #[inline]
    #[must_use]
    pub const fn as_wxyz(self) -> [T; 4] {
        self.0.as_wxyz()
    }

    /// Normalizes `q` into a `UnitQuaternion`, returning an error if zero or non-finite.
    pub fn try_from_quaternion(q: Quaternion<T>) -> Result<Self, ScalarError> {
        q.try_normalize().map(Self)
    }

    /// Normalizes `[w, x, y, z]` or falls back to [`Self::identity`] when degenerate/non-finite.
    #[must_use]
    pub fn from_wxyz_or_identity(wxyz: [T; 4]) -> Self {
        Quaternion::from_wxyz(wxyz)
            .try_normalize()
            .map_or_else(|_| Self::identity(), Self)
    }

    /// Constructs a rotation of `angle` radians around `axis` (falling back to identity if `axis` is zero).
    #[must_use]
    pub fn from_axis_angle(axis: [T; 3], angle: T) -> Self {
        let len = axis[0].hypot(axis[1]).hypot(axis[2]);
        if len <= T::from_f64(1e-12) || !len.is_finite() || !angle.is_finite() {
            return Self::identity();
        }
        let half = angle * T::HALF;
        let s = half.sin() / len;
        Self(Quaternion::new(
            half.cos(),
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
        ))
    }

    /// Constructs a rotation from an axis-angle vector $\omega \in \mathbb{R}^3$ (`‖ω‖` = angle in radians).
    #[must_use]
    pub fn from_scaled_axis(axis_angle: [T; 3]) -> Self {
        let angle = axis_angle[0].hypot(axis_angle[1]).hypot(axis_angle[2]);
        if angle <= T::from_f64(1e-12) || !angle.is_finite() {
            return Self::identity();
        }
        Self::from_axis_angle(axis_angle, angle)
    }

    /// Shortest-arc rotation mapping non-zero vector `from` onto `to`.
    #[must_use]
    pub fn rotation_between(from: [T; 3], to: [T; 3]) -> Option<Self> {
        let n_from = from[0].hypot(from[1]).hypot(from[2]);
        let n_to = to[0].hypot(to[1]).hypot(to[2]);
        let tol = T::from_f64(1e-12);
        if n_from <= tol || n_to <= tol || !n_from.is_finite() || !n_to.is_finite() {
            return None;
        }
        let u = [from[0] / n_from, from[1] / n_from, from[2] / n_from];
        let v = [to[0] / n_to, to[1] / n_to, to[2] / n_to];
        let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        if dot < T::from_f64(-0.999_999) {
            let ortho = if FieldScalar::abs(u[0]) < T::from_f64(0.9) {
                [T::ZERO, -u[2], u[1]]
            } else {
                [u[2], T::ZERO, -u[0]]
            };
            return Some(Self::from_axis_angle(ortho, T::PI));
        }
        let cx = u[1] * v[2] - u[2] * v[1];
        let cy = u[2] * v[0] - u[0] * v[2];
        let cz = u[0] * v[1] - u[1] * v[0];
        Quaternion::new(T::ONE + dot, cx, cy, cz)
            .try_normalize()
            .ok()
            .map(Self)
    }

    /// Inverse rotation $q^{-1} = \overline{q}$.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        Self(self.0.conj())
    }

    /// Rotates a 3D vector `v` via $q v q^{-1}$ with zero heap allocations.
    #[inline]
    #[must_use]
    pub fn rotate_vec3(self, v: [T; 3]) -> [T; 3] {
        let q = self.0;
        let tx = T::TWO * (q.y * v[2] - q.z * v[1]);
        let ty = T::TWO * (q.z * v[0] - q.x * v[2]);
        let tz = T::TWO * (q.x * v[1] - q.y * v[0]);
        [
            v[0] + q.w * tx + (q.y * tz - q.z * ty),
            v[1] + q.w * ty + (q.z * tx - q.x * tz),
            v[2] + q.w * tz + (q.x * ty - q.y * tx),
        ]
    }

    /// Converts `self` to a row-major $3 \times 3$ orthonormal rotation matrix $R \in SO(3)$.
    #[must_use]
    pub fn to_rotation_matrix3(self) -> [[T; 3]; 3] {
        let q = self.0;
        let xx = T::TWO * q.x * q.x;
        let yy = T::TWO * q.y * q.y;
        let zz = T::TWO * q.z * q.z;
        let xy = T::TWO * q.x * q.y;
        let xz = T::TWO * q.x * q.z;
        let yz = T::TWO * q.y * q.z;
        let wx = T::TWO * q.w * q.x;
        let wy = T::TWO * q.w * q.y;
        let wz = T::TWO * q.w * q.z;
        [
            [T::ONE - (yy + zz), xy - wz, xz + wy],
            [xy + wz, T::ONE - (xx + zz), yz - wx],
            [xz - wy, yz + wx, T::ONE - (xx + yy)],
        ]
    }
}

impl<T: RealScalar> Mul for UnitQuaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_wxyz_or_identity((self.0 * rhs.0).as_wxyz())
    }
}
