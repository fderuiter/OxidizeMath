#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use crate::quaternion::Quaternion;
use crate::traits::RealScalar;
use core::fmt;
use core::ops::Mul;

/// Unit-norm quaternion $q \in \operatorname{Sp}(1) \cong \operatorname{SU}(2)$ representing
/// a 3D spatial rotation in $\operatorname{SO}(3)$ via $v \mapsto q v q^*$.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitQuaternion<T>(Quaternion<T>);

impl<T: RealScalar> Default for UnitQuaternion<T> {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: RealScalar> UnitQuaternion<T> {
    /// Identity rotation $(1, 0, 0, 0)$.
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self(Quaternion::ONE)
    }

    /// Returns the underlying normalized [`Quaternion<T>`].
    #[inline]
    #[must_use]
    pub const fn quaternion(self) -> Quaternion<T> {
        self.0
    }

    /// Returns scalar-first coordinates `[w, x, y, z]`.
    #[inline]
    #[must_use]
    pub const fn to_wxyz(self) -> [T; 4] {
        self.0.to_wxyz()
    }

    /// Normalizes a non-zero finite [`Quaternion<T>`] into `UnitQuaternion<T>`.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `q` is non-finite or its norm is below `MIN_POSITIVE`.
    pub fn try_from_quaternion(q: Quaternion<T>) -> Result<Self, ScalarError> {
        if !q.is_finite() {
            return Err(ScalarError::NonInvertible);
        }
        let n = q.norm();
        if !n.is_finite() || n <= T::MIN_POSITIVE {
            return Err(ScalarError::NonInvertible);
        }
        let out = Self(q.scale(T::ONE / n));
        assert!(out.is_finite());
        Ok(out)
    }

    /// Normalizes `[w, x, y, z]` into a `UnitQuaternion<T>`, falling back to [`Self::identity`]
    /// if the norm is degenerate ($\le 10^{-6}$), while preserving `NaN`/`Infinity` detection
    /// via [`Self::try_from_wxyz`].
    #[must_use]
    pub fn from_wxyz_lossy(coords: [T; 4]) -> Self {
        let q = Quaternion::from_wxyz(coords);
        if !q.is_finite() {
            return Self(q);
        }
        let n = q.norm();
        if n > T::from_f64(1.0e-6) {
            Self(q.scale(T::ONE / n))
        } else {
            Self::identity()
        }
    }

    /// Validates and normalizes scalar-first `[w, x, y, z]` coordinates.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `coords` is zero or non-finite.
    pub fn try_from_wxyz(coords: [T; 4]) -> Result<Self, ScalarError> {
        Self::try_from_quaternion(Quaternion::from_wxyz(coords))
    }

    /// Constructs a right-handed rotation of `angle` radians around `axis` (normalizing `axis`).
    /// Returns [`Self::identity`] if `axis` has zero length.
    #[must_use]
    pub fn from_axis_angle(axis: [T; 3], angle: T) -> Self {
        let len = axis[0].hypot(axis[1]).hypot(axis[2]);
        if !len.is_finite() || len <= T::MIN_POSITIVE {
            return Self::identity();
        }
        let half = angle * T::HALF;
        let (s, c) = half.sin_cos();
        let k = s / len;
        Self(Quaternion::new(c, axis[0] * k, axis[1] * k, axis[2] * k)).renormalized()
    }

    /// Constructs a rotation from an axis-angle vector $\boldsymbol{\omega}$ whose magnitude is the angle in radians.
    #[must_use]
    pub fn from_scaled_axis(axis_angle: [T; 3]) -> Self {
        let angle = axis_angle[0].hypot(axis_angle[1]).hypot(axis_angle[2]);
        if angle <= T::EPSILON {
            let half = T::HALF;
            return Self::from_wxyz_lossy([
                T::ONE,
                axis_angle[0] * half,
                axis_angle[1] * half,
                axis_angle[2] * half,
            ]);
        }
        Self::from_axis_angle(axis_angle, angle)
    }

    /// Minimal-arc rotation mapping direction `from` onto direction `to`.
    /// Returns `None` if either vector is near zero or if the vectors are antiparallel ($180^\circ$).
    #[must_use]
    pub fn rotation_between(from: [T; 3], to: [T; 3]) -> Option<Self> {
        let na = from[0].hypot(from[1]).hypot(from[2]);
        let nb = to[0].hypot(to[1]).hypot(to[2]);
        let min_len = T::from_f64(1.0e-12);
        if na <= min_len || nb <= min_len {
            return None;
        }
        let a = [from[0] / na, from[1] / na, from[2] / na];
        let b = [to[0] / nb, to[1] / nb, to[2] / nb];
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        if dot < T::from_f64(-0.999_999) {
            return None;
        }
        let cx = a[1] * b[2] - a[2] * b[1];
        let cy = a[2] * b[0] - a[0] * b[2];
        let cz = a[0] * b[1] - a[1] * b[0];
        Self::try_from_quaternion(Quaternion::new(T::ONE + dot, cx, cy, cz)).ok()
    }

    /// Returns `true` when all four coordinates are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Re-projects `self` onto the unit 3-sphere $S^3$ (or `identity` if zero).
    #[must_use]
    pub fn renormalized(self) -> Self {
        Self::from_wxyz_lossy(self.to_wxyz())
    }

    /// Inverse rotation $q^{-1} = q^*$.
    #[inline]
    #[must_use]
    pub fn inv(self) -> Self {
        Self(self.0.conj())
    }

    /// Converts this unit quaternion into its row-major $3 \times 3$ rotation matrix $R \in \operatorname{SO}(3)$.
    #[must_use]
    pub fn to_rotation_matrix(self) -> [[T; 3]; 3] {
        let q = self.renormalized().0;
        let two = T::TWO;
        let xx = two * q.x * q.x;
        let yy = two * q.y * q.y;
        let zz = two * q.z * q.z;
        let xy = two * q.x * q.y;
        let xz = two * q.x * q.z;
        let yz = two * q.y * q.z;
        let wx = two * q.w * q.x;
        let wy = two * q.w * q.y;
        let wz = two * q.w * q.z;
        [
            [T::ONE - (yy + zz), xy - wz, xz + wy],
            [xy + wz, T::ONE - (xx + zz), yz - wx],
            [xz - wy, yz + wx, T::ONE - (xx + yy)],
        ]
    }

    /// Rotates a 3D vector $\mathbf{v}$ via $q \mathbf{v} q^*$.
    #[must_use]
    pub fn rotate_vec3(self, v: [T; 3]) -> [T; 3] {
        let q = self.0;
        let two = T::TWO;
        let tx = two * (q.y * v[2] - q.z * v[1]);
        let ty = two * (q.z * v[0] - q.x * v[2]);
        let tz = two * (q.x * v[1] - q.y * v[0]);
        [
            v[0] + q.w * tx + (q.y * tz - q.z * ty),
            v[1] + q.w * ty + (q.z * tx - q.x * tz),
            v[2] + q.w * tz + (q.x * ty - q.y * tx),
        ]
    }

    /// Casts coordinates to single-precision `UnitQuaternion<f32>` and renormalizes.
    #[must_use]
    pub fn to_f32(self) -> UnitQuaternion<f32> {
        let [w, x, y, z] = self.to_wxyz();
        UnitQuaternion::from_wxyz_lossy([
            w.to_f64() as f32,
            x.to_f64() as f32,
            y.to_f64() as f32,
            z.to_f64() as f32,
        ])
    }

    /// Casts coordinates to double-precision `UnitQuaternion<f64>` and renormalizes.
    #[must_use]
    pub fn to_f64(self) -> UnitQuaternion<f64> {
        let [w, x, y, z] = self.to_wxyz();
        UnitQuaternion::from_wxyz_lossy([w.to_f64(), x.to_f64(), y.to_f64(), z.to_f64()])
    }
}

impl<T: RealScalar> Mul for UnitQuaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0).renormalized()
    }
}

impl<T: fmt::Display> fmt::Display for UnitQuaternion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
