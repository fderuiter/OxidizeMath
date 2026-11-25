// [cite:engine_pde]
use math_commons::error::MathError;
use math_commons::math_kernel::types::StepSize;
use math_scalars::RealField;

/// 2D Biharmonic and characteristic upwind finite-difference operators.
pub mod biharmonic;
/// Boundary-aware 1D, 2D, and 3D Laplacian, gradient, and upwind grid evaluators.
pub mod eval;
mod upwind;

#[cfg(feature = "serde")]
mod const_scalar_array {
    use core::marker::PhantomData;
    use serde::ser::SerializeTuple;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer, T: Serialize, const N: usize>(
        arr: &[T; N],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(N)?;
        for val in arr {
            tup.serialize_element(val)?;
        }
        tup.end()
    }

    pub(super) fn deserialize<
        'de,
        D: Deserializer<'de>,
        T: Copy + Default + Deserialize<'de>,
        const N: usize,
    >(
        deserializer: D,
    ) -> Result<[T; N], D::Error> {
        struct ArrayVisitor<T, const M: usize>(PhantomData<T>);

        impl<'de, T: Copy + Default + Deserialize<'de>, const M: usize> serde::de::Visitor<'de>
            for ArrayVisitor<T, M>
        {
            type Value = [T; M];

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(formatter, "an array of {M} scalar values")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = [T::default(); M];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(out)
            }
        }

        deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    }
}

/// Six-neighbor 3D axial state values around a central grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: Copy + Default + serde::Deserialize<'de>"
    ))
)]
pub struct StencilNeighbors3D<const N: usize, T = f64> {
    /// Neighbor at `(x - 1, y, z)` (clamped at `x == 0`).
    #[cfg_attr(feature = "serde", serde(with = "const_scalar_array"))]
    pub xm: [T; N],
    /// Neighbor at `(x + 1, y, z)` (clamped at `x == width - 1`).
    #[cfg_attr(feature = "serde", serde(with = "const_scalar_array"))]
    pub xp: [T; N],
    /// Neighbor at `(x, y - 1, z)` (clamped at `y == 0`).
    #[cfg_attr(feature = "serde", serde(with = "const_scalar_array"))]
    pub ym: [T; N],
    /// Neighbor at `(x, y + 1, z)` (clamped at `y == height - 1`).
    #[cfg_attr(feature = "serde", serde(with = "const_scalar_array"))]
    pub yp: [T; N],
    /// Neighbor at `(x, y, z - 1)` (clamped at `z == 0`).
    #[cfg_attr(feature = "serde", serde(with = "const_scalar_array"))]
    pub zm: [T; N],
    /// Neighbor at `(x, y, z + 1)` (clamped at `z == depth - 1`).
    #[cfg_attr(feature = "serde", serde(with = "const_scalar_array"))]
    pub zp: [T; N],
}

/// Standard 1D, 2D, and 3D spatial finite-difference operators (Laplacian, gradient, upwind, and 2D Biharmonic).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StencilOperators<T = f64> {
    /// Grid spacing $\Delta x$ along the primary axis.
    pub dx: StepSize<T>,
    /// Grid spacing $\Delta y$ along the secondary axis.
    pub dy: StepSize<T>,
    /// Grid spacing $\Delta z$ along the tertiary axis.
    pub dz: StepSize<T>,
    /// Precomputed reciprocal $1 / \Delta x$.
    pub inv_dx: T,
    /// Precomputed reciprocal $1 / \Delta y$.
    pub inv_dy: T,
    /// Precomputed reciprocal $1 / (2 \Delta x)$.
    pub inv_two_dx: T,
    /// Precomputed reciprocal $1 / \Delta x^2$.
    pub inv_dx_sq: T,
    /// Precomputed reciprocal $1 / \Delta y^2$.
    pub inv_dy_sq: T,
    /// Precomputed reciprocal $1 / \Delta z^2$.
    pub inv_dz_sq: T,
}

impl<T: RealField> StencilOperators<T> {
    /// Creates stencil operators in `#![no_std]` contexts, returning `None` if any step is `<= 0` or non-finite.
    #[verified_engine::verified]
    pub fn try_new_checked(dx: StepSize<T>, dy: StepSize<T>, dz: StepSize<T>) -> Option<Self> {
        Self::try_new_3d(dx, dy, dz).ok()
    }

    /// Creates 1D stencil operators after validating that `dx` is finite and strictly positive.
    #[verified_engine::verified]
    pub fn try_new(dx: StepSize<T>) -> Result<Self, MathError> {
        Self::try_new_3d(dx, StepSize(T::ONE), StepSize(T::ONE))
    }

    /// Creates 2D stencil operators after validating that `dx` and `dy` are finite and strictly positive.
    #[verified_engine::verified]
    pub fn try_new_2d(dx: StepSize<T>, dy: StepSize<T>) -> Result<Self, MathError> {
        Self::try_new_3d(dx, dy, StepSize(T::ONE))
    }

    /// Creates 3D stencil operators after validating that `dx`, `dy`, and `dz` are finite and strictly positive.
    #[verified_engine::verified]
    pub fn try_new_3d(
        dx: StepSize<T>,
        dy: StepSize<T>,
        dz: StepSize<T>,
    ) -> Result<Self, MathError> {
        for step in [*dx, *dy, *dz] {
            if !(step > T::ZERO && step.is_finite()) {
                return Err(MathError::StrictlyOutOfBounds {
                    value: step.to_f64(),
                    min: 0.0,
                    max: f64::INFINITY,
                });
            }
        }
        debug_assert!(*dx > T::ZERO && *dy > T::ZERO && *dz > T::ZERO);
        Ok(Self {
            dx,
            dy,
            dz,
            inv_dx: T::ONE / *dx,
            inv_dy: T::ONE / *dy,
            inv_two_dx: T::ONE / (T::TWO * *dx),
            inv_dx_sq: T::ONE / (*dx * *dx),
            inv_dy_sq: T::ONE / (*dy * *dy),
            inv_dz_sq: T::ONE / (*dz * *dz),
        })
    }

    /// Creates 1D stencil operators, panicking if `*dx <= 0.0` or non-finite; prefer [`Self::try_new`].
    #[verified_engine::verified]
    pub fn new(dx: StepSize<T>) -> Self {
        Self::new_3d(dx, StepSize(T::ONE), StepSize(T::ONE))
    }

    /// Creates 2D stencil operators, panicking if `*dx <= 0.0` or `*dy <= 0.0`; prefer [`Self::try_new_2d`].
    #[verified_engine::verified]
    pub fn new_2d(dx: StepSize<T>, dy: StepSize<T>) -> Self {
        Self::new_3d(dx, dy, StepSize(T::ONE))
    }

    /// Precomputes 3D finite-difference reciprocal spacings for `(dx, dy, dz)`; prefer [`Self::try_new_3d`].
    #[verified_engine::verified]
    pub fn new_3d(dx: StepSize<T>, dy: StepSize<T>, dz: StepSize<T>) -> Self {
        match Self::try_new_3d(dx, dy, dz) {
            Ok(ops) => ops,
            Err(_) => {
                assert!(
                    *dx > T::ZERO && dx.is_finite(),
                    "Step size dx must be positive and non-zero"
                );
                assert!(
                    *dy > T::ZERO && dy.is_finite(),
                    "Step size dy must be positive and non-zero"
                );
                panic!("Step size dz must be positive and non-zero");
            }
        }
    }

    /// Central difference for 1st derivative (1D gradient): `(next - prev) / (2 dx)`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_1st(&self, prev: T, next: T) -> T {
        debug_assert!(self.inv_two_dx > T::ZERO);
        (next - prev) * self.inv_two_dx
    }

    /// Central difference for 2nd derivative (1D Laplacian).
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_2nd(&self, prev: T, curr: T, next: T) -> T {
        debug_assert!(self.inv_dx_sq > T::ZERO);
        (next - T::TWO * curr + prev) * self.inv_dx_sq
    }

    /// Central difference for 2nd derivative (5-point 2D Laplacian).
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_2nd_2d(&self, curr: T, left: T, right: T, up: T, down: T) -> T {
        debug_assert!(self.inv_dx_sq > T::ZERO && self.inv_dy_sq > T::ZERO);
        let d2x = (right - T::TWO * curr + left) * self.inv_dx_sq;
        let d2y = (down - T::TWO * curr + up) * self.inv_dy_sq;
        d2x + d2y
    }

    /// Central difference for 2nd derivative (7-point 3D Laplacian) for `[xm, xp, ym, yp, zm, zp]`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_2nd_3d(&self, curr: T, neighbors: [T; 6]) -> T {
        debug_assert!(
            self.inv_dx_sq > T::ZERO && self.inv_dy_sq > T::ZERO && self.inv_dz_sq > T::ZERO
        );
        let d2x = (neighbors[1] - T::TWO * curr + neighbors[0]) * self.inv_dx_sq;
        let d2y = (neighbors[3] - T::TWO * curr + neighbors[2]) * self.inv_dy_sq;
        let d2z = (neighbors[5] - T::TWO * curr + neighbors[4]) * self.inv_dz_sq;
        d2x + d2y + d2z
    }

    /// First-order upwind advection flux derivative $\partial_x (v u)$ for constant drift `v`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn upwind_flux(&self, v: T, prev: T, curr: T, next: T) -> T {
        debug_assert!(self.inv_dx > T::ZERO);
        if v > T::ZERO {
            (curr * v - prev * v) * self.inv_dx
        } else {
            (next * v - curr * v) * self.inv_dx
        }
    }
}
