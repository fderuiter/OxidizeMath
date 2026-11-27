// [cite:engine_pde]
use math_commons::error::MathError;
use math_commons::math_kernel::types::StepSize;

/// 2D Biharmonic and characteristic upwind finite-difference operators.
pub mod biharmonic;
/// Boundary-aware 1D, 2D, and 3D Laplacian, gradient, and upwind grid evaluators.
pub mod eval;

#[cfg(feature = "serde")]
mod const_f64_array {
    use serde::ser::SerializeTuple;
    use serde::{Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer, const N: usize>(
        arr: &[f64; N],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(N)?;
        for &val in arr {
            tup.serialize_element(&val)?;
        }
        tup.end()
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>, const N: usize>(
        deserializer: D,
    ) -> Result<[f64; N], D::Error> {
        struct ArrayVisitor<const M: usize>;

        impl<'de, const M: usize> serde::de::Visitor<'de> for ArrayVisitor<M> {
            type Value = [f64; M];

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(formatter, "an array of {M} f64 values")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = [0.0_f64; M];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(out)
            }
        }

        deserializer.deserialize_tuple(N, ArrayVisitor::<N>)
    }
}

/// Six-neighbor 3D axial state values around a central grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StencilNeighbors3D<const N: usize> {
    /// Neighbor at `(x - 1, y, z)` (clamped at `x == 0`).
    #[cfg_attr(feature = "serde", serde(with = "const_f64_array"))]
    pub xm: [f64; N],
    /// Neighbor at `(x + 1, y, z)` (clamped at `x == width - 1`).
    #[cfg_attr(feature = "serde", serde(with = "const_f64_array"))]
    pub xp: [f64; N],
    /// Neighbor at `(x, y - 1, z)` (clamped at `y == 0`).
    #[cfg_attr(feature = "serde", serde(with = "const_f64_array"))]
    pub ym: [f64; N],
    /// Neighbor at `(x, y + 1, z)` (clamped at `y == height - 1`).
    #[cfg_attr(feature = "serde", serde(with = "const_f64_array"))]
    pub yp: [f64; N],
    /// Neighbor at `(x, y, z - 1)` (clamped at `z == 0`).
    #[cfg_attr(feature = "serde", serde(with = "const_f64_array"))]
    pub zm: [f64; N],
    /// Neighbor at `(x, y, z + 1)` (clamped at `z == depth - 1`).
    #[cfg_attr(feature = "serde", serde(with = "const_f64_array"))]
    pub zp: [f64; N],
}

/// Standard 1D, 2D, and 3D spatial finite-difference operators (Laplacian, gradient, upwind, and 2D Biharmonic).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StencilOperators {
    /// Grid spacing $\Delta x$ along the primary axis.
    pub dx: StepSize,
    /// Grid spacing $\Delta y$ along the secondary axis.
    pub dy: StepSize,
    /// Grid spacing $\Delta z$ along the tertiary axis.
    pub dz: StepSize,
    /// Precomputed reciprocal $1 / \Delta x$.
    pub inv_dx: f64,
    /// Precomputed reciprocal $1 / \Delta y$.
    pub inv_dy: f64,
    /// Precomputed reciprocal $1 / (2 \Delta x)$.
    pub inv_two_dx: f64,
    /// Precomputed reciprocal $1 / \Delta x^2$.
    pub inv_dx_sq: f64,
    /// Precomputed reciprocal $1 / \Delta y^2$.
    pub inv_dy_sq: f64,
    /// Precomputed reciprocal $1 / \Delta z^2$.
    pub inv_dz_sq: f64,
}

impl StencilOperators {
    /// Creates 1D stencil operators directly from raw `f64` spacing `dx` in `#![no_std]` callers, returning `None` if `dx <= 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn try_new_1d_f64(dx: f64) -> Option<Self> {
        Self::try_new(StepSize(dx)).ok()
    }

    /// Creates stencil operators in `#![no_std]` contexts, returning `None` if any step is `<= 0` or non-finite.
    #[verified_engine::verified]
    pub fn try_new_checked(dx: StepSize, dy: StepSize, dz: StepSize) -> Option<Self> {
        Self::try_new_3d(dx, dy, dz).ok()
    }

    /// Creates 1D stencil operators after validating that `dx` is finite and strictly positive.
    #[verified_engine::verified]
    pub fn try_new(dx: StepSize) -> Result<Self, MathError> {
        Self::try_new_3d(dx, StepSize(1.0), StepSize(1.0))
    }

    /// Creates 2D stencil operators after validating that `dx` and `dy` are finite and strictly positive.
    #[verified_engine::verified]
    pub fn try_new_2d(dx: StepSize, dy: StepSize) -> Result<Self, MathError> {
        Self::try_new_3d(dx, dy, StepSize(1.0))
    }

    /// Creates 3D stencil operators after validating that `dx`, `dy`, and `dz` are finite and strictly positive.
    #[verified_engine::verified]
    pub fn try_new_3d(dx: StepSize, dy: StepSize, dz: StepSize) -> Result<Self, MathError> {
        for step in [*dx, *dy, *dz] {
            if !(step > 0.0 && step.is_finite()) {
                return Err(MathError::StrictlyOutOfBounds {
                    value: step,
                    min: 0.0,
                    max: f64::INFINITY,
                });
            }
        }
        debug_assert!(*dx > 0.0 && *dy > 0.0 && *dz > 0.0);
        Ok(Self {
            dx,
            dy,
            dz,
            inv_dx: 1.0 / *dx,
            inv_dy: 1.0 / *dy,
            inv_two_dx: 1.0 / (2.0 * *dx),
            inv_dx_sq: 1.0 / (*dx * *dx),
            inv_dy_sq: 1.0 / (*dy * *dy),
            inv_dz_sq: 1.0 / (*dz * *dz),
        })
    }

    /// Creates 1D stencil operators, panicking if `*dx <= 0.0` or non-finite; prefer [`Self::try_new`].
    #[verified_engine::verified]
    pub fn new(dx: StepSize) -> Self {
        Self::new_3d(dx, StepSize(1.0), StepSize(1.0))
    }

    /// Creates 2D stencil operators, panicking if `*dx <= 0.0` or `*dy <= 0.0`; prefer [`Self::try_new_2d`].
    #[verified_engine::verified]
    pub fn new_2d(dx: StepSize, dy: StepSize) -> Self {
        Self::new_3d(dx, dy, StepSize(1.0))
    }

    /// Precomputes 3D finite-difference reciprocal spacings for `(dx, dy, dz)`; prefer [`Self::try_new_3d`].
    #[verified_engine::verified]
    pub fn new_3d(dx: StepSize, dy: StepSize, dz: StepSize) -> Self {
        match Self::try_new_3d(dx, dy, dz) {
            Ok(ops) => ops,
            Err(_) => {
                assert!(
                    *dx > 0.0 && dx.is_finite(),
                    "Step size dx must be positive and non-zero"
                );
                assert!(
                    *dy > 0.0 && dy.is_finite(),
                    "Step size dy must be positive and non-zero"
                );
                panic!("Step size dz must be positive and non-zero");
            }
        }
    }

    /// Central difference for 1st derivative (1D gradient): `(next - prev) / (2 dx)`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_1st(&self, prev: f64, next: f64) -> f64 {
        debug_assert!(self.inv_two_dx > 0.0);
        (next - prev) * self.inv_two_dx
    }

    /// Central difference for 2nd derivative (1D Laplacian).
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_2nd(&self, prev: f64, curr: f64, next: f64) -> f64 {
        debug_assert!(self.inv_dx_sq > 0.0);
        (next - 2.0 * curr + prev) * self.inv_dx_sq
    }

    /// Central difference for 2nd derivative (5-point 2D Laplacian).
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_2nd_2d(&self, curr: f64, left: f64, right: f64, up: f64, down: f64) -> f64 {
        debug_assert!(self.inv_dx_sq > 0.0 && self.inv_dy_sq > 0.0);
        let d2x = (right - 2.0 * curr + left) * self.inv_dx_sq;
        let d2y = (down - 2.0 * curr + up) * self.inv_dy_sq;
        d2x + d2y
    }

    /// Central difference for 2nd derivative (7-point 3D Laplacian) for `[xm, xp, ym, yp, zm, zp]`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_2nd_3d(&self, curr: f64, neighbors: [f64; 6]) -> f64 {
        debug_assert!(self.inv_dx_sq > 0.0 && self.inv_dy_sq > 0.0 && self.inv_dz_sq > 0.0);
        let d2x = (neighbors[1] - 2.0 * curr + neighbors[0]) * self.inv_dx_sq;
        let d2y = (neighbors[3] - 2.0 * curr + neighbors[2]) * self.inv_dy_sq;
        let d2z = (neighbors[5] - 2.0 * curr + neighbors[4]) * self.inv_dz_sq;
        d2x + d2y + d2z
    }

    /// First-order upwind advection flux derivative $\partial_x (v u)$ for constant drift `v`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn upwind_flux(&self, v: f64, prev: f64, curr: f64, next: f64) -> f64 {
        debug_assert!(self.inv_dx > 0.0);
        if v > 0.0 {
            (curr * v - prev * v) * self.inv_dx
        } else {
            (next * v - curr * v) * self.inv_dx
        }
    }

    /// First-order conservative donor-cell flux divergence $(F_{i+1/2} - F_{i-1/2}) / \Delta x$
    /// for spatially varying interface velocities `v_left` ($v_{i-1/2}$) and `v_right` ($v_{i+1/2}$).
    ///
    /// $$
    /// F_{i+1/2} = \max(v_{i+1/2}, 0)\,m_i + \min(v_{i+1/2}, 0)\,m_{i+1}
    /// $$
    ///
    /// Reference: Achdou & Capuzzo-Dolcetta (2010), DOI: 10.1137/090758477;
    /// [`docs/math_specs/engine_pde.md`](../../../docs/math_specs/engine_pde.md).
    ///
    /// # Stability
    /// Monotone donor-cell flux splitting preserves non-negativity under explicit forward stepping
    /// when $\Delta t (\max(v_{i+1/2}, 0) - \min(v_{i-1/2}, 0)) \le \Delta x$ and conserves discrete
    /// mass $\sum_i m_i \Delta x$ by telescoping across interior interfaces.
    ///
    /// # Domain of Validity
    /// Requires positive spatial step $\Delta x > 0$ and finite interface velocities and cell densities.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn conservative_upwind_flux(
        &self,
        v_left: f64,
        v_right: f64,
        prev: f64,
        curr: f64,
        next: f64,
    ) -> f64 {
        debug_assert!(self.inv_dx > 0.0);
        let flux_left = v_left.max(0.0) * prev + v_left.min(0.0) * curr;
        let flux_right = v_right.max(0.0) * curr + v_right.min(0.0) * next;
        (flux_right - flux_left) * self.inv_dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_upwind_flux_compresses_uniform_density_and_conserves_mass() {
        let ops = StencilOperators::new(StepSize(0.25));
        // Constant velocity matches upwind_flux identically.
        assert!(
            (ops.conservative_upwind_flux(1.5, 1.5, 0.2, 0.6, 0.9)
                - ops.upwind_flux(1.5, 0.2, 0.6, 0.9))
            .abs()
                < 1e-15
        );
        assert!(
            (ops.conservative_upwind_flux(-1.5, -1.5, 0.2, 0.6, 0.9)
                - ops.upwind_flux(-1.5, 0.2, 0.6, 0.9))
            .abs()
                < 1e-15
        );

        // Uniform m0 = 0.5 under converging drift (v_left > 0 > v_right) yields negative divergence (-dm/dt < 0 => dm/dt > 0).
        let div_center = ops.conservative_upwind_flux(1.0, -1.0, 0.5, 0.5, 0.5);
        assert!((div_center - (-4.0)).abs() < 1e-15);

        // Telescoping sum over a closed 5-cell grid with v_{-1/2} = v_{4+1/2} = 0 vanishes to machine precision.
        let m = [0.5_f64; 5];
        let v_faces = [0.0, 0.8, 0.3, -0.4, -0.9, 0.0];
        let mut total_div = 0.0;
        for i in 0..5 {
            let prev = if i == 0 { m[0] } else { m[i - 1] };
            let next = if i == 4 { m[4] } else { m[i + 1] };
            total_div += ops.conservative_upwind_flux(v_faces[i], v_faces[i + 1], prev, m[i], next);
        }
        assert!(total_div.abs() < 1e-15);
    }
}
