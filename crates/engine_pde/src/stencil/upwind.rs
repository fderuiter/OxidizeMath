// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::error::PdeError;
use crate::stencil::StencilOperators;
use crate::stencil::eval::{check_1d, check_2d, check_3d};
use math_scalars::RealField;

impl<T: RealField> StencilOperators<T> {
    /// Fallibly evaluates 1D upwind advection $\vec{v} \cdot \nabla u$ at `x`.
    #[inline]
    #[verified_engine::verified]
    pub fn try_upwind_1d_at(
        &self,
        u: &[T],
        n: usize,
        x: usize,
        vx: T,
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        check_1d(u, n, x)?;
        let (prev, curr, next) = bc.sample_1d(u, n, x);
        debug_assert!(x < n && u.len() >= n);
        Ok(self.upwind_flux(vx, prev, curr, next))
    }

    /// Evaluates 1D upwind advection $\vec{v} \cdot \nabla u$ at `x`; prefer [`Self::try_upwind_1d_at`].
    #[inline]
    #[verified_engine::verified]
    pub fn upwind_1d_at(&self, u: &[T], n: usize, x: usize, vx: T, bc: BoundaryCondition<T>) -> T {
        self.try_upwind_1d_at(u, n, x, vx, bc)
            .expect("StencilOperators::upwind_1d_at preconditions violated")
    }

    /// Fallibly evaluates 2D directional upwind advection $v_x \partial_x u + v_y \partial_y u$ at `(x, y)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn try_upwind_2d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        vel: (T, T),
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        check_2d(u, width, height, x, y)?;
        let (curr, left, right, up, down) = bc.sample_2d(u, width, height, x, y);
        let fx = self.upwind_flux(vel.0, left, curr, right);
        let fy = if vel.1 > T::ZERO {
            (curr - up) * vel.1 * self.inv_dy
        } else {
            (down - curr) * vel.1 * self.inv_dy
        };
        debug_assert!(x < width && y < height && *self.dy > T::ZERO);
        Ok(fx + fy)
    }

    /// Evaluates 2D directional upwind advection $v_x \partial_x u + v_y \partial_y u$ at `(x, y)`; prefer [`Self::try_upwind_2d_at`].
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn upwind_2d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        vel: (T, T),
        bc: BoundaryCondition<T>,
    ) -> T {
        self.try_upwind_2d_at(u, width, height, x, y, vel, bc)
            .expect("StencilOperators::upwind_2d_at preconditions violated")
    }

    /// Fallibly evaluates 3D directional upwind advection $\vec{v} \cdot \nabla u$ at `(x, y, z)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn try_upwind_3d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        vel: (T, T, T),
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        check_3d(u, width, height, depth, x, y, z)?;
        let (curr, nb) = bc.sample_3d(u, width, height, depth, x, y, z);
        let fx = self.upwind_flux(vel.0, nb[0], curr, nb[1]);
        let inv_dz = T::ONE / *self.dz;
        let fy = if vel.1 > T::ZERO {
            (curr - nb[2]) * vel.1 * self.inv_dy
        } else {
            (nb[3] - curr) * vel.1 * self.inv_dy
        };
        let fz = if vel.2 > T::ZERO {
            (curr - nb[4]) * vel.2 * inv_dz
        } else {
            (nb[5] - curr) * vel.2 * inv_dz
        };
        debug_assert!(x < width && y < height && z < depth);
        Ok(fx + fy + fz)
    }

    /// Evaluates 3D directional upwind advection $\vec{v} \cdot \nabla u$ at `(x, y, z)`; prefer [`Self::try_upwind_3d_at`].
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn upwind_3d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        vel: (T, T, T),
        bc: BoundaryCondition<T>,
    ) -> T {
        self.try_upwind_3d_at(u, width, height, depth, x, y, z, vel, bc)
            .expect("StencilOperators::upwind_3d_at preconditions violated")
    }
}
