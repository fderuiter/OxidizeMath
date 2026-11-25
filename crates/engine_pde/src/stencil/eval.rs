// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::error::PdeError;
use crate::stencil::StencilOperators;
use math_scalars::RealField;

#[inline]
pub(super) fn check_1d<T>(u: &[T], n: usize, x: usize) -> Result<(), PdeError> {
    if x >= n {
        return Err(PdeError::IndexOutOfBounds {
            axis: "x",
            index: x,
            len: n,
        });
    }
    if u.len() < n {
        return Err(PdeError::BufferTooSmall {
            buffer: "u",
            required: n,
            actual: u.len(),
        });
    }
    Ok(())
}

#[inline]
pub(super) fn check_2d<T>(
    u: &[T],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> Result<(), PdeError> {
    if x >= width {
        return Err(PdeError::IndexOutOfBounds {
            axis: "x",
            index: x,
            len: width,
        });
    }
    if y >= height {
        return Err(PdeError::IndexOutOfBounds {
            axis: "y",
            index: y,
            len: height,
        });
    }
    let total = width * height;
    if u.len() < total {
        return Err(PdeError::BufferTooSmall {
            buffer: "u",
            required: total,
            actual: u.len(),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn check_3d<T>(
    u: &[T],
    width: usize,
    height: usize,
    depth: usize,
    x: usize,
    y: usize,
    z: usize,
) -> Result<(), PdeError> {
    if x >= width {
        return Err(PdeError::IndexOutOfBounds {
            axis: "x",
            index: x,
            len: width,
        });
    }
    if y >= height {
        return Err(PdeError::IndexOutOfBounds {
            axis: "y",
            index: y,
            len: height,
        });
    }
    if z >= depth {
        return Err(PdeError::IndexOutOfBounds {
            axis: "z",
            index: z,
            len: depth,
        });
    }
    let total = width * height * depth;
    if u.len() < total {
        return Err(PdeError::BufferTooSmall {
            buffer: "u",
            required: total,
            actual: u.len(),
        });
    }
    Ok(())
}

impl<T: RealField> StencilOperators<T> {
    /// Fallibly evaluates the boundary-aware 1D Laplacian $\partial_{xx} u$ at index `x`.
    #[inline]
    #[verified_engine::verified]
    pub fn try_laplacian_1d_at(
        &self,
        u: &[T],
        n: usize,
        x: usize,
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        check_1d(u, n, x)?;
        if bc.is_dirichlet() && (x == 0 || x + 1 == n) {
            return Ok(T::ZERO);
        }
        let (prev, curr, next) = bc.sample_1d(u, n, x);
        debug_assert!(x < n && u.len() >= n);
        Ok(self.central_diff_2nd(prev, curr, next))
    }

    /// Evaluates the boundary-aware 1D Laplacian $\partial_{xx} u$ at index `x`; prefer [`Self::try_laplacian_1d_at`].
    #[inline]
    #[verified_engine::verified]
    pub fn laplacian_1d_at(&self, u: &[T], n: usize, x: usize, bc: BoundaryCondition<T>) -> T {
        self.try_laplacian_1d_at(u, n, x, bc)
            .expect("StencilOperators::laplacian_1d_at preconditions violated")
    }

    /// Fallibly evaluates the boundary-aware 2D Laplacian $\nabla^2 u$ at `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn try_laplacian_2d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        check_2d(u, width, height, x, y)?;
        if bc.is_dirichlet() && (x == 0 || x + 1 == width || y == 0 || y + 1 == height) {
            return Ok(T::ZERO);
        }
        let (curr, left, right, up, down) = bc.sample_2d(u, width, height, x, y);
        debug_assert!(x < width && y < height);
        Ok(self.central_diff_2nd_2d(curr, left, right, up, down))
    }

    /// Evaluates the boundary-aware 2D Laplacian $\nabla^2 u$ at `(x, y)`; prefer [`Self::try_laplacian_2d_at`].
    #[inline]
    #[verified_engine::verified]
    pub fn laplacian_2d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition<T>,
    ) -> T {
        self.try_laplacian_2d_at(u, width, height, x, y, bc)
            .expect("StencilOperators::laplacian_2d_at preconditions violated")
    }

    /// Fallibly evaluates the boundary-aware 3D Laplacian $\nabla^2 u$ at `(x, y, z)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn try_laplacian_3d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        check_3d(u, width, height, depth, x, y, z)?;
        if bc.is_dirichlet()
            && (x == 0 || x + 1 == width || y == 0 || y + 1 == height || z == 0 || z + 1 == depth)
        {
            return Ok(T::ZERO);
        }
        let (curr, nb) = bc.sample_3d(u, width, height, depth, x, y, z);
        debug_assert!(x < width && y < height && z < depth);
        Ok(self.central_diff_2nd_3d(curr, nb))
    }

    /// Evaluates the boundary-aware 3D Laplacian $\nabla^2 u$ at `(x, y, z)`; prefer [`Self::try_laplacian_3d_at`].
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn laplacian_3d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        bc: BoundaryCondition<T>,
    ) -> T {
        self.try_laplacian_3d_at(u, width, height, depth, x, y, z, bc)
            .expect("StencilOperators::laplacian_3d_at preconditions violated")
    }

    /// Fallibly evaluates the boundary-aware 1D central gradient $\partial_x u$ at `x`.
    #[inline]
    #[verified_engine::verified]
    pub fn try_gradient_1d_at(
        &self,
        u: &[T],
        n: usize,
        x: usize,
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        check_1d(u, n, x)?;
        let (prev, _curr, next) = bc.sample_1d(u, n, x);
        debug_assert!(x < n && u.len() >= n);
        Ok(self.central_diff_1st(prev, next))
    }

    /// Evaluates the boundary-aware 1D central gradient $\partial_x u$ at `x`; prefer [`Self::try_gradient_1d_at`].
    #[inline]
    #[verified_engine::verified]
    pub fn gradient_1d_at(&self, u: &[T], n: usize, x: usize, bc: BoundaryCondition<T>) -> T {
        self.try_gradient_1d_at(u, n, x, bc)
            .expect("StencilOperators::gradient_1d_at preconditions violated")
    }

    /// Fallibly evaluates the boundary-aware 2D central gradient $(\partial_x u, \partial_y u)$ at `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn try_gradient_2d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition<T>,
    ) -> Result<(T, T), PdeError> {
        check_2d(u, width, height, x, y)?;
        let (_curr, left, right, up, down) = bc.sample_2d(u, width, height, x, y);
        let gx = (right - left) * self.inv_two_dx;
        let gy = (down - up) * T::HALF * self.inv_dy;
        debug_assert!(x < width && y < height && *self.dy > T::ZERO);
        Ok((gx, gy))
    }

    /// Evaluates the boundary-aware 2D central gradient $(\partial_x u, \partial_y u)$ at `(x, y)`; prefer [`Self::try_gradient_2d_at`].
    #[inline]
    #[verified_engine::verified]
    pub fn gradient_2d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition<T>,
    ) -> (T, T) {
        self.try_gradient_2d_at(u, width, height, x, y, bc)
            .expect("StencilOperators::gradient_2d_at preconditions violated")
    }

    /// Fallibly evaluates the boundary-aware 3D central gradient $(\partial_x u, \partial_y u, \partial_z u)$ at `(x, y, z)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn try_gradient_3d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        bc: BoundaryCondition<T>,
    ) -> Result<(T, T, T), PdeError> {
        check_3d(u, width, height, depth, x, y, z)?;
        let (_curr, nb) = bc.sample_3d(u, width, height, depth, x, y, z);
        let gx = (nb[1] - nb[0]) * self.inv_two_dx;
        let gy = (nb[3] - nb[2]) * T::HALF * self.inv_dy;
        let gz = (nb[5] - nb[4]) / (T::TWO * *self.dz);
        debug_assert!(*self.dy > T::ZERO && *self.dz > T::ZERO);
        Ok((gx, gy, gz))
    }

    /// Evaluates the boundary-aware 3D central gradient $(\partial_x u, \partial_y u, \partial_z u)$ at `(x, y, z)`; prefer [`Self::try_gradient_3d_at`].
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn gradient_3d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        bc: BoundaryCondition<T>,
    ) -> (T, T, T) {
        self.try_gradient_3d_at(u, width, height, depth, x, y, z, bc)
            .expect("StencilOperators::gradient_3d_at preconditions violated")
    }

    /// Fallibly evaluates $|\nabla u|^2$ in 1D at cell `x`.
    #[inline]
    #[verified_engine::verified]
    pub fn try_grad_sq_1d_at(
        &self,
        u: &[T],
        n: usize,
        x: usize,
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        let g = self.try_gradient_1d_at(u, n, x, bc)?;
        let sq = g * g;
        debug_assert!(!sq.is_finite() || sq >= T::ZERO);
        Ok(sq)
    }

    /// Evaluates $|\nabla u|^2$ in 1D at cell `x`; prefer [`Self::try_grad_sq_1d_at`].
    #[inline]
    #[verified_engine::verified]
    pub fn grad_sq_1d_at(&self, u: &[T], n: usize, x: usize, bc: BoundaryCondition<T>) -> T {
        self.try_grad_sq_1d_at(u, n, x, bc)
            .expect("StencilOperators::grad_sq_1d_at preconditions violated")
    }

    /// Fallibly evaluates $|\nabla u|^2 = (\partial_x u)^2 + (\partial_y u)^2$ in 2D at `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn try_grad_sq_2d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition<T>,
    ) -> Result<T, PdeError> {
        let (gx, gy) = self.try_gradient_2d_at(u, width, height, x, y, bc)?;
        let sq = gx * gx + gy * gy;
        debug_assert!(!sq.is_finite() || sq >= T::ZERO);
        Ok(sq)
    }

    /// Evaluates $|\nabla u|^2 = (\partial_x u)^2 + (\partial_y u)^2$ in 2D at `(x, y)`; prefer [`Self::try_grad_sq_2d_at`].
    #[inline]
    #[verified_engine::verified]
    pub fn grad_sq_2d_at(
        &self,
        u: &[T],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition<T>,
    ) -> T {
        self.try_grad_sq_2d_at(u, width, height, x, y, bc)
            .expect("StencilOperators::grad_sq_2d_at preconditions violated")
    }
}
