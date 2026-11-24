// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use math_commons::math_kernel::types::StepSize;

/// Six-neighbor 3D axial state values around a central grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StencilNeighbors3D<const N: usize> {
    /// Neighbor at `(x - 1, y, z)` (clamped at `x == 0`).
    pub xm: [f64; N],
    /// Neighbor at `(x + 1, y, z)` (clamped at `x == width - 1`).
    pub xp: [f64; N],
    /// Neighbor at `(x, y - 1, z)` (clamped at `y == 0`).
    pub ym: [f64; N],
    /// Neighbor at `(x, y + 1, z)` (clamped at `y == height - 1`).
    pub yp: [f64; N],
    /// Neighbor at `(x, y, z - 1)` (clamped at `z == 0`).
    pub zm: [f64; N],
    /// Neighbor at `(x, y, z + 1)` (clamped at `z == depth - 1`).
    pub zp: [f64; N],
}

/// Standard 1D, 2D, and 3D spatial finite-difference operators (Laplacian, gradient, upwind, and 2D Biharmonic).
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Creates 1D stencil operators in `#![no_std]` contexts, returning `None` if `dx <= 0` or non-finite.
    #[verified_engine::verified]
    pub fn try_new_checked(dx: StepSize, dy: StepSize, dz: StepSize) -> Option<Self> {
        for step in [*dx, *dy, *dz] {
            if !(step > 0.0 && step.is_finite()) {
                return None;
            }
        }
        debug_assert!(*dx > 0.0 && *dy > 0.0 && *dz > 0.0);
        Some(Self {
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

    /// Creates 1D stencil operators after validating that `dx` is finite and strictly positive.
    #[cfg(feature = "std")]
    #[verified_engine::verified]
    pub fn try_new(dx: StepSize) -> Result<Self, math_commons::error::MathError> {
        Self::try_new_3d(dx, StepSize(1.0), StepSize(1.0))
    }

    /// Creates 2D stencil operators after validating that `dx` and `dy` are finite and strictly positive.
    #[cfg(feature = "std")]
    #[verified_engine::verified]
    pub fn try_new_2d(dx: StepSize, dy: StepSize) -> Result<Self, math_commons::error::MathError> {
        Self::try_new_3d(dx, dy, StepSize(1.0))
    }

    /// Creates 3D stencil operators after validating that `dx`, `dy`, and `dz` are finite and strictly positive.
    #[cfg(feature = "std")]
    #[verified_engine::verified]
    pub fn try_new_3d(
        dx: StepSize,
        dy: StepSize,
        dz: StepSize,
    ) -> Result<Self, math_commons::error::MathError> {
        for step in [*dx, *dy, *dz] {
            if !(step > 0.0 && step.is_finite()) {
                return Err(math_commons::error::MathError::StrictlyOutOfBounds {
                    value: step,
                    min: 0.0,
                    max: f64::INFINITY,
                });
            }
        }
        debug_assert!(*dx > 0.0 && *dy > 0.0 && *dz > 0.0);
        Ok(Self::try_new_checked(dx, dy, dz).expect("Step sizes validated positive"))
    }

    /// Creates 1D stencil operators, panicking if `*dx <= 0.0` or non-finite.
    #[verified_engine::verified]
    pub fn new(dx: StepSize) -> Self {
        Self::new_3d(dx, StepSize(1.0), StepSize(1.0))
    }

    /// Creates 2D stencil operators, panicking if `*dx <= 0.0` or `*dy <= 0.0`.
    #[verified_engine::verified]
    pub fn new_2d(dx: StepSize, dy: StepSize) -> Self {
        Self::new_3d(dx, dy, StepSize(1.0))
    }

    /// Precomputes 3D finite-difference reciprocal spacings for `(dx, dy, dz)`.
    #[verified_engine::verified]
    pub fn new_3d(dx: StepSize, dy: StepSize, dz: StepSize) -> Self {
        assert!(
            *dx > 0.0 && dx.is_finite(),
            "Step size dx must be positive and non-zero"
        );
        assert!(
            *dy > 0.0 && dy.is_finite(),
            "Step size dy must be positive and non-zero"
        );
        assert!(
            *dz > 0.0 && dz.is_finite(),
            "Step size dz must be positive and non-zero"
        );
        Self::try_new_checked(dx, dy, dz).expect("Step sizes validated positive")
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

    /// Evaluates the boundary-aware 1D Laplacian $\partial_{xx} u$ at index `x`.
    #[inline]
    #[verified_engine::verified]
    pub fn laplacian_1d_at(&self, u: &[f64], n: usize, x: usize, bc: BoundaryCondition) -> f64 {
        assert!(x < n && u.len() >= n);
        if bc.is_dirichlet() && (x == 0 || x + 1 == n) {
            return 0.0;
        }
        let (prev, curr, next) = bc.sample_1d(u, n, x);
        self.central_diff_2nd(prev, curr, next)
    }

    /// Evaluates the boundary-aware 2D Laplacian $\nabla^2 u$ at `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn laplacian_2d_at(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition,
    ) -> f64 {
        assert!(x < width && y < height && u.len() >= width * height);
        if bc.is_dirichlet() && (x == 0 || x + 1 == width || y == 0 || y + 1 == height) {
            return 0.0;
        }
        let (curr, left, right, up, down) = bc.sample_2d(u, width, height, x, y);
        self.central_diff_2nd_2d(curr, left, right, up, down)
    }

    /// Evaluates the boundary-aware 3D Laplacian $\nabla^2 u$ at `(x, y, z)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn laplacian_3d_at(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        bc: BoundaryCondition,
    ) -> f64 {
        assert!(x < width && y < height && z < depth && u.len() >= width * height * depth);
        if bc.is_dirichlet()
            && (x == 0 || x + 1 == width || y == 0 || y + 1 == height || z == 0 || z + 1 == depth)
        {
            return 0.0;
        }
        let (curr, nb) = bc.sample_3d(u, width, height, depth, x, y, z);
        self.central_diff_2nd_3d(curr, nb)
    }

    /// Evaluates the boundary-aware 1D central gradient $\partial_x u$ at `x`.
    #[inline]
    #[verified_engine::verified]
    pub fn gradient_1d_at(&self, u: &[f64], n: usize, x: usize, bc: BoundaryCondition) -> f64 {
        assert!(x < n && u.len() >= n);
        let (prev, _curr, next) = bc.sample_1d(u, n, x);
        self.central_diff_1st(prev, next)
    }

    /// Evaluates the boundary-aware 2D central gradient $(\partial_x u, \partial_y u)$ at `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn gradient_2d_at(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition,
    ) -> (f64, f64) {
        assert!(x < width && y < height && u.len() >= width * height);
        let (_curr, left, right, up, down) = bc.sample_2d(u, width, height, x, y);
        let gx = (right - left) * self.inv_two_dx;
        let gy = (down - up) / (2.0 * *self.dy);
        (gx, gy)
    }

    /// Evaluates the boundary-aware 3D central gradient $(\partial_x u, \partial_y u, \partial_z u)$ at `(x, y, z)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn gradient_3d_at(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        bc: BoundaryCondition,
    ) -> (f64, f64, f64) {
        assert!(x < width && y < height && z < depth && u.len() >= width * height * depth);
        let (_curr, nb) = bc.sample_3d(u, width, height, depth, x, y, z);
        let gx = (nb[1] - nb[0]) * self.inv_two_dx;
        let gy = (nb[3] - nb[2]) / (2.0 * *self.dy);
        let gz = (nb[5] - nb[4]) / (2.0 * *self.dz);
        (gx, gy, gz)
    }

    /// Evaluates $|\nabla u|^2$ in 1D at cell `x`.
    #[inline]
    #[verified_engine::verified]
    pub fn grad_sq_1d_at(&self, u: &[f64], n: usize, x: usize, bc: BoundaryCondition) -> f64 {
        assert!(x < n && u.len() >= n);
        let g = self.gradient_1d_at(u, n, x, bc);
        g * g
    }

    /// Evaluates $|\nabla u|^2 = (\partial_x u)^2 + (\partial_y u)^2$ in 2D at `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn grad_sq_2d_at(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition,
    ) -> f64 {
        assert!(x < width && y < height && u.len() >= width * height);
        let (gx, gy) = self.gradient_2d_at(u, width, height, x, y, bc);
        gx * gx + gy * gy
    }

    /// Evaluates 1D upwind advection $\vec{v} \cdot \nabla u$ at `x`.
    #[inline]
    #[verified_engine::verified]
    pub fn upwind_1d_at(
        &self,
        u: &[f64],
        n: usize,
        x: usize,
        vx: f64,
        bc: BoundaryCondition,
    ) -> f64 {
        assert!(x < n && u.len() >= n);
        let (prev, curr, next) = bc.sample_1d(u, n, x);
        self.upwind_flux(vx, prev, curr, next)
    }

    /// Evaluates 2D directional upwind advection $v_x \partial_x u + v_y \partial_y u$ at `(x, y)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn upwind_2d_at(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        vel: (f64, f64),
        bc: BoundaryCondition,
    ) -> f64 {
        assert!(x < width && y < height && u.len() >= width * height);
        let (curr, left, right, up, down) = bc.sample_2d(u, width, height, x, y);
        let fx = self.upwind_flux(vel.0, left, curr, right);
        let fy = if vel.1 > 0.0 {
            (curr - up) * vel.1 * self.inv_dy
        } else {
            (down - curr) * vel.1 * self.inv_dy
        };
        fx + fy
    }

    /// Evaluates 3D directional upwind advection $\vec{v} \cdot \nabla u$ at `(x, y, z)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[verified_engine::verified]
    pub fn upwind_3d_at(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        depth: usize,
        x: usize,
        y: usize,
        z: usize,
        vel: (f64, f64, f64),
        bc: BoundaryCondition,
    ) -> f64 {
        assert!(x < width && y < height && z < depth && u.len() >= width * height * depth);
        let (curr, nb) = bc.sample_3d(u, width, height, depth, x, y, z);
        let fx = self.upwind_flux(vel.0, nb[0], curr, nb[1]);
        let inv_dz = 1.0 / *self.dz;
        let fy = if vel.1 > 0.0 {
            (curr - nb[2]) * vel.1 * self.inv_dy
        } else {
            (nb[3] - curr) * vel.1 * self.inv_dy
        };
        let fz = if vel.2 > 0.0 {
            (curr - nb[4]) * vel.2 * inv_dz
        } else {
            (nb[5] - curr) * vel.2 * inv_dz
        };
        fx + fy + fz
    }
}
