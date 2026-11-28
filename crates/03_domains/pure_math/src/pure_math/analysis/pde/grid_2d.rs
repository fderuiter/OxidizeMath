//! Local 2D uniform-grid finite-difference spacing and stencil helpers for `pure_math` PDEs.

use math_commons::error::MathError;
use math_commons::math_kernel::types::{StepSize, flatten_2d_index};

/// Precomputed 2D spatial grid spacings and reciprocal finite-difference weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GridSpacing2D {
    /// Grid spacing $\Delta x$ along the primary axis.
    pub dx: StepSize,
    /// Grid spacing $\Delta y$ along the secondary axis.
    pub dy: StepSize,
    /// Precomputed reciprocal $1 / \Delta x$.
    pub inv_dx: f64,
    /// Precomputed reciprocal $1 / \Delta y$.
    pub inv_dy: f64,
    /// Precomputed reciprocal $1 / \Delta x^2$.
    pub inv_dx_sq: f64,
    /// Precomputed reciprocal $1 / \Delta y^2$.
    pub inv_dy_sq: f64,
}

impl GridSpacing2D {
    /// Creates 2D grid spacing weights after validating that `dx` and `dy` are finite and strictly positive.
    #[verified_engine::verified]
    pub(crate) fn try_new_2d(dx: StepSize, dy: StepSize) -> Result<Self, MathError> {
        for step in [*dx, *dy] {
            if !(step > 0.0 && step.is_finite()) {
                return Err(MathError::StrictlyOutOfBounds {
                    value: step,
                    min: 0.0,
                    max: f64::INFINITY,
                });
            }
        }
        debug_assert!(*dx > 0.0 && *dy > 0.0);
        Ok(Self {
            dx,
            dy,
            inv_dx: 1.0 / *dx,
            inv_dy: 1.0 / *dy,
            inv_dx_sq: 1.0 / (*dx * *dx),
            inv_dy_sq: 1.0 / (*dy * *dy),
        })
    }

    /// Evaluates the 5-point 2D Laplacian $(u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/\Delta x^2 + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/\Delta y^2$.
    #[inline(always)]
    #[verified_engine::verified]
    pub(crate) fn central_diff_2nd_2d(
        &self,
        curr: f64,
        left: f64,
        right: f64,
        up: f64,
        down: f64,
    ) -> f64 {
        debug_assert!(self.inv_dx_sq > 0.0 && self.inv_dy_sq > 0.0);
        let d2x = (right - 2.0 * curr + left) * self.inv_dx_sq;
        let d2y = (down - 2.0 * curr + up) * self.inv_dy_sq;
        d2x + d2y
    }

    /// Evaluates $\nabla^4 u = \nabla^2(\nabla^2 u)$ at `(x, y)` from a precomputed 2D Laplacian buffer.
    #[inline]
    #[verified_engine::verified]
    pub(crate) fn eval_biharmonic_2d_from_laplacian(
        &self,
        laplacian: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        symmetric_neumann: bool,
    ) -> f64 {
        if width == 0 || height == 0 || x >= width || y >= height {
            return 0.0;
        }
        let curr_lap = laplacian[flatten_2d_index(x, y, width)];
        if !symmetric_neumann {
            if x == 0 || x + 1 >= width || y == 0 || y + 1 >= height {
                return 0.0;
            }
            let left = laplacian[flatten_2d_index(x - 1, y, width)];
            let right = laplacian[flatten_2d_index(x + 1, y, width)];
            let up = laplacian[flatten_2d_index(x, y - 1, width)];
            let down = laplacian[flatten_2d_index(x, y + 1, width)];
            return self.central_diff_2nd_2d(curr_lap, left, right, up, down);
        }
        let xm = if x > 0 { x - 1 } else { usize::from(width > 1) };
        let xp = if x + 1 < width {
            x + 1
        } else {
            width.saturating_sub(2)
        };
        let ym = if y > 0 {
            y - 1
        } else {
            usize::from(height > 1)
        };
        let yp = if y + 1 < height {
            y + 1
        } else {
            height.saturating_sub(2)
        };
        let left = laplacian[flatten_2d_index(xm, y, width)];
        let right = laplacian[flatten_2d_index(xp, y, width)];
        let up = laplacian[flatten_2d_index(x, ym, width)];
        let down = laplacian[flatten_2d_index(x, yp, width)];
        self.central_diff_2nd_2d(curr_lap, left, right, up, down)
    }
}
