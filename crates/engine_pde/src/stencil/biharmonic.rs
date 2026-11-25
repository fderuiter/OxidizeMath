// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::error::PdeError;
use crate::stencil::StencilOperators;
use math_commons::math_kernel::types::{Scalar, flatten_2d_index};

impl StencilOperators {
    /// Advances one CFL-safe semi-Lagrangian characteristic upwind step (`dt <= dx`) for
    /// $\partial_t u + \partial_x u = -\mu(x) u$ with inflow boundary `boundary_inflow`.
    #[verified_engine::verified]
    pub fn step_characteristic_upwind_1d(
        &self,
        src: &[f64],
        damping_rate: &[f64],
        boundary_inflow: f64,
        dt: f64,
        dst: &mut [f64],
    ) {
        let n = src.len().min(damping_rate.len()).min(dst.len());
        if n == 0 {
            return;
        }
        debug_assert!(*self.dx > 0.0);
        let lambda = if dt.is_finite() {
            (dt * self.inv_dx).clamp(0.0, 1.0)
        } else {
            0.0
        };
        dst[0] = sanitize_nonnegative(boundary_inflow);
        let dt_pos = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        for i in 1..n {
            let u_curr = sanitize_nonnegative(src[i]);
            let u_prev = sanitize_nonnegative(src[i - 1]);
            let foot = (1.0 - lambda) * u_curr + lambda * u_prev;
            let mu = sanitize_nonnegative(damping_rate[i]);
            let survival = <f64 as Scalar>::exp(-mu * dt_pos);
            dst[i] = sanitize_nonnegative(foot * survival);
        }
    }

    /// Fallibly populates `out_lap` with the discrete 2D Laplacian $\nabla^2 u$ across `width x height`.
    ///
    /// # Errors
    /// Returns [`PdeError`] if `width * height` overflows or `u`/`out_lap` has fewer than `width * height` cells.
    #[verified_engine::verified]
    pub fn try_compute_laplacian_2d(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        bc: BoundaryCondition,
        out_lap: &mut [f64],
    ) -> Result<(), PdeError> {
        let total = width
            .checked_mul(height)
            .ok_or(PdeError::InvalidGridDimension {
                axis: "grid",
                size: usize::MAX,
            })?;
        if u.len() < total {
            return Err(PdeError::BufferTooSmall {
                buffer: "u",
                required: total,
                actual: u.len(),
            });
        }
        if out_lap.len() < total {
            return Err(PdeError::BufferTooSmall {
                buffer: "out_lap",
                required: total,
                actual: out_lap.len(),
            });
        }
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                out_lap[idx] = self.try_laplacian_2d_at(u, width, height, x, y, bc)?;
            }
        }
        debug_assert!(out_lap.len() >= total);
        Ok(())
    }

    /// Populates `out_lap` with the discrete 2D Laplacian $\nabla^2 u$ across `width x height`.
    ///
    /// # Panics
    /// Panics if `u` or `out_lap` is shorter than `width * height`; prefer [`Self::try_compute_laplacian_2d`].
    #[verified_engine::verified]
    pub fn compute_laplacian_2d(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        bc: BoundaryCondition,
        out_lap: &mut [f64],
    ) {
        self.try_compute_laplacian_2d(u, width, height, bc, out_lap)
            .expect("compute_laplacian_2d preconditions violated");
    }

    /// Fallibly evaluates the 2D Biharmonic operator $\nabla^4 u = \nabla^2(\nabla^2 u)$ at `(x, y)`
    /// from a precomputed 2D Laplacian field `laplacian`.
    ///
    /// # Errors
    /// Returns [`PdeError`] if `(x, y)` is out of bounds or `laplacian` is shorter than `width * height`.
    #[inline]
    #[verified_engine::verified]
    pub fn try_eval_biharmonic_2d_from_laplacian(
        &self,
        laplacian: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition,
    ) -> Result<f64, PdeError> {
        validate_2d_cell(laplacian, "laplacian", width, height, x, y)?;
        let on_edge = x == 0 || x + 1 == width || y == 0 || y + 1 == height;
        if on_edge && bc.is_dirichlet() {
            return Ok(0.0);
        }
        let (curr, left, right, up, down) = bc.sample_2d(laplacian, width, height, x, y);
        debug_assert!(x < width && y < height);
        Ok(self.central_diff_2nd_2d(curr, left, right, up, down))
    }

    /// Evaluates the 2D Biharmonic operator $\nabla^4 u = \nabla^2(\nabla^2 u)$ at `(x, y)`
    /// from a precomputed 2D Laplacian field `laplacian`.
    ///
    /// # Panics
    /// Panics if `(x, y)` is out of bounds or `laplacian` is undersized;
    /// prefer [`Self::try_eval_biharmonic_2d_from_laplacian`].
    #[inline]
    #[verified_engine::verified]
    pub fn eval_biharmonic_2d_from_laplacian(
        &self,
        laplacian: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition,
    ) -> f64 {
        self.try_eval_biharmonic_2d_from_laplacian(laplacian, width, height, x, y, bc)
            .expect("eval_biharmonic_2d_from_laplacian preconditions violated")
    }

    /// Fallibly evaluates the 2D Biharmonic operator $\nabla^4 u(x, y)$ directly from `u` via the 13-point stencil.
    ///
    /// # Errors
    /// Returns [`PdeError`] if `(x, y)` is out of bounds or `u` is shorter than `width * height`.
    #[verified_engine::verified]
    pub fn try_eval_biharmonic_2d(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition,
    ) -> Result<f64, PdeError> {
        validate_2d_cell(u, "u", width, height, x, y)?;
        let on_edge = x == 0 || x + 1 == width || y == 0 || y + 1 == height;
        if on_edge && bc.is_dirichlet() {
            return Ok(0.0);
        }
        let (xm, xp) = neighbor_coords(bc, x, width);
        let (ym, yp) = neighbor_coords(bc, y, height);
        let lap_c = self.try_laplacian_2d_at(u, width, height, x, y, bc)?;
        let lap_l = neighbor_laplacian(self, u, width, height, xm, Some(y), bc)?;
        let lap_r = neighbor_laplacian(self, u, width, height, xp, Some(y), bc)?;
        let lap_u = neighbor_laplacian(self, u, width, height, Some(x), ym, bc)?;
        let lap_d = neighbor_laplacian(self, u, width, height, Some(x), yp, bc)?;
        debug_assert!(x < width && y < height);
        Ok(self.central_diff_2nd_2d(lap_c, lap_l, lap_r, lap_u, lap_d))
    }

    /// Evaluates the 2D Biharmonic operator $\nabla^4 u(x, y)$ directly from `u` via the 13-point stencil
    /// without requiring an external Laplacian buffer.
    ///
    /// # Panics
    /// Panics if `(x, y)` is out of bounds or `u` is undersized; prefer [`Self::try_eval_biharmonic_2d`].
    #[verified_engine::verified]
    pub fn eval_biharmonic_2d(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        x: usize,
        y: usize,
        bc: BoundaryCondition,
    ) -> f64 {
        self.try_eval_biharmonic_2d(u, width, height, x, y, bc)
            .expect("eval_biharmonic_2d preconditions violated")
    }
}

#[inline]
#[verified_engine::verified]
fn validate_2d_cell(
    buf: &[f64],
    buffer: &'static str,
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
    let total = width
        .checked_mul(height)
        .ok_or(PdeError::InvalidGridDimension {
            axis: "grid",
            size: usize::MAX,
        })?;
    if buf.len() < total {
        return Err(PdeError::BufferTooSmall {
            buffer,
            required: total,
            actual: buf.len(),
        });
    }
    debug_assert!(x < width && y < height && buf.len() >= total);
    Ok(())
}

#[inline]
fn sanitize_nonnegative(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 { v } else { 0.0 }
}

#[inline]
fn neighbor_coords(bc: BoundaryCondition, i: usize, len: usize) -> (Option<usize>, Option<usize>) {
    if len <= 1 {
        return match bc {
            BoundaryCondition::Dirichlet(_) => (None, None),
            _ => (Some(0), Some(0)),
        };
    }
    match bc {
        BoundaryCondition::Periodic => (
            Some(if i > 0 { i - 1 } else { len - 1 }),
            Some(if i + 1 < len { i + 1 } else { 0 }),
        ),
        BoundaryCondition::Neumann => (Some(i.saturating_sub(1)), Some((i + 1).min(len - 1))),
        BoundaryCondition::SymmetricNeumann => (
            Some(if i > 0 { i - 1 } else { 1 }),
            Some(if i + 1 < len { i + 1 } else { len - 2 }),
        ),
        BoundaryCondition::Dirichlet(_) => (
            if i > 0 { Some(i - 1) } else { None },
            if i + 1 < len { Some(i + 1) } else { None },
        ),
    }
}

#[inline]
fn neighbor_laplacian(
    ops: &StencilOperators,
    u: &[f64],
    width: usize,
    height: usize,
    x: Option<usize>,
    y: Option<usize>,
    bc: BoundaryCondition,
) -> Result<f64, PdeError> {
    match (x, y) {
        (Some(xi), Some(yi)) => ops.try_laplacian_2d_at(u, width, height, xi, yi, bc),
        _ => Ok(0.0),
    }
}
