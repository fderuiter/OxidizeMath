// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
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

    /// Populates `out_lap` with the discrete 2D Laplacian $\nabla^2 u$ across `width x height`.
    #[verified_engine::verified]
    pub fn compute_laplacian_2d(
        &self,
        u: &[f64],
        width: usize,
        height: usize,
        bc: BoundaryCondition,
        out_lap: &mut [f64],
    ) {
        let total = width * height;
        assert!(u.len() >= total && out_lap.len() >= total);
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                out_lap[idx] = self.laplacian_2d_at(u, width, height, x, y, bc);
            }
        }
    }

    /// Evaluates the 2D Biharmonic operator $\nabla^4 u = \nabla^2(\nabla^2 u)$ at `(x, y)`
    /// from a precomputed 2D Laplacian field `laplacian`.
    ///
    /// Used by both Kirchhoff-Love plate bending (`wave_2d`) and conserved Cahn-Hilliard SPDE steppers.
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
        assert!(x < width && y < height && laplacian.len() >= width * height);
        let on_edge = x == 0 || x + 1 == width || y == 0 || y + 1 == height;
        if on_edge && bc.is_dirichlet() {
            return 0.0;
        }
        let (curr, left, right, up, down) = bc.sample_2d(laplacian, width, height, x, y);
        self.central_diff_2nd_2d(curr, left, right, up, down)
    }

    /// Evaluates the 2D Biharmonic operator $\nabla^4 u(x, y)$ directly from `u` via the 13-point stencil
    /// without requiring an external Laplacian buffer.
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
        assert!(x < width && y < height && u.len() >= width * height);
        let on_edge = x == 0 || x + 1 == width || y == 0 || y + 1 == height;
        if on_edge && bc.is_dirichlet() {
            return 0.0;
        }
        let (xm, xp) = neighbor_coords(bc, x, width);
        let (ym, yp) = neighbor_coords(bc, y, height);
        let lap_c = self.laplacian_2d_at(u, width, height, x, y, bc);
        let lap_l = neighbor_laplacian(self, u, width, height, xm, Some(y), bc);
        let lap_r = neighbor_laplacian(self, u, width, height, xp, Some(y), bc);
        let lap_u = neighbor_laplacian(self, u, width, height, Some(x), ym, bc);
        let lap_d = neighbor_laplacian(self, u, width, height, Some(x), yp, bc);
        self.central_diff_2nd_2d(lap_c, lap_l, lap_r, lap_u, lap_d)
    }
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
) -> f64 {
    match (x, y) {
        (Some(xi), Some(yi)) => ops.laplacian_2d_at(u, width, height, xi, yi, bc),
        _ => 0.0,
    }
}
