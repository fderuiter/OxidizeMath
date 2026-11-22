use math_commons::math_kernel::types::{StepSize, flatten_2d_index};

/// Standard spatial numerical operators for fused stencil evaluations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StencilOperators {
    #[allow(missing_docs)]
    pub dx: StepSize,
    #[allow(missing_docs)]
    pub dy: StepSize,
    #[allow(missing_docs)]
    pub inv_dx: f64,
    #[allow(missing_docs)]
    pub inv_dy: f64,
    #[allow(missing_docs)]
    pub inv_two_dx: f64,
    #[allow(missing_docs)]
    pub inv_two_dy: f64,
    #[allow(missing_docs)]
    pub inv_dx_sq: f64,
    #[allow(missing_docs)]
    pub inv_dy_sq: f64,
}

impl StencilOperators {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(dx: StepSize) -> Self {
        Self::new_2d(dx, StepSize(1.0))
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new_2d(dx: StepSize, dy: StepSize) -> Self {
        assert!(*dx > 0.0, "Step size dx must be positive and non-zero");
        assert!(*dy > 0.0, "Step size dy must be positive and non-zero");
        let inv_dx = 1.0 / *dx;
        let inv_dy = 1.0 / *dy;
        let inv_two_dx = 1.0 / (2.0 * *dx);
        let inv_two_dy = 1.0 / (2.0 * *dy);
        let inv_dx_sq = 1.0 / (*dx * *dx);
        let inv_dy_sq = 1.0 / (*dy * *dy);
        Self {
            dx,
            dy,
            inv_dx,
            inv_dy,
            inv_two_dx,
            inv_two_dy,
            inv_dx_sq,
            inv_dy_sq,
        }
    }

    /// Central difference for 1st derivative in X: `(next - prev) / (2 * dx)`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_1st(&self, prev: f64, next: f64) -> f64 {
        (next - prev) * self.inv_two_dx
    }

    /// Central difference for 1st derivative in Y: `(down - up) / (2 * dy)`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_1st_y(&self, up: f64, down: f64) -> f64 {
        (down - up) * self.inv_two_dy
    }

    /// Central difference for 2nd derivative (Laplacian) in 1D.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_2nd(&self, prev: f64, curr: f64, next: f64) -> f64 {
        (next - 2.0 * curr + prev) * self.inv_dx_sq
    }

    /// Central difference for 2nd derivative (Laplacian) in 2D.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_2nd_2d(&self, curr: f64, left: f64, right: f64, up: f64, down: f64) -> f64 {
        let d2x = (right - 2.0 * curr + left) * self.inv_dx_sq;
        let d2y = (down - 2.0 * curr + up) * self.inv_dy_sq;
        d2x + d2y
    }

    /// Upwind scheme for 1D X-advection flux: evaluates `d/dx (v * c)` with donor-cell selection.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn upwind_flux(&self, v: f64, prev: f64, curr: f64, next: f64) -> f64 {
        if v > 0.0 {
            (curr * v - prev * v) * self.inv_dx
        } else {
            (next * v - curr * v) * self.inv_dx
        }
    }

    /// Upwind scheme for 1D Y-advection flux: evaluates `d/dy (vy * c)` with donor-cell selection.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn upwind_flux_y(&self, vy: f64, up: f64, curr: f64, down: f64) -> f64 {
        if vy > 0.0 {
            (curr * vy - up * vy) * self.inv_dy
        } else {
            (down * vy - curr * vy) * self.inv_dy
        }
    }

    /// First-order 2D upwind directional advection operator `vx * dc/dx + vy * dc/dy`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn upwind_flux_2d(
        &self,
        vx: f64,
        vy: f64,
        curr: f64,
        left: f64,
        right: f64,
        up: f64,
        down: f64,
    ) -> f64 {
        self.upwind_flux(vx, left, curr, right) + self.upwind_flux_y(vy, up, curr, down)
    }

    /// Conservative 2D donor-cell flux divergence across `[west, east, up, down]` cell faces.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn upwind_divergence_2d(
        &self,
        face_vel: [f64; 4],
        curr: f64,
        left: f64,
        right: f64,
        up: f64,
        down: f64,
    ) -> f64 {
        let flux_w = if face_vel[0] >= 0.0 {
            face_vel[0] * left
        } else {
            face_vel[0] * curr
        };
        let flux_e = if face_vel[1] >= 0.0 {
            face_vel[1] * curr
        } else {
            face_vel[1] * right
        };
        let flux_n = if face_vel[2] >= 0.0 {
            face_vel[2] * up
        } else {
            face_vel[2] * curr
        };
        let flux_s = if face_vel[3] >= 0.0 {
            face_vel[3] * curr
        } else {
            face_vel[3] * down
        };
        (flux_e - flux_w) * self.inv_dx + (flux_s - flux_n) * self.inv_dy
    }
}

/// Computes `(idx, left, right, up, down)` row-major indices with zero-flux reflection at domain
/// edges and solid `obstacles` cells.
#[inline(always)]
#[verified_engine::verified]
pub fn reflect_2d_neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    obstacles: &[bool],
) -> (usize, usize, usize, usize, usize) {
    let idx = flatten_2d_index(x, y, width);
    let cand_l = if x > 0 { idx - 1 } else { idx };
    let cand_r = if x + 1 < width { idx + 1 } else { idx };
    let cand_u = if y > 0 { idx - width } else { idx };
    let cand_d = if y + 1 < height { idx + width } else { idx };
    let is_obs = |i: usize| i < obstacles.len() && obstacles[i];
    let idx_l = if is_obs(cand_l) { idx } else { cand_l };
    let idx_r = if is_obs(cand_r) { idx } else { cand_r };
    let idx_u = if is_obs(cand_u) { idx } else { cand_u };
    let idx_d = if is_obs(cand_d) { idx } else { cand_d };
    (idx, idx_l, idx_r, idx_u, idx_d)
}

/// Computes `[west, east, up, down]` normal face velocities, enforcing zero normal velocity at
/// reflected domain boundaries and solid obstacle interfaces.
#[inline(always)]
#[verified_engine::verified]
pub fn advection_face_velocities(
    idx: usize,
    idx_l: usize,
    idx_r: usize,
    idx_u: usize,
    idx_d: usize,
    vx: &[f64],
    vy: &[f64],
) -> [f64; 4] {
    let ux_c = if idx < vx.len() { vx[idx] } else { 0.0 };
    let uy_c = if idx < vy.len() { vy[idx] } else { 0.0 };
    let vw = if idx_l == idx || idx_l >= vx.len() {
        0.0
    } else {
        0.5 * (vx[idx_l] + ux_c)
    };
    let ve = if idx_r == idx || idx_r >= vx.len() {
        0.0
    } else {
        0.5 * (ux_c + vx[idx_r])
    };
    let vn = if idx_u == idx || idx_u >= vy.len() {
        0.0
    } else {
        0.5 * (vy[idx_u] + uy_c)
    };
    let vs = if idx_d == idx || idx_d >= vy.len() {
        0.0
    } else {
        0.5 * (uy_c + vy[idx_d])
    };
    [vw, ve, vn, vs]
}

/// A generic Fused Stencil Stepper for performance-optimized grid updates.
/// By evaluating operators inside the grid loop, this avoids memory bandwidth bottlenecks
/// caused by temporary derivative buffers.
pub struct FusedStencilStepper {
    #[allow(missing_docs)]
    pub ops: StencilOperators,
}

impl FusedStencilStepper {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(dx: StepSize) -> Self {
        Self {
            ops: StencilOperators::new(dx),
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new_2d(dx: StepSize, dy: StepSize) -> Self {
        Self {
            ops: StencilOperators::new_2d(dx, dy),
        }
    }

    /// Applies a 1D fused stencil pass over a single array.
    /// `dir`: +1.0 for forward-time, -1.0 for backward-time equations.
    /// Boundaries are not updated and must be handled by the caller.
    #[verified_engine::verified]
    pub fn step_1d_slice<F>(&self, src: &[f64], dst: &mut [f64], dt: f64, dir: f64, mut op: F)
    where
        F: FnMut(usize, f64, f64, f64, &StencilOperators) -> f64,
    {
        let n = src.len();
        if n > 2 && dst.len() >= n {
            for (i, win) in src.windows(3).enumerate() {
                let prev = win[0];
                let curr = win[1];
                let next = win[2];
                let rhs = op(i + 1, prev, curr, next, &self.ops);
                dst[i + 1] = curr + dir * dt * rhs;
            }
        }
    }

    /// Applies a 1D fused stencil pass over multiple coupled arrays with Neumann boundaries.
    #[verified_engine::verified]
    pub fn step_1d_coupled_neumann<const N: usize, F>(
        &self,
        n: usize,
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(usize, [f64; N], [f64; N], [f64; N], &StencilOperators) -> [f64; N],
    {
        if n == 0 {
            return;
        }

        let get_state = |idx| {
            let mut val = [0.0; N];
            for s in 0..N {
                if idx < src[s].len() {
                    val[s] = src[s][idx];
                }
            }
            val
        };

        // Left boundary
        {
            let curr = get_state(0);
            let prev = curr; // Neumann
            let next = if n > 1 { get_state(1) } else { curr };
            let rhs = op(0, prev, curr, next, &self.ops);
            for s in 0..N {
                if !dst[s].is_empty() {
                    dst[s][0] = curr[s] + dir * dt * rhs[s];
                }
            }
        }

        // Interior
        if n > 2 {
            for i in 1..n - 1 {
                let prev = get_state(i - 1);
                let curr = get_state(i);
                let next = get_state(i + 1);
                let rhs = op(i, prev, curr, next, &self.ops);
                for s in 0..N {
                    if i < dst[s].len() {
                        dst[s][i] = curr[s] + dir * dt * rhs[s];
                    }
                }
            }
        }

        // Right boundary
        if n > 1 {
            let i = n - 1;
            let curr = get_state(i);
            let prev = get_state(i - 1);
            let next = curr; // Neumann
            let rhs = op(i, prev, curr, next, &self.ops);
            for s in 0..N {
                if i < dst[s].len() {
                    dst[s][i] = curr[s] + dir * dt * rhs[s];
                }
            }
        }
    }

    /// Applies a 2D fused stencil pass over multiple coupled arrays with Neumann boundaries.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d_coupled_neumann<const N: usize, F>(
        &self,
        dim: (usize, usize),
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(
            usize,
            [f64; N],
            [f64; N],
            [f64; N],
            [f64; N],
            [f64; N],
            &StencilOperators,
        ) -> [f64; N],
    {
        let (width, height) = dim;
        let n = width * height;
        if n == 0 {
            return;
        }

        let get_state = |idx| {
            let mut val = [0.0; N];
            for s in 0..N {
                if idx < src[s].len() {
                    val[s] = src[s][idx];
                }
            }
            val
        };

        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);

                let idx_l = if x > 0 { idx - 1 } else { idx };
                let idx_r = if x < width - 1 { idx + 1 } else { idx };
                let idx_u = if y > 0 { idx - width } else { idx };
                let idx_d = if y < height - 1 { idx + width } else { idx };

                let curr = get_state(idx);
                let left = get_state(idx_l);
                let right = get_state(idx_r);
                let up = get_state(idx_u);
                let down = get_state(idx_d);

                let rhs = op(idx, curr, left, right, up, down, &self.ops);

                for s in 0..N {
                    if idx < dst[s].len() {
                        dst[s][idx] = curr[s] + dir * dt * rhs[s];
                    }
                }
            }
        }
    }

    /// Applies a 2D fused advection-diffusion-reaction stencil pass with zero-flux reflection at
    /// domain boundaries and solid `obstacles` cells.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d_coupled_advection_neumann<const N: usize, F>(
        &self,
        dim: (usize, usize),
        src: [&[f64]; N],
        dst: [&mut [f64]; N],
        vx: &[f64],
        vy: &[f64],
        obstacles: &[bool],
        dt: f64,
        dir: f64,
        mut op: F,
    ) where
        F: FnMut(
            usize,
            [f64; 4],
            [f64; N],
            [f64; N],
            [f64; N],
            [f64; N],
            [f64; N],
            &StencilOperators,
        ) -> [f64; N],
    {
        let (width, height) = dim;
        if width == 0 || height == 0 {
            return;
        }
        let get_state = |idx| {
            let mut val = [0.0; N];
            for s in 0..N {
                if idx < src[s].len() {
                    val[s] = src[s][idx];
                }
            }
            val
        };
        for y in 0..height {
            for x in 0..width {
                let (idx, idx_l, idx_r, idx_u, idx_d) =
                    reflect_2d_neighbors(x, y, width, height, obstacles);
                let curr = get_state(idx);
                if idx < obstacles.len() && obstacles[idx] {
                    for s in 0..N {
                        if idx < dst[s].len() {
                            dst[s][idx] = curr[s];
                        }
                    }
                    continue;
                }
                let face_vel = advection_face_velocities(idx, idx_l, idx_r, idx_u, idx_d, vx, vy);
                let rhs = op(
                    idx,
                    face_vel,
                    curr,
                    get_state(idx_l),
                    get_state(idx_r),
                    get_state(idx_u),
                    get_state(idx_d),
                    &self.ops,
                );
                for s in 0..N {
                    if idx < dst[s].len() {
                        dst[s][idx] = curr[s] + dir * dt * rhs[s];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
