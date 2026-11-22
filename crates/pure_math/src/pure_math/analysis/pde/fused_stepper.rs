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
        let inv_dx_sq = 1.0 / (*dx * *dx);
        let inv_dy_sq = 1.0 / (*dy * *dy);
        Self {
            dx,
            dy,
            inv_dx,
            inv_dy,
            inv_two_dx,
            inv_dx_sq,
            inv_dy_sq,
        }
    }

    /// Central difference for 1st derivative (gradient): (u_{i+1} - u_{i-1}) / 2dx
    #[inline(always)]
    #[verified_engine::verified]
    pub fn central_diff_1st(&self, prev: f64, next: f64) -> f64 {
        (next - prev) * self.inv_two_dx
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

    /// Upwind scheme for advection flux: evaluates d/dx (v * m) or similar upwind terms.
    /// `v` is the drift velocity. Returns the first derivative approximation based on upwind direction.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn upwind_flux(&self, v: f64, prev: f64, curr: f64, next: f64) -> f64 {
        if v > 0.0 {
            (curr * v - prev * v) * self.inv_dx
        } else {
            (next * v - curr * v) * self.inv_dx
        }
    }

    /// Positivity-preserving 2D first-order upwind advective operator (`u_x * dc/dx + u_y * dc/dy`).
    ///
    /// Subtracting `dt * ops.upwind_2d((ux, uy), curr, left, right, up, down)` in forward-Euler
    /// transport preserves non-negativity whenever `|ux| * dt * inv_dx + |uy| * dt * inv_dy <= 1.0`.
    #[inline(always)]
    #[verified_engine::verified]
    pub fn upwind_2d(
        &self,
        velocity: (f64, f64),
        curr: f64,
        left: f64,
        right: f64,
        up: f64,
        down: f64,
    ) -> f64 {
        let (ux, uy) = velocity;
        let adv_x = if ux >= 0.0 {
            ux * (curr - left) * self.inv_dx
        } else {
            ux * (right - curr) * self.inv_dx
        };
        let adv_y = if uy >= 0.0 {
            uy * (curr - up) * self.inv_dy
        } else {
            uy * (down - curr) * self.inv_dy
        };
        adv_x + adv_y
    }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stencil_operators_precomputation() {
        let ops = StencilOperators::new_2d(StepSize(0.5), StepSize(0.25));
        assert!((ops.inv_dx - 2.0).abs() < 1e-12);
        assert!((ops.inv_dy - 4.0).abs() < 1e-12);
        assert!((ops.inv_two_dx - 1.0).abs() < 1e-12);
        assert!((ops.inv_dx_sq - 4.0).abs() < 1e-12);
        assert!((ops.inv_dy_sq - 16.0).abs() < 1e-12);

        let ops_1d = StencilOperators::new(StepSize(2.0));
        assert!((ops_1d.inv_dx - 0.5).abs() < 1e-12);
        assert!((ops_1d.inv_dy - 1.0).abs() < 1e-12);
        assert!((ops_1d.inv_two_dx - 0.25).abs() < 1e-12);
        assert!((ops_1d.inv_dx_sq - 0.25).abs() < 1e-12);
        assert!((ops_1d.inv_dy_sq - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "Step size dx must be positive and non-zero")]
    fn test_stencil_operators_invalid_dx() {
        StencilOperators::new(StepSize(0.0));
    }

    #[test]
    #[should_panic(expected = "Step size dy must be positive and non-zero")]
    fn test_stencil_operators_invalid_dy() {
        StencilOperators::new_2d(StepSize(1.0), StepSize(-0.5));
    }

    #[test]
    fn test_stencil_operator_evaluations() {
        let dx = 0.2;
        let dy = 0.4;
        let ops = StencilOperators::new_2d(StepSize(dx), StepSize(dy));

        let cd1 = ops.central_diff_1st(1.0, 3.0);
        let cd1_expected = (3.0 - 1.0) / (2.0 * dx);
        assert!((cd1 - cd1_expected).abs() < 1e-12);

        let cd2 = ops.central_diff_2nd(1.0, 4.0, 9.0);
        let cd2_expected = (9.0 - 2.0 * 4.0 + 1.0) / (dx * dx);
        assert!((cd2 - cd2_expected).abs() < 1e-12);

        let cd2_2d = ops.central_diff_2nd_2d(5.0, 2.0, 8.0, 3.0, 7.0);
        let d2x_expected = (8.0 - 2.0 * 5.0 + 2.0) / (dx * dx);
        let d2y_expected = (7.0 - 2.0 * 5.0 + 3.0) / (dy * dy);
        assert!((cd2_2d - (d2x_expected + d2y_expected)).abs() < 1e-12);

        let flux_pos = ops.upwind_flux(2.0, 1.0, 3.0, 5.0);
        let flux_pos_expected = (3.0 * 2.0 - 1.0 * 2.0) / dx;
        assert!((flux_pos - flux_pos_expected).abs() < 1e-12);

        let flux_neg = ops.upwind_flux(-2.0, 1.0, 3.0, 5.0);
        let flux_neg_expected = (5.0 * (-2.0) - 3.0 * (-2.0)) / dx;
        assert!((flux_neg - flux_neg_expected).abs() < 1e-12);

        let up2d_pos = ops.upwind_2d((0.5, 0.25), 4.0, 1.0, 9.0, 2.0, 8.0);
        let expected_pos = 0.5 * (4.0 - 1.0) / dx + 0.25 * (4.0 - 2.0) / dy;
        assert!((up2d_pos - expected_pos).abs() < 1e-12);

        let up2d_neg = ops.upwind_2d((-0.5, -0.25), 4.0, 1.0, 9.0, 2.0, 8.0);
        let expected_neg = -0.5 * (9.0 - 4.0) / dx + (-0.25) * (8.0 - 4.0) / dy;
        assert!((up2d_neg - expected_neg).abs() < 1e-12);

        // Positivity under CFL <= 1: spike c=1 surrounded by zeros stays >= 0
        let dt = 0.08; // |0.5|*0.08/0.2 + |0.25|*0.08/0.4 = 0.20 + 0.05 = 0.25 <= 1.0
        let next_spike = 1.0 - dt * ops.upwind_2d((0.5, -0.25), 1.0, 0.0, 0.0, 0.0, 0.0);
        assert!(next_spike >= 0.0 && (next_spike - 0.75).abs() < 1e-12);
    }
}
