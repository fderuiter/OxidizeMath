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
        let inv_two_dx = 1.0 / (2.0 * *dx);
        let inv_dx_sq = 1.0 / (*dx * *dx);
        let inv_dy_sq = 1.0 / (*dy * *dy);
        Self {
            dx,
            dy,
            inv_dx,
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

    /// Applies a 2D fused stencil pass over multiple coupled arrays with periodic (toroidal) boundaries.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d_coupled_periodic<const N: usize, F>(
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
        if width == 0 || height == 0 {
            return;
        }

        let get_state = |idx: usize| {
            let mut val = [0.0; N];
            for s in 0..N {
                if idx < src[s].len() {
                    val[s] = src[s][idx];
                }
            }
            val
        };

        for y in 0..height {
            let y_u = if y > 0 { y - 1 } else { height - 1 };
            let y_d = if y + 1 < height { y + 1 } else { 0 };
            for x in 0..width {
                let x_l = if x > 0 { x - 1 } else { width - 1 };
                let x_r = if x + 1 < width { x + 1 } else { 0 };

                let idx = flatten_2d_index(x, y, width);
                let curr = get_state(idx);
                let left = get_state(flatten_2d_index(x_l, y, width));
                let right = get_state(flatten_2d_index(x_r, y, width));
                let up = get_state(flatten_2d_index(x, y_u, width));
                let down = get_state(flatten_2d_index(x, y_d, width));

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
    #[verified_engine::verified]
    fn test_step_2d_coupled_periodic_mass_conservation_and_wrap() {
        let (width, height) = (8_usize, 8_usize);
        let n = width * height;
        let stepper = FusedStencilStepper::new_2d(StepSize(1.0), StepSize(1.0));
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        // Place impulse at corner (0, 0) so flux must wrap to (width - 1, 0) and (0, height - 1)
        u[flatten_2d_index(0, 0, width)] = 4.0;
        v[flatten_2d_index(width - 1, height - 1, width)] = 2.0;

        let initial_u_mass: f64 = u.iter().sum();
        let initial_v_mass: f64 = v.iter().sum();
        let mut next_u = vec![0.0; n];
        let mut next_v = vec![0.0; n];

        for _ in 0..20 {
            stepper.step_2d_coupled_periodic(
                (width, height),
                [u.as_slice(), v.as_slice()],
                [next_u.as_mut_slice(), next_v.as_mut_slice()],
                0.1,
                1.0,
                |_i, curr, left, right, up, down, ops| {
                    [
                        0.5 * ops.central_diff_2nd_2d(curr[0], left[0], right[0], up[0], down[0]),
                        0.25 * ops.central_diff_2nd_2d(curr[1], left[1], right[1], up[1], down[1]),
                    ]
                },
            );
            std::mem::swap(&mut u, &mut next_u);
            std::mem::swap(&mut v, &mut next_v);
        }

        let final_u_mass: f64 = u.iter().sum();
        let final_v_mass: f64 = v.iter().sum();
        assert!((final_u_mass - initial_u_mass).abs() < 1e-12);
        assert!((final_v_mass - initial_v_mass).abs() < 1e-12);
        // Wrapped neighbors (7, 0) and (1, 0) must receive identical symmetric flux from (0, 0)
        let left_wrap = u[flatten_2d_index(width - 1, 0, width)];
        let right_inner = u[flatten_2d_index(1, 0, width)];
        assert!((left_wrap - right_inner).abs() < 1e-12);
        assert!(left_wrap > 0.0);
    }

    #[test]
    fn test_stencil_operators_precomputation() {
        let ops = StencilOperators::new_2d(StepSize(0.5), StepSize(0.25));
        assert!((ops.inv_dx - 2.0).abs() < 1e-12);
        assert!((ops.inv_two_dx - 1.0).abs() < 1e-12);
        assert!((ops.inv_dx_sq - 4.0).abs() < 1e-12);
        assert!((ops.inv_dy_sq - 16.0).abs() < 1e-12);

        let ops_1d = StencilOperators::new(StepSize(2.0));
        assert!((ops_1d.inv_dx - 0.5).abs() < 1e-12);
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
    }
}
