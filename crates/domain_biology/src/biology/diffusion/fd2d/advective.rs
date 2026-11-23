//! Zero-allocation 2D advective-diffusive stencil operator with internal zero-flux solid masks.

use super::{FiniteDifference2D, SpatialDiffusion};
use crate::biology::diffusion::fd2d::geometry::GeometryStrategy;
use crate::biology::morphogenesis::reaction::ReactionKinetics;
use math_commons::math_kernel::types::flatten_2d_index;
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

/// 2D finite-difference diffusion + first-order conservative upwind advection strategy
/// with internal zero-flux solid obstacle masks and preallocated row-major buffers.
#[derive(Debug, Clone)]
pub struct AdvectiveFiniteDifference2D {
    /// Underlying Cartesian 2D 5-point stencil geometry and outer Neumann boundary.
    pub base: FiniteDifference2D,
    /// Dimensionless Péclet advection gain scaling fluid velocity transport.
    pub peclet: f64,
    ux: Vec<f64>,
    uy: Vec<f64>,
    obstacles: Vec<bool>,
}

impl AdvectiveFiniteDifference2D {
    /// Creates an advective-diffusive 2D operator with zero initial velocity and no solid obstacles.
    #[verified_engine::verified]
    pub fn new(base: FiniteDifference2D, peclet: f64) -> Self {
        let len = base.geometry.size();
        Self {
            base,
            peclet,
            ux: vec![0.0; len],
            uy: vec![0.0; len],
            obstacles: vec![false; len],
        }
    }

    /// Returns the grid width in cells.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        *self.base.geometry.width
    }

    /// Returns the grid height in cells.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        *self.base.geometry.height
    }

    /// Returns total cell count (`width * height`).
    #[verified_engine::verified]
    pub fn grid_len(&self) -> usize {
        self.ux.len()
    }

    /// Returns the row-major horizontal fluid velocity slice (`flatten_2d_index(x, y, width)`).
    #[verified_engine::verified]
    pub fn velocity_x(&self) -> &[f64] {
        &self.ux
    }

    /// Returns the row-major vertical fluid velocity slice (`flatten_2d_index(x, y, width)`).
    #[verified_engine::verified]
    pub fn velocity_y(&self) -> &[f64] {
        &self.uy
    }

    /// Returns the row-major solid obstacle mask (`true` = impermeable zero-flux solid).
    #[verified_engine::verified]
    pub fn obstacles(&self) -> &[bool] {
        &self.obstacles
    }

    /// Borrows mutable `(ux, uy, obstacles)` row-major buffers simultaneously for zero-allocation copying.
    #[verified_engine::verified]
    pub fn velocity_and_obstacles_mut(&mut self) -> (&mut [f64], &mut [f64], &mut [bool]) {
        (&mut self.ux, &mut self.uy, &mut self.obstacles)
    }

    /// Copies `(ux, uy, obstacles)` from caller slices without reallocating.
    #[verified_engine::verified]
    pub fn set_velocity_and_obstacles(&mut self, ux: &[f64], uy: &[f64], obstacles: &[bool]) {
        let n = self.grid_len();
        assert_eq!(ux.len(), n, "ux slice length mismatch");
        assert_eq!(uy.len(), n, "uy slice length mismatch");
        assert_eq!(obstacles.len(), n, "obstacles slice length mismatch");
        self.ux.copy_from_slice(ux);
        self.uy.copy_from_slice(uy);
        self.obstacles.copy_from_slice(obstacles);
    }

    /// Sets velocity and obstacle flag at `(x, y)`.
    #[verified_engine::verified]
    pub fn set_cell_flow(&mut self, x: usize, y: usize, vx: f64, vy: f64, is_obstacle: bool) {
        let w = self.width();
        let h = self.height();
        if x < w && y < h {
            let idx = flatten_2d_index(x, y, w);
            self.obstacles[idx] = is_obstacle;
            self.ux[idx] = if is_obstacle { 0.0 } else { vx };
            self.uy[idx] = if is_obstacle { 0.0 } else { vy };
        }
    }

    /// Resets velocity buffers to zero and clears all solid obstacles.
    #[verified_engine::verified]
    pub fn clear_flow_and_obstacles(&mut self) {
        self.ux.fill(0.0);
        self.uy.fill(0.0);
        self.obstacles.fill(false);
    }

    /// Computes the combined 2D explicit diffusive + upwind advective CFL time-step bound.
    #[verified_engine::verified]
    pub fn max_stable_dt(&self, diffusion_coeffs: &[f64], safety: f64) -> f64 {
        let clamped_safety = safety.clamp(0.05, 0.95);
        let d_max = diffusion_coeffs
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(0.0_f64, f64::max);
        let inv_dx = 1.0 / *self.base.geometry.dx;
        let inv_dy = 1.0 / *self.base.geometry.dy;
        let diff_rate = 2.0 * d_max * (inv_dx * inv_dx + inv_dy * inv_dy);

        let pe = if self.peclet.is_finite() {
            self.peclet.abs()
        } else {
            0.0
        };
        let mut max_ux = 0.0_f64;
        let mut max_uy = 0.0_f64;
        if pe > 0.0 {
            for i in 0..self.ux.len() {
                if !self.obstacles[i] {
                    max_ux = max_ux.max(self.ux[i].abs());
                    max_uy = max_uy.max(self.uy[i].abs());
                }
            }
        }
        let adv_rate = pe * (max_ux * inv_dx + max_uy * inv_dy);
        let total_rate = diff_rate + adv_rate;
        if total_rate <= 1e-12 {
            1.0
        } else {
            clamped_safety / total_rate
        }
    }

    #[inline(always)]
    #[verified_engine::verified]
    fn face_velocities(&self, idx: usize, face_open: [bool; 4]) -> [f64; 4] {
        let pe = self.peclet;
        if pe == 0.0 {
            return [0.0; 4];
        }
        let w = *self.base.geometry.width;
        [
            if face_open[0] {
                0.5 * pe * (self.ux[idx - 1] + self.ux[idx])
            } else {
                0.0
            },
            if face_open[1] {
                0.5 * pe * (self.ux[idx] + self.ux[idx + 1])
            } else {
                0.0
            },
            if face_open[2] {
                0.5 * pe * (self.uy[idx - w] + self.uy[idx])
            } else {
                0.0
            },
            if face_open[3] {
                0.5 * pe * (self.uy[idx] + self.uy[idx + w])
            } else {
                0.0
            },
        ]
    }
}

impl<const N: usize> SpatialDiffusion<N> for AdvectiveFiniteDifference2D {
    fn stepper(&self) -> FusedStencilStepper {
        FusedStencilStepper::new_2d(self.base.geometry.dx, self.base.geometry.dy)
    }

    #[verified_engine::verified]
    fn step_fused<K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        let w = self.width();
        let h = self.height();
        if N == 0 || w * h == 0 {
            return;
        }
        SpatialDiffusion::<N>::stepper(self).step_2d_coupled_masked_neumann(
            (w, h),
            &self.obstacles,
            state,
            next_state,
            dt,
            1.0,
            |idx, curr, nb, ops| {
                let face_vel = self.face_velocities(idx, nb.face_open);
                let rates = kinetics.reaction(curr);
                let mut rhs = [0.0; N];
                for s in 0..N {
                    let lap = ops.central_diff_2nd_2d(
                        curr[s],
                        nb.left[s],
                        nb.right[s],
                        nb.up[s],
                        nb.down[s],
                    );
                    let adv = ops.upwind_divergence_2d(
                        curr[s],
                        [nb.left[s], nb.right[s], nb.up[s], nb.down[s]],
                        face_vel,
                    );
                    rhs[s] = coeffs[s] * lap - adv + rates[s];
                }
                rhs
            },
        );
    }

    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], mut op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        let w = self.width();
        let h = self.height();
        let n = w * h;
        if N == 0 || n == 0 {
            return;
        }
        for species in state.iter().take(N) {
            assert!(
                species.len() >= n,
                "Buffer too small for AdvectiveFiniteDifference2D"
            );
        }
        let ops = SpatialDiffusion::<N>::stepper(self).ops;
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                let mut curr = [0.0; N];
                for s in 0..N {
                    curr[s] = state[s][idx];
                }
                if self.obstacles[idx] {
                    op(idx, curr, [0.0; N]);
                    continue;
                }
                let w_open = x > 0 && !self.obstacles[idx - 1];
                let e_open = x + 1 < w && !self.obstacles[idx + 1];
                let n_open = y > 0 && !self.obstacles[idx - w];
                let s_open = y + 1 < h && !self.obstacles[idx + w];
                let face_vel = self.face_velocities(idx, [w_open, e_open, n_open, s_open]);
                let mut transport = [0.0; N];
                for s in 0..N {
                    let left = if w_open { state[s][idx - 1] } else { curr[s] };
                    let right = if e_open { state[s][idx + 1] } else { curr[s] };
                    let up = if n_open { state[s][idx - w] } else { curr[s] };
                    let down = if s_open { state[s][idx + w] } else { curr[s] };
                    let lap = ops.central_diff_2nd_2d(curr[s], left, right, up, down);
                    let adv = ops.upwind_divergence_2d(curr[s], [left, right, up, down], face_vel);
                    transport[s] = coeffs[s] * lap - adv;
                }
                op(idx, curr, transport);
            }
        }
    }
}
