//! Zero-allocation 2D periodic Edwards-Wilkinson and Kardar-Parisi-Zhang heightfield solver
//! built on [`pure_math::pure_math::analysis::pde::fused_stepper::StencilOperators`] and
//! [`pure_math::pure_math::analysis::pde::spde::SpatialNoiseBuffer`].

// [cite:interface_growth]

use math_commons::math_kernel::types::{StepSize, flatten_2d_index};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::pde::fused_stepper::StencilOperators;
use pure_math::pure_math::analysis::pde::spde::SpatialNoiseBuffer;

use super::kpz_1d::{
    InterfaceGrowthParams, compute_mean, compute_rms_width, validate_interface_params,
};

/// Pre-allocated 2D periodic stochastic heightfield solver for EW ($\lambda = 0$) and KPZ ($\lambda \ne 0$).
#[derive(Clone, Debug)]
pub struct InterfaceGrowth2D {
    width: usize,
    height: usize,
    params: InterfaceGrowthParams,
    ops_xy: StencilOperators,
    ops_y: StencilOperators,
    heights: Vec<f64>,
    next_heights: Vec<f64>,
    noise_buf: SpatialNoiseBuffer,
    time: f64,
    steps: u64,
}

impl InterfaceGrowth2D {
    /// Constructs a flat ($h(x, y) = 0$) 2D periodic heightfield of size `width * height`.
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize, params: InterfaceGrowthParams) -> Self {
        assert!(
            width >= 4 && height >= 4,
            "2D interface grid dimensions must be >= 4x4"
        );
        validate_interface_params(&params);
        let step = StepSize(params.dx);
        let ops_xy = StencilOperators::new_2d(step, step);
        let ops_y = StencilOperators::new(step);
        let total = width * height;
        Self {
            width,
            height,
            params,
            ops_xy,
            ops_y,
            heights: vec![0.0; total],
            next_heights: vec![0.0; total],
            noise_buf: SpatialNoiseBuffer::new_2d(width, height),
            time: 0.0,
            steps: 0,
        }
    }

    /// Resets the 2D heightfield to $h(x, y) = 0$ and clears elapsed time.
    #[verified_engine::verified]
    pub fn reset_flat(&mut self) {
        self.heights.fill(0.0);
        self.next_heights.fill(0.0);
        self.time = 0.0;
        self.steps = 0;
    }

    /// Updates physical parameters and precomputed [`StencilOperators`] without heap allocation.
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: InterfaceGrowthParams) {
        validate_interface_params(&params);
        if (params.dx - self.params.dx).abs() > f64::EPSILON {
            let step = StepSize(params.dx);
            self.ops_xy = StencilOperators::new_2d(step, step);
            self.ops_y = StencilOperators::new(step);
        }
        self.params = params;
    }

    /// Grid width $W$.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height $H$.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Elapsed simulation time $t$.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Total integration steps completed since initialization or reset.
    #[verified_engine::verified]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Current solver parameters.
    #[verified_engine::verified]
    pub fn params(&self) -> InterfaceGrowthParams {
        self.params
    }

    /// Immutable slice of the row-major 2D heightfield $h(x, y, t)$.
    #[verified_engine::verified]
    pub fn heights(&self) -> &[f64] {
        &self.heights
    }

    /// Mutable slice of the row-major 2D heightfield.
    pub fn heights_mut(&mut self) -> &mut [f64] {
        &mut self.heights
    }

    /// Deposits a periodic 2D Gaussian mound centered at `(cx, cy)` without heap allocation.
    #[verified_engine::verified]
    pub fn deposit_mound(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        assert!(amplitude.is_finite(), "Mound amplitude must be finite");
        let w = self.width;
        let h = self.height;
        let tx = cx % w;
        let ty = cy % h;
        let r = radius.max(1) as f64;
        let cutoff_sq = 9.0 * r * r;
        let inv_two_r_sq = 0.5 / (r * r);

        for y in 0..h {
            let ry = (y as isize - ty as isize).unsigned_abs();
            let dy = ry.min(h - ry) as f64;
            for x in 0..w {
                let rx = (x as isize - tx as isize).unsigned_abs();
                let dx = rx.min(w - rx) as f64;
                let d2 = dx * dx + dy * dy;
                if d2 <= cutoff_sq {
                    let idx = flatten_2d_index(x, y, w);
                    self.heights[idx] += amplitude * (-d2 * inv_two_r_sq).exp();
                }
            }
        }
    }

    /// Advances the 2D heightfield by one Euler-Maruyama step with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, rng: &mut OxidizeRng) {
        let total = self.width * self.height;
        let cell_area = self.params.dx * self.params.dx;
        let noise_scale = self.params.noise_sigma * (self.params.dt / cell_area).sqrt();
        let noise = self.noise_buf.fill_gaussian(rng, total, noise_scale);
        step_2d_periodic_kernel(
            self.width,
            self.height,
            &self.heights,
            &mut self.next_heights,
            noise,
            &self.ops_xy,
            &self.ops_y,
            &self.params,
        );
        std::mem::swap(&mut self.heights, &mut self.next_heights);
        self.time += self.params.dt;
        self.steps += 1;
    }

    /// Advances the 2D heightfield by `n_steps` Euler-Maruyama steps with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_n(&mut self, rng: &mut OxidizeRng, n_steps: usize) {
        for _ in 0..n_steps {
            self.step(rng);
        }
    }

    /// Spatial mean height $\bar{h}(t)$.
    #[verified_engine::verified]
    pub fn mean_height(&self) -> f64 {
        compute_mean(&self.heights)
    }

    /// RMS interface width $W(L, t) = \sqrt{\langle (h(x, y, t) - \bar{h}(t))^2 \rangle}$.
    #[verified_engine::verified]
    pub fn interface_width(&self) -> f64 {
        compute_rms_width(&self.heights)
    }
}

#[allow(clippy::too_many_arguments)]
fn step_2d_periodic_kernel(
    width: usize,
    height: usize,
    src: &[f64],
    dst: &mut [f64],
    noise: &[f64],
    ops_xy: &StencilOperators,
    ops_y: &StencilOperators,
    params: &InterfaceGrowthParams,
) {
    let dx_sq = params.dx * params.dx;
    let twelfth_dx_fourth = (dx_sq * dx_sq) / 12.0;
    let half_lambda = 0.5 * params.lambda;
    let tame_coeff = params.dt * half_lambda.abs() * ops_xy.inv_dx;
    let dt = params.dt;
    let nu = params.nu;

    for y in 0..height {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let xp = if x + 1 == width { 0 } else { x + 1 };

            let idx = flatten_2d_index(x, y, width);
            let curr = src[idx];
            let left = src[flatten_2d_index(xm, y, width)];
            let right = src[flatten_2d_index(xp, y, width)];
            let up = src[flatten_2d_index(x, ym, width)];
            let down = src[flatten_2d_index(x, yp, width)];

            let lap = ops_xy.central_diff_2nd_2d(curr, left, right, up, down);
            let gx = ops_xy.central_diff_1st(left, right);
            let d2x = ops_xy.central_diff_2nd(left, curr, right);
            let gy = ops_y.central_diff_1st(up, down);
            let d2y = ops_y.central_diff_2nd(up, curr, down);

            let grad_sq_ls = (gx * gx + twelfth_dx_fourth * d2x * d2x)
                + (gy * gy + twelfth_dx_fourth * d2y * d2y);
            let nl_tamed = (half_lambda * grad_sq_ls) / (1.0 + tame_coeff * grad_sq_ls);
            let rhs = nu * lap + nl_tamed;
            dst[idx] = curr + dt * rhs + noise[idx];
        }
    }
}
