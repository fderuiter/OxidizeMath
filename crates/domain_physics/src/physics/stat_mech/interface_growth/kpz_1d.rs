//! Zero-allocation 1D Edwards-Wilkinson ($\lambda = 0$) and Kardar-Parisi-Zhang ($\lambda \ne 0$)
//! stochastic interface growth solver built on [`engine_pde::StencilOperators`] and
//! [`engine_pde::inject_additive_noise_1d`].

// [cite:interface_growth]

use crate::physics::stat_mech::edwards_wilkinson::{interface_roughness, mean_field};
use engine_pde::{BoundaryCondition, StencilOperators, inject_additive_noise_1d};
use math_commons::math_kernel::types::{SolverWorkspace, StepSize};
use oxidize_core::rng::OxidizeRng;

/// Continuum and numerical parameters for 1D/2D Edwards-Wilkinson and KPZ interface evolution:
/// $\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2} |\nabla h|^2 + \sigma \eta(\mathbf{x}, t)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceGrowthParams {
    /// Surface tension / diffusive smoothing coefficient $\nu > 0$.
    pub nu: f64,
    /// Lateral KPZ nonlinear growth coupling $\lambda$ ($\lambda = 0$ selects Edwards-Wilkinson).
    pub lambda: f64,
    /// Uncorrelated Gaussian white-noise amplitude $\sigma \ge 0$.
    pub noise_sigma: f64,
    /// Spatial lattice spacing $\Delta x > 0$.
    pub dx: f64,
    /// Explicit Euler-Maruyama time step $\Delta t > 0$.
    pub dt: f64,
}

impl Default for InterfaceGrowthParams {
    fn default() -> Self {
        Self {
            nu: 0.5,
            lambda: 3.2,
            noise_sigma: 1.0,
            dx: 1.0,
            dt: 0.02,
        }
    }
}

/// Pre-allocated 1D periodic stochastic interface solver using Lam-Shin conservative KPZ stencils.
#[derive(Clone, Debug)]
pub struct InterfaceGrowth1D {
    len: usize,
    params: InterfaceGrowthParams,
    ops: StencilOperators,
    heights: Vec<f64>,
    next_heights: Vec<f64>,
    noise_scratch: Vec<f64>,
    time: f64,
    steps: u64,
}

impl InterfaceGrowth1D {
    /// Constructs a flat ($h_i = 0$) 1D periodic interface of `len` sites with pre-allocated buffers.
    #[verified_engine::verified]
    pub fn new(len: usize, params: InterfaceGrowthParams) -> Self {
        assert!(len >= 4, "1D interface lattice length must be >= 4");
        validate_interface_params(&params);
        let ops = StencilOperators::new(StepSize(params.dx));
        Self {
            len,
            params,
            ops,
            heights: vec![0.0; len],
            next_heights: vec![0.0; len],
            noise_scratch: vec![0.0; len],
            time: 0.0,
            steps: 0,
        }
    }

    /// Resets the height profile to a flat interface $h_i = 0$ and clears elapsed time.
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
            self.ops = StencilOperators::new(StepSize(params.dx));
        }
        self.params = params;
    }

    /// Number of 1D periodic lattice sites $L$.
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always `false` because valid lattices contain at least 4 sites.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.len == 0
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

    /// Immutable slice of the current height profile $h(x, t)$.
    #[verified_engine::verified]
    pub fn heights(&self) -> &[f64] {
        &self.heights
    }

    /// Mutable slice of the current height profile for custom initial conditions.
    pub fn heights_mut(&mut self) -> &mut [f64] {
        &mut self.heights
    }

    /// Deposits a localized Gaussian mound centered at `center` without heap allocation.
    #[verified_engine::verified]
    pub fn deposit_mound(&mut self, center: usize, radius: usize, amplitude: f64) {
        assert!(amplitude.is_finite(), "Mound amplitude must be finite");
        let n = self.len;
        let c = center % n;
        let r = radius.max(1) as f64;
        let inv_two_r_sq = 0.5 / (r * r);
        for (i, h) in self.heights.iter_mut().enumerate() {
            let raw = (i as isize - c as isize).unsigned_abs();
            let dist = raw.min(n - raw) as f64;
            if dist <= 3.0 * r {
                *h += amplitude * (-dist * dist * inv_two_r_sq).exp();
            }
        }
    }

    /// Advances the 1D interface by a single Euler-Maruyama step with zero heap allocations.
    #[verified_engine::verified]
    pub fn step(&mut self, rng: &mut OxidizeRng) {
        step_1d_periodic_kernel(
            &self.heights,
            &mut self.next_heights,
            &self.ops,
            &self.params,
        );
        let intensity = 0.5 * self.params.noise_sigma * self.params.noise_sigma;
        if intensity > 0.0 {
            let mut ws = SolverWorkspace::new(&mut self.noise_scratch);
            inject_additive_noise_1d(
                &mut self.next_heights,
                self.params.dx,
                self.params.dt,
                intensity,
                BoundaryCondition::Periodic,
                &mut ws,
                rng,
            );
        }
        std::mem::swap(&mut self.heights, &mut self.next_heights);
        self.time += self.params.dt;
        self.steps += 1;
    }

    /// Advances the 1D interface by `n_steps` Euler-Maruyama steps with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_n(&mut self, rng: &mut OxidizeRng, n_steps: usize) {
        for _ in 0..n_steps {
            self.step(rng);
        }
    }

    /// Spatial mean interface height $\bar{h}(t) = \frac{1}{L}\sum_{i=0}^{L-1} h_i(t)$.
    #[verified_engine::verified]
    pub fn mean_height(&self) -> f64 {
        mean_field(&self.heights)
    }

    /// RMS interface width (roughness) $W(L, t) = \sqrt{\frac{1}{L}\sum_{i=0}^{L-1} (h_i(t) - \bar{h}(t))^2}$.
    #[verified_engine::verified]
    pub fn interface_width(&self) -> f64 {
        interface_roughness(&self.heights)
    }
}

pub(crate) fn validate_interface_params(params: &InterfaceGrowthParams) {
    assert!(
        params.nu.is_finite() && params.nu > 0.0,
        "Surface tension nu must be finite and positive"
    );
    assert!(
        params.lambda.is_finite(),
        "KPZ coupling lambda must be finite"
    );
    assert!(
        params.noise_sigma.is_finite() && params.noise_sigma >= 0.0,
        "Noise amplitude sigma must be finite and non-negative"
    );
    assert!(
        params.dx.is_finite() && params.dx > 0.0,
        "Grid spacing dx must be finite and positive"
    );
    assert!(
        params.dt.is_finite() && params.dt > 0.0,
        "Time step dt must be finite and positive"
    );
}

fn step_1d_periodic_kernel(
    src: &[f64],
    dst: &mut [f64],
    ops: &StencilOperators,
    params: &InterfaceGrowthParams,
) {
    let n = src.len();
    let dx_sq = params.dx * params.dx;
    let twelfth_dx_fourth = (dx_sq * dx_sq) / 12.0;
    let half_lambda = 0.5 * params.lambda;
    let tame_coeff = params.dt * half_lambda.abs() * ops.inv_dx;
    let dt = params.dt;
    let nu = params.nu;

    for i in 0..n {
        let prev = src[if i == 0 { n - 1 } else { i - 1 }];
        let curr = src[i];
        let next = src[if i + 1 == n { 0 } else { i + 1 }];

        let lap = ops.central_diff_2nd(prev, curr, next);
        let grad = ops.central_diff_1st(prev, next);
        // Lam-Shin conservative gradient-squared identity via StencilOperators + tamed Euler denominator:
        let grad_sq_ls = grad * grad + twelfth_dx_fourth * lap * lap;
        let nl_tamed = (half_lambda * grad_sq_ls) / (1.0 + tame_coeff * grad_sq_ls);
        let rhs = nu * lap + nl_tamed;
        dst[i] = curr + dt * rhs;
    }
}
